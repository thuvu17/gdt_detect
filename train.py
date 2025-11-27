import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import warnings
import joblib
import os
from copy import deepcopy
from sklearn.base import clone
from torch.utils.data import DataLoader, TensorDataset

import hydra
from omegaconf import OmegaConf, DictConfig
import wandb

warnings.filterwarnings('ignore')

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        # For binary classification we'll return logits (no final activation)
        # and use BCEWithLogitsLoss which combines a sigmoid + binary CE in a
        # numerically stable way. For multiclass, change output_size and loss.

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
def load_data(file_path):
    # convert to dataframes for easier handling
    adata=sc.read_h5ad(file_path)
    X_df = pd.DataFrame(adata.X, columns=adata.var_names)

    print("Expression Matrix (X)")
    print(X_df.head())

    labels = adata.obs['celltype']

    # change to binary labels: any celltype containing 'gdt' -> 'gdt', else 'other'
    # This avoids accidental multi-class labels when there are multiple gdt subtypes.
    labels = labels.apply(lambda x: 'gdt' if isinstance(x, str) and 'gdt' in x.lower() else 'other')
    labels = labels.astype('category')

    print("\nCell Type Labels")
    print(labels.value_counts())

    le = LabelEncoder()
    y = le.fit_transform(labels)

    return adata, y

def load_wandb(cfg):
    """Initialize wandb logging."""
    wandb_run = wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    return wandb_run

def five_fold_cv(model, X, y):
    # separate data into 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    aucs = []
    best_model = None
    best_auc = -np.inf
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # clone model to avoid reusing the same fitted instance across folds
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        acc = model_clone.score(X_test, y_test)
        # compute AUC robustly (predict_proba preferred; fallback to decision_function)
        try:
            probs = model_clone.predict_proba(X_test)
            # if binary, take column 1
            if probs.shape[1] == 2:
                auc = roc_auc_score(y_test, probs[:, 1])
            else:
                # if multiclass, compute macro-average over classes (sklearn handles multiclass)
                auc = roc_auc_score(y_test, probs, multi_class='ovr')
        except Exception:
            try:
                scores = model_clone.decision_function(X_test)
                auc = roc_auc_score(y_test, scores)
            except Exception:
                auc = np.nan
        accs.append(acc)
        aucs.append(auc)
        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            best_model = deepcopy(model_clone)
    return best_model, np.mean(accs), np.nanmean(aucs)

def LASSO_train(X, y):
    # Use LogisticRegression with L1 penalty; cross-validation is handled by five_fold_cv.
    lasso = LogisticRegression(penalty='l1', solver='saga', max_iter=100)
    best_model, cv_acc, cv_auc = five_fold_cv(lasso, X, y)
    print(f"LASSO CV Accuracy: {cv_acc}, CV AUC: {cv_auc}")
    return best_model

def RF_train(X, y, n_estimators=100, random_state=42):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    best_model, cv_acc, cv_auc = five_fold_cv(rf, X, y)
    print(f"Random Forest CV Accuracy: {cv_acc}, CV AUC: {cv_auc}")
    return best_model

def MLP_train(config, X, y):

    load_wandb(config)

    # Use BCEWithLogitsLoss and mini-batch training with DataLoader.
    epochs = config.model.num_epochs
    batch_size = config.model.batch_size
    learning_rate = config.model.learning_rate
    early_stop_patience = config.model.early_stop_patience
    output_path = config.general.output_path
    fold_path = os.path.join(output_path, "folds")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_count = 1
    fold_accuracies = []
    fold_aucs = []
    fold_best_model = None
    best_fold_auc = -np.inf

    X_arr = np.array(X)
    y_arr = np.array(y)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for train_index, test_index in kf.split(X_arr):
        X_train_np, X_test_np = X_arr[train_index], X_arr[test_index]
        y_train_np, y_test_np = y_arr[train_index], y_arr[test_index]

        # convert to tensors
        X_train = torch.FloatTensor(X_train_np)
        X_test = torch.FloatTensor(X_test_np)
        y_train = torch.FloatTensor(y_train_np).unsqueeze(1)
        y_test = torch.FloatTensor(y_test_np).unsqueeze(1)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = MLP(input_size=X.shape[1], hidden_size=config.model.hidden, output_size=1).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        early_stopping_counter = 0
        best_auc = -np.inf

        for epoch in range(epochs):
            model.train()
            epoch_losses = []
            tqdm_loader = tqdm(train_loader, desc=f"Fold {fold_count} Epoch {epoch+1}/{epochs}")
            for X_batch, y_batch in tqdm_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                tqdm_loader.set_postfix(loss=loss.item())
                tqdm_loader.update()

            # evaluate on validation/test split of this fold
            model.eval()
            with torch.no_grad():
                X_test_device = X_test.to(device)
                test_logits = model(X_test_device)
                test_probs = torch.sigmoid(test_logits).cpu().numpy().ravel()
                y_test_np_arr = y_test.cpu().numpy().ravel()
                try:
                    auc = roc_auc_score(y_test_np_arr, test_probs)
                except Exception:
                    auc = np.nan

            wandb.log({
                f"Fold_{fold_count}_Train/Loss": np.mean(epoch_losses) if epoch_losses else None,
                f"Fold_{fold_count}_Val/AUC": auc,
                f"Fold_{fold_count}_Epoch": epoch
            })

            # early stopping based on AUC
            if not np.isnan(auc) and auc > best_auc:
                best_auc = auc
                early_stopping_counter = 0
                # save best weights for this fold
                os.makedirs(fold_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(fold_path, f"mlp_best_model_fold_{fold_count}.pt"))
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch} for fold {fold_count}")
                break

        # load best model for this fold and evaluate final metrics
        model.load_state_dict(torch.load(os.path.join(fold_path, f"mlp_best_model_fold_{fold_count}.pt")))
        model.to(device)
        model.eval()
        with torch.no_grad():
            X_test_device = X_test.to(device)
            test_logits = model(X_test_device)
            test_probs = torch.sigmoid(test_logits).cpu().numpy().ravel()
            y_test_np_arr = y_test.cpu().numpy().ravel()
            predicted = (test_probs > 0.5).astype(int)
            accuracy = (predicted == y_test_np_arr).mean()
            try:
                auc = roc_auc_score(y_test_np_arr, test_probs)
            except Exception:
                auc = np.nan

        print(f"Fold {fold_count} - MLP Accuracy: {accuracy}, AUC: {auc}")
        fold_accuracies.append(accuracy)
        fold_aucs.append(auc)

        if not np.isnan(auc) and auc > best_fold_auc:
            best_fold_auc = auc
            # keep loaded model (already contains best fold weights)
            fold_best_model = deepcopy(model)

        fold_count += 1

    # save best overall model
    if fold_best_model is not None:
        torch.save(fold_best_model.state_dict(), os.path.join(output_path, "mlp_best_overall_model.pt"))

    return fold_best_model, np.mean(fold_accuracies), np.nanmean(fold_aucs)

# main function
@hydra.main(version_base=None, config_path="./", config_name="config.yaml")
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)

    np.random.seed(42)
    torch.manual_seed(42)

    print("\nLoading data...")
    adata, labels = load_data(cfg.data.data_path)
    # ensure output directory exists early
    os.makedirs(cfg.general.output_path, exist_ok=True)
    print("\nTraining model...")
    if cfg.model.model_choice == 'lasso':
        model = LASSO_train(adata.X, labels)
    elif cfg.model.model_choice == 'random_forest':
        model = RF_train(adata.X, labels)
    elif cfg.model.model_choice == 'mlp':
        model, cv_acc, cv_auc = MLP_train(cfg, adata.X, labels)
        print(f"MLP CV Accuracy: {cv_acc}, CV AUC: {cv_auc}")
    else:
        raise ValueError("Invalid model type. Choose from: lasso, random_forest, mlp")

    if cfg.model.model_choice in ['lasso', 'random_forest']:
        print("Extracting selected genes...")
        # feature selection based on model coefficients or feature importances
        if cfg.model.model_choice == 'lasso':
            coefs = model.coef_[0]
            selected_genes = adata.var_names[coefs != 0]
            # print selected genes
            print(f"{len(selected_genes)} genes selected.")
            # save selected genes to a file
            with open(cfg.general.output_path + "selected_genes_lasso.txt", "w") as f:
                for gene in selected_genes:
                    f.write(gene + "\n")
        elif cfg.model.model_choice == 'random_forest':
            importances = model.feature_importances_
            selected_genes = adata.var_names[importances > np.percentile(importances, 95)]
            # print selected genes
            print(f"Selected genes ({len(selected_genes)}): {selected_genes.tolist()}")
            # save selected genes to a file
            with open(cfg.general.output_path + "selected_genes_rf.txt", "w") as f:
                for gene in selected_genes:
                    f.write(gene + "\n")
        print("\nSaving trained model...")
        # save the trained model
        joblib.dump(model, cfg.general.output_path + f"{cfg.model.model_choice}_final_model.pkl")
        print(f"Trained {cfg.model.model_choice} model saved to {cfg.general.output_path + f'{cfg.model.model_choice}_final_model.pkl'}")

    if cfg.model.model_choice == 'mlp':
        print("\nSaving trained MLP model...")
        # save the trained MLP model weights
        torch.save(model.state_dict(), cfg.general.output_path + "mlp_final_model.pt")
        print(f"Trained MLP model saved to {cfg.general.output_path + 'mlp_final_model.pt'}")
    print("\nTraining completed.")

if __name__ == "__main__":
    main()

# # use LASSO regression, random forests, and autoencoders to do feature selection

# # LaSSO Regression
# le = LabelEncoder()
# y = le.fit_transform(labels)
# X_train, X_test, y_train, y_test = train_test_split(adata.X, y, test_size=0.2, random_state=42, stratify=y)
# lasso = LassoCV(cv=5).fit(X_train, y_train)
# lasso_coef = pd.Series(lasso.coef_, index=adata.var_names)
# selected_genes_lasso = lasso_coef[lasso_coef != 0].index.tolist
# print(f"\nSelected genes by LASSO: {selected_genes_lasso}")
# # print("Selected genes count:", len(selected_genes_lasso))

# # test classification accuracy
# y_pred = lasso.predict(X_test)
# y_pred_class = (y_pred > 0.5).astype(int)
# accuracy = accuracy_score(y_test, y_pred_class)
# print(f"LASSO Classification Accuracy: {accuracy}")


# # Random Forests    
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)
# importances = rf.feature_importances_
# rf_importances = pd.Series(importances, index=adata.var_names)
# selected_genes_rf = rf_importances[rf_importances > np.percentile(rf_importances, 90)].index.tolist()
# # print(f"\nSelected genes by Random Forests: {selected_genes_rf}")
# print("Selected genes count:", len(selected_genes_rf))

# # test classification accuracy
# y_pred_rf = rf.predict(X_test)
# accuracy_rf = accuracy_score(y_test, y_pred_rf)
# print(f"Random Forests Classification Accuracy: {accuracy_rf}")

# # Autoencoder
# input_dim = adata.X.shape[1]
# encoding_dim = 50  # dimension of encoded representation
# input_layer = Input(shape=(input_dim,))
# encoded = Dense(encoding_dim, activation='relu')(input_layer)
# decoded = Dense(input_dim, activation='sigmoid')(encoded)
# autoencoder = Model(input_layer, decoded)
# autoencoder.compile(optimizer='adam', loss='mean_squared_error')
# autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_split=0.2, verbose=0)
# # Get weights of the first layer
# weights = autoencoder.layers[1].get_weights()[0]
# weight_importance = pd.Series(np.mean(np.abs(weights), axis=1), index=adata.var_names)
# selected_genes_ae = weight_importance[weight_importance > np.percentile(weight_importance, 90)].index.tolist()
# # print(f"\nSelected genes by Autoencoder: {selected_genes_ae}")
# print("Selected genes count:", len(selected_genes_ae))

# # test classification accuracy using a simple dense network
# encoded_input = Input(shape=(encoding_dim,))
# decoded_output = autoencoder.layers[-1](encoded_input)
# encoder = Model(input_layer, encoded)
# X_train_encoded = encoder.predict(X_train)
# X_test_encoded = encoder.predict(X_test)
# classifier_input = Input(shape=(encoding_dim,))
# classifier_output = Dense(len(np.unique(y)), activation='softmax')(classifier_input)
# classifier = Model(classifier_input, classifier_output)
# classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# classifier.fit(X_train_encoded, y_train, epochs=50, batch_size=256, shuffle=True, validation_split=0.2, verbose=0)
# loss, accuracy_ae = classifier.evaluate(X_test_encoded, y_test, verbose=0)
# print(f"Autoencoder Classification Accuracy: {accuracy_ae}")

# # use three layer MLP to do classification
# input_dim = adata.X.shape[1]
# model = MLP(input_size=input_dim, hidden_size=100, output_size=len(np.unique(y)))
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# X_tensor = torch.FloatTensor(adata.X)
# y_tensor = torch.LongTensor(y)

# tqdm_epochs = tqdm(range(50), desc="Training MLP")

# # training loop
# for epoch in tqdm_epochs:
#     model.train()
#     optimizer.zero_grad()
#     outputs = model(X_tensor)
#     loss = criterion(outputs, y_tensor)
#     loss.backward()
#     optimizer.step()
#     tqdm_epochs.set_postfix(loss=loss.item())
#     tqdm_epochs.update()

# # evaluate
# model.eval()
# with torch.no_grad():
#     outputs = model(X_tensor)
#     _, predicted = torch.max(outputs.data, 1)
#     accuracy_mlp = (predicted == y_tensor).sum().item() / y_tensor.size(0)
# print(f"MLP Classification Accuracy: {accuracy_mlp}")

# # print summary model performance
# print("\nModel Performance Summary:")
# print(f"LASSO Accuracy: {accuracy}")
# print(f"Random Forests Accuracy: {accuracy_rf}")
# print(f"Autoencoder Accuracy: {accuracy_ae}")
# print(f"MLP Accuracy: {accuracy_mlp}")