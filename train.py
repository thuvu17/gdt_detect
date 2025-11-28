import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
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
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5): # 增加 dropout参数
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate) # 定义 Dropout

    def forward(self, x):
        out = self.dropout(self.relu(self.fc1(x))) # 使用 Dropout
        out = self.dropout(self.relu(self.fc2(out))) # 使用 Dropout
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

    # set gdt as positive class (1), other as negative class (0)
    y = labels.cat.codes.replace({labels.cat.categories.get_loc('gdt'): 1, labels.cat.categories.get_loc('other'): 0}).values

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

        model = MLP(input_size=X.shape[1], hidden_size=config.model.hidden, output_size=1, dropout_rate=config.model.dropout).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        early_stopping_counter = 0
        best_auc = -np.inf

        for epoch in range(epochs):
            model.train()
            epoch_losses = []
            epoch_accuracy = 0.0
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
                epoch_accuracy += ((torch.sigmoid(logits) > 0.5) == y_batch).float().mean().item()
                tqdm_loader.set_postfix(loss=loss.item(), accuracy=epoch_accuracy / (len(tqdm_loader)))
                tqdm_loader.update()

            # evaluate on validation/test split of this fold
            model.eval()
            with torch.no_grad():
                X_test_device = X_test.to(device)
                test_logits = model(X_test_device)
                test_probs = torch.sigmoid(test_logits).cpu().numpy().ravel()
                y_test_np_arr = y_test.cpu().numpy().ravel()
                val_auc = roc_auc_score(y_test_np_arr, test_probs)
                
                predicted = (test_probs > 0.5).astype(int)
                val_accuracy = (predicted == y_test_np_arr).mean()


            wandb.log({
                f"Fold_{fold_count}_Train/Loss": np.mean(epoch_losses) if epoch_losses else None,
                f"Fold_{fold_count}_Train/Accuracy": epoch_accuracy / (len(tqdm_loader)),
                f"Fold_{fold_count}_Val/AUC": val_auc,
                f"Fold_{fold_count}_Val/Accuracy": val_accuracy,
                f"Fold_{fold_count}_Epoch": epoch
            })

            # early stopping based on AUC
            if not np.isnan(val_auc) and val_auc > best_auc:
                best_auc = val_auc
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
            auc = roc_auc_score(y_test_np_arr, test_probs)

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