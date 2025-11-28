import torch
from train import MLP
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import scanpy as sc
import pandas as pd
import argparse
import joblib
import os

import hydra
from omegaconf import OmegaConf, DictConfig

def load_model(model_type, model_path):

    if model_type == 'lasso':
        model = joblib.load(model_path)
    elif model_type == 'random_forest':
        model = joblib.load(model_path)
    elif model_type == 'mlp':
        model = MLP(input_size=2000, hidden_size=64, output_size=1, dropout_rate=0.5)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        raise ValueError("Invalid model type. Choose from: lasso, random_forest, mlp")
    return model

def predict(model, X):
    if isinstance(model, MLP):
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32)
            outputs = model(inputs).squeeze().numpy()
            predictions = (outputs >= 0.5).astype(int)
    else:
        predictions = model.predict(X)
    return predictions

def predict_proba(model, X):
    if isinstance(model, MLP):
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32)
            outputs = model(inputs).squeeze().numpy()
            probabilities = torch.sigmoid(torch.tensor(outputs)).numpy()
    else:
        probabilities = model.predict_proba(X)[:, 1]
    return probabilities

def encode_labels(labels):
    labels = labels.apply(lambda x: 'gdt' if isinstance(x, str) and 'gdt' in x.lower() else 'other')
    labels = labels.astype('category')
    print("\nCell Type Labels")
    print(labels.value_counts())

    # set gdt as positive class (1), other as negative class (0)
    encoded_labels = labels.cat.codes.replace({labels.cat.categories.get_loc('gdt'): 1, labels.cat.categories.get_loc('other'): 0}).values
    return encoded_labels

def slience_genes(adata, model_adata, selected_genes):
    # slience selected_genes by using median expression in model_adata
    gene_indices = [adata.var_names.get_loc(gene) for gene in selected_genes if gene in adata.var_names]
    median_expression = model_adata[:, selected_genes].X.mean(axis=0)
    for i, gene_idx in enumerate(gene_indices):
        adata.X[:, gene_idx] = median_expression[i]
    return adata

def align_genes(adata, model_adata, model_genes):
    # use median to fill missing genes in adata
    missing_genes = list(set(model_genes) - set(adata.var_names))
    if missing_genes:
        adata = slience_genes(adata, model_adata, missing_genes)
    # ensure adata has the same genes order as model_genes
    adata = adata[:, model_genes]
    return adata

def intergrate_to_reference(adata, model_adata):
    # find common genes
    pass
    return adata

# @hydra.main(version_base=None, config_path="./", config_name="config.yaml")
def main():
    # cfg = OmegaConf.to_container(cfg, resolve=True)
    # cfg = OmegaConf.create(cfg)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='random_forest', help='Type of model: lasso, random_forest, mlp')
    parser.add_argument('--model_path', type=str, default='model_output', help='Path to the trained model file')
    parser.add_argument('--input_data', type=str, default='data/processed/pbmc_4k8k.h5ad', help='Path to input data file (e.g., h5ad)')
    parser.add_argument('--output_path', type=str, default='evaluation_results', help='Path to save prediction results')
    parser.add_argument('--model_data', type=str, default='./data/processed/pbmc_4k8k.h5ad', help='Path to model data file (e.g., h5ad) for gene alignment')
    parser.add_argument('--silence_genes', type=str, default=None, help='Path to file with genes to silence')
    args = parser.parse_args()

    # Load model
    if args.model_type in ['lasso', 'random_forest']:
        model_path = args.model_path+'/' + args.model_type + '_final_model.pkl'
    else:
        model_path = args.model_path+'/' + args.model_type + '_final_model.pt'
    prediction_path = args.output_path + '/' + args.model_type + '_predictions.csv'
    os.makedirs(args.output_path, exist_ok=True)
    performance_path = args.output_path + '/' + args.model_type + '_performance.txt'
    os.makedirs(args.output_path, exist_ok=True)
    model = load_model(args.model_type, model_path)
    # Load data
    adata = sc.read_h5ad(args.input_data)

    # Align genes
    # adata = align_genes(adata, model_adata, model.var_names if hasattr(model, 'var_names') else model_adata.var_names)

    if args.silence_genes:
        model_adata = sc.read_h5ad(args.model_data)
        with open(args.silence_genes, 'r') as f:
            genes_to_silence = [line.strip() for line in f]
        adata = slience_genes(adata, model_adata, genes_to_silence)

    # Encode labels
    y = encode_labels(adata.obs['celltype'])

    # Predict
    if args.model_type == 'mlp':
        with torch.no_grad():
            adata_tensor = torch.tensor(adata.X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            logits = model(adata_tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy().ravel()
            y = y.cpu().numpy().ravel()
            predictions = (probabilities > 0.5).astype(int)
    else:
        predictions = predict(model, adata.X)
        probabilities = predict_proba(model, adata.X)
    # evaluate
    accuracy = (predictions == y).mean()
    auc = roc_auc_score(y, probabilities)
    print(f"Prediction Accuracy: {accuracy}")
    print(f"Prediction AUC: {auc}")
    # Save performance
    with open(performance_path, 'w') as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"AUC: {auc}\n")
    # Save results
    results_df = pd.DataFrame({'Prediction': predictions, 'Probability': probabilities})
    results_df.to_csv(prediction_path, index=False)

if __name__ == '__main__':
    main()