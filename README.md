# Machine Learning Pipeline for  Gamma-delta T cells (GDT) from Single Cell RNA-seq PBMC dataset


## Data Preparation

### 1. Prepare PBMC 3p/5p and purified data (`prep_data.ipynb`)
**Purpose**: Load raw data, clean PBMC datasets, and perform quality control

**Steps**:
- Load raw single-cell data into AnnData format
- Clean PBMC datasets by removing cells similar to GDT (close-to-gdt cell types)
- Perform quality control filtering on all datasets:
  - Filter low-quality cells and genes
  - Calculate QC metrics (mitochondrial content, gene counts, etc.)
- Save processed data to `data/processed/`

**Output**: QC-filtered `.h5ad` files for PBMC 3p/5p and all purified datasets

### 2. Manual PBMC 4k/8k annotation (`annotate_pbmc4k8k.ipynb`)
**Purpose**: Add manual cell type annotations and clean PBMC 4k/8k datasets

**Steps**:
- Add manual cell type annotations to PBMC 4k and 8k datasets
- Additional QC cleanup to remove cells close to GDT phenotype
- Ensure high-quality training data

**Output**: Annotated and cleaned PBMC 4k/8k datasets

### 3. Integration (`integrate.ipynb`)
**Purpose**: Train scVI model and align datasets in shared latent space

**Steps**:
- **Prepare datasets**:
  - Training set: PBMC 4k, 8k, CD4T, CD8T, NK, GDT
  - Validation set: PBMC 3p, 5p, CD4T, CD8T, NK, GDT
- **Train scVI model** on training set (4k/8k):
  - Learn 30-dimensional latent representation
  - Correct for batch effects across datasets
  - Save trained model to `data/ready/scvi_model/`
- **Align validation set** (3p/5p) to trained model:
  - Load pretrained model
  - Handle gene overlap (zero-pad missing genes)
  - Fine-tune query model on validation data
  - Project validation data into same latent space

**Outputs**:
- `X_scVI`: Low-dimensional embeddings (30D) for visualization and clustering
- `scVI_corrected`: Batch-corrected gene expression (high-dimensional)
- `scVI_corrected_log`: Log-transformed corrected expression
- Saved files:
  - `data/ready/pbmc_4k8k_clean_gdt_scvi_integrated.h5ad` (training set)
  - `data/ready/pbmc_3p5p_clean_gdt_scvi_integrated.h5ad` (validation set)
  - `data/ready/scvi_model/` (trained model)

### 4. Testing (`test_data.ipynb`)
**Purpose**: Validate that scVI embeddings map training and validation sets correctly

**Steps**:
- Load integrated training and validation datasets
- Compute UMAP projections from X_scVI embeddings
- Visualize combined datasets to verify:
  - Training and validation cells intermix (same latent space)
  - Same cell types cluster together across datasets
  - Batch effects are corrected
  - GDT cells map to consistent regions

**Key Visualizations**:
- Train vs Validation split (should intermix)
- Cell type clustering (should align across splits)
- Dataset/batch distribution (should be integrated)
- GDT vs non-GDT separation

## Data Structure

```
data/
├── raw/                          # Raw input data
│   ├── pbmc_4k/, pbmc_8k/       # 10X PBMC datasets
│   ├── cd4t/, cd8t_donor1/, cd8t_donor2/
│   ├── nk/
│   └── gdt_3donors.tsv
├── processed/                    # QC-filtered datasets
│   ├── pbmc_4k_clean_qc.h5ad
│   ├── pbmc_8k_clean_qc.h5ad
│   ├── pbmc_3p_clean_qc.h5ad
│   ├── pbmc_5p_clean_qc.h5ad
│   ├── cd4t_qc.h5ad
│   ├── cd8t_d1_qc.h5ad
│   ├── nk_qc.h5ad
│   └── gdt_3donors_qc.h5ad
└── ready/                        # Integration outputs
    ├── pbmc_4k8k_clean_gdt_scvi_integrated.h5ad    # Training set
    ├── pbmc_3p5p_clean_gdt_scvi_integrated.h5ad    # Validation set
    └── scvi_model/               # Trained scVI model
```

## Dependencies

- Python 3.11+
- scanpy 1.11.5
- scvi-tools 1.4.0+
- numpy
- pandas
- matplotlib

## Usage

Run notebooks in order:
```bash
1. prep_data.ipynb
2. annotate_pbmc4k8k.ipynb
3. integrate.ipynb
4. test_data.ipynb
```

## Helper Functions (`data_functions.py`)

- `quality_control()`: Perform QC filtering on single-cell data
- `balance_datasets()`: Downsample datasets for balanced training
- `integrate_scvi()`: Train scVI model on multiple datasets
- `align_to_scvi_model()`: Align query data to pretrained scVI model
- `save_h5ad()`: Save AnnData objects to file

## Notes

- Gene overlap between training and validation: 409/451 genes (90.7%)
- Missing genes in validation set are zero-padded
- scVI model trained with 30 latent dimensions, 200 epochs
- Query model fine-tuned for 200 epochs on validation data
