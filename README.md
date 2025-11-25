# CompMed: Computational Medicine Project

Single-cell RNA-seq analysis and integration of multiple immune cell datasets.

## Project Structure

- `data/` - Raw and processed single-cell datasets
  - `processed/` - QC-filtered h5ad files
  - `pbmc_2_batch/` - PBMC batch datasets
- `load_data.ipynb` - Data loading and exploration
- `prep_data.ipynb` - Data preprocessing, integration, and export
- `integrate_data.R` - R/Seurat integration workflow
- `compmed_env/` - Python virtual environment (not tracked)

## Datasets

- **PBMC**: Peripheral blood mononuclear cells (2 batches)
- **CD4 T cells**: CD4+ T lymphocytes
- **CD8 T cells**: CD8+ T lymphocytes (multiple donors)
- **NK cells**: Natural killer cells
- **GDT cells**: Gamma-delta T cells (3 donors)

## Workflow

1. **Data Loading** (`load_data.ipynb`)
   - Load processed h5ad files
   - Preview cell metadata and gene information

2. **Data Preprocessing** (`prep_data.ipynb`)
   - Concatenate datasets
   - Downsample to balance cell types (max 900 cells per type)
   - Normalize and log-transform
   - Select highly variable genes
   - PCA and Harmony integration
   - UMAP visualization and clustering
   - Export training data (features + labels)

3. **R Integration** (`integrate_data.R`)
   - Alternative Seurat-based integration workflow

## Requirements

### Python
```bash
pip install scanpy anndata pandas numpy scipy h5py harmonypy scikit-misc leidenalg jupyter
```

### R
```r
install.packages(c("Seurat", "dplyr"))
remotes::install_github("mojaveazure/seurat-disk")
```

## Setup

```bash
# Create virtual environment
python -m venv compmed_env

# Activate environment
.\compmed_env\Scripts\Activate.ps1  # Windows
source compmed_env/bin/activate     # Linux/Mac

# Install packages
pip install scanpy anndata pandas numpy scipy h5py harmonypy scikit-misc leidenalg jupyter ipykernel
```

## Usage

Run notebooks in order:
1. `load_data.ipynb` - Explore the data
2. `prep_data.ipynb` - Process and integrate data

## Output Files

- `gdt_X_pca_harmony.csv.gz` - Feature matrix (Harmony-integrated PCA)
- `gdt_y_labels.csv.gz` - Cell type labels (including binary GDT labels)

## License

MIT License
