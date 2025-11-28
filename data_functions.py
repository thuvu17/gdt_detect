from pathlib import Path
import pandas as pd
import scanpy as sc
import numpy as np
# Delay scvi import to avoid OpenMP crash on module load


# Convert tsv file to AnnData
def convert_tsv_to_anndata(file_path, cell_type):
    """Convert a TSV file to an AnnData object."""
    df = pd.read_csv(file_path, sep='\t', index_col=0)
    adata = sc.AnnData(df.transpose())
    adata.obs['celltype'] = cell_type
    return adata


# Convert 10x matrix folder to AnnData
def convert_10x_to_anndata(folder_path, cell_type):
    """
    Load a 10x-format matrix folder into AnnData.
    Works for both compressed (.gz) and uncompressed files.
    """
    adata = sc.read_10x_mtx(
        folder_path,
        var_names="gene_symbols",
        make_unique=True
    )
    adata.obs["celltype"] = cell_type
    return adata


# Quality control function
def quality_control(
        adata,
        min_genes=200,
        max_genes=6000,
        min_cells=3,
        max_mt=10,
):
    adata_qc = adata.copy() 

    # Filter low-gene cells and rare genes
    sc.pp.filter_cells(adata_qc, min_genes=min_genes)
    sc.pp.filter_genes(adata_qc, min_cells=min_cells)
    # Mark mitochondrial genes (robust to case)
    adata_qc.var["mt"] = adata_qc.var_names.str.upper().str.startswith("MT-")

    # Compute QC metrics
    sc.pp.calculate_qc_metrics(
        adata_qc,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    # Filter out likely doublets / weird cells with too many genes
    adata_qc = adata_qc[adata_qc.obs["n_genes_by_counts"] < max_genes, :].copy()

    # Filter out high-mito cells
    adata_qc = adata_qc[adata_qc.obs["pct_counts_mt"] < max_mt, :].copy()

    # Print final cell count
    print(f"{adata_qc.obs['celltype'].unique()} cells before QC: {adata.n_obs}, after QC: {adata_qc.n_obs}")

    return adata_qc


# Helper function to save cleaned datasets
def save_h5ad(adata, filename, folder="data/processed"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    filepath = folder / filename
    adata.write_h5ad(filepath)
    print(f"Saved {filepath}")


# Balance datasets by sampling
def balance_datasets(adatas, max_cells_per_dataset):
    """
    Input:
    - adatas: dict of AnnData objects, keys are dataset names
    - max_cells_per_dataset: dict of maximum number of cells to sample per dataset,
        keys are the corresponding dataset names
    Output:
    - balanced_adatas: dict of AnnData objects after balancing
    - unused: 
    """
    balanced_adatas = {}
    unused_adatas = {}
    for name, adata in adatas.items():
        n_cells = adata.n_obs
        max_n = max_cells_per_dataset.get(name, n_cells)
        if n_cells > max_n:
            selected_inds = adata.obs.sample(n=max_n, random_state=42).index
            unselected_inds = adata.obs.index.difference(selected_inds)
            adata_balanced = adata[selected_inds, :].copy()
            adata_unused = adata[unselected_inds, :].copy()
            print(f"Balanced {name}: sampled {max_n} cells from {n_cells}")
        else:
            adata_balanced = adata
            print(f"Balanced {name}: using all {n_cells} cells")
        balanced_adatas[name] = adata_balanced
        unused_adatas[name] = adata_unused if n_cells > max_n else None
    return balanced_adatas, unused_adatas


# Perform data integration with scVI
def integrate_scvi(adatas, latent_dim=30):
    import scvi  # Import here to avoid OpenMP crash on module load
    
    # Concatenate datasets
    adata_combined = sc.concat(adatas.values(), join='inner')
    
    # Add dataset labels for batch correction (extract everything before last underscore)
    adata_combined.obs['dataset'] = adata_combined.obs_names.map(lambda x: x.rsplit('_', 1)[0])

    # Setup scVI model
    scvi.model.SCVI.setup_anndata(adata_combined, batch_key="dataset")

    # Create and train the model
    model = scvi.model.SCVI(adata_combined, n_latent=latent_dim)
    model.train()

    # Get the latent representation
    adata_combined.obsm["X_scVI"] = model.get_latent_representation()

    # Get log-normalized batch-corrected gene expression
    adata_combined.layers["scVI_corrected"] = model.get_normalized_expression()
    adata_combined.layers["scVI_corrected_log"] = np.log1p(adata_combined.layers["scVI_corrected"])

    return adata_combined, model


# Align new data to an existing scVI model
def align_to_scvi_model(adatas, model_path, reference_adata=None):
    """
    Align new datasets to an existing trained scVI model.
    Parameters:
    - adatas: dict of AnnData objects to align
    - model_path: path to the saved scVI model
    - reference_adata: reference AnnData used for training (required)
    Returns:
    - adata_aligned: AnnData with aligned latent representations
    """
    import scvi  # Import here to avoid OpenMP crash on module load

    # Load the trained model with reference adata
    model = scvi.model.SCVI.load(model_path, adata=reference_adata)
    
    # Concatenate new datasets
    adata_query = sc.concat(adatas.values(), join='inner')
    
    # Add dataset labels (extract everything before last underscore)
    adata_query.obs['dataset'] = adata_query.obs_names.map(lambda x: x.rsplit('_', 1)[0])
    
    # Subset to genes present in both, then pad with zeros for missing genes
    common_genes = reference_adata.var_names.intersection(adata_query.var_names)
    missing_genes = reference_adata.var_names.difference(adata_query.var_names)
    print(f"Common genes: {len(common_genes)}/{len(reference_adata.var_names)}")
    print(f"Missing in query: {len(missing_genes)}")
    
    # First subset to common genes
    adata_query_subset = adata_query[:, common_genes].copy()
    
    # Add missing genes with zeros
    import pandas as pd
    if len(missing_genes) > 0:
        # Create a zero matrix for missing genes
        missing_matrix = np.zeros((adata_query_subset.n_obs, len(missing_genes)))
        missing_adata = sc.AnnData(
            X=missing_matrix,
            obs=adata_query_subset.obs.copy(),
            var=pd.DataFrame(index=missing_genes)
        )
        # Concatenate common + missing genes, then reorder to match reference
        adata_query_full = sc.concat([adata_query_subset, missing_adata], axis=1)
        # Reorder to match reference and preserve obs
        adata_query_reordered = adata_query_full[:, reference_adata.var_names].copy()
        # Make sure 'dataset' column is preserved
        adata_query_reordered.obs = adata_query.obs.copy()
        adata_query = adata_query_reordered
    else:
        adata_query = adata_query_subset
    
    # Prepare query data for the model
    # This handles new batch categories automatically
    query_model = scvi.model.SCVI.load_query_data(adata_query, model)
    
    # Train the query model (fine-tuning on new data)
    print("Training query model...")
    query_model.train(max_epochs=200, plan_kwargs={'weight_decay': 0.0})
    
    # Get latent representation using the query model
    adata_query.obsm["X_scVI"] = query_model.get_latent_representation()
    
    # Get log-normalized batch-corrected gene expression
    adata_query.layers["scVI_corrected"] = query_model.get_normalized_expression()
    adata_query.layers["scVI_corrected_log"] = np.log1p(adata_query.layers["scVI_corrected"])
    
    return adata_query
