import numpy as np
import anndata as ad
import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def synthetic_adata():
    """Create a tiny AnnData with clear signal between two microglia subtypes and a few HEK cells.
    Columns in .obs: harvest, culture, chip, label_ref, is_hek
    obsm contains a faux X_harmony embedding.
    """
    rng = np.random.default_rng(42)

    n_genes = 300
    n_mg   = 60  # microglia cells
    n_hek  = 10  # HEK controls
    n      = n_mg + n_hek

    # base counts
    X = rng.poisson(lam=1.0, size=(n, n_genes)).astype(float)

    # two microglia subtypes with signal on disjoint gene blocks
    mg_labels = np.array(["homeostatic"] * (n_mg // 2) + ["activated"] * (n_mg - n_mg // 2))
    hek_labels = np.array(["HEK"] * n_hek)
    labels = np.concatenate([mg_labels, hek_labels])

    # Boost signal: activated upregulates genes 0..19, homeostatic 20..39
    act_idx = np.where(labels == "activated")[0]
    hom_idx = np.where(labels == "homeostatic")[0]
    X[act_idx, 0:20]  += rng.normal(loc=3.0, scale=0.5, size=(len(act_idx), 20))
    X[hom_idx, 20:40] += rng.normal(loc=3.0, scale=0.5, size=(len(hom_idx), 20))

    # HEK cells upregulate a different block to help novelty detection
    hek_idx = np.where(labels == "HEK")[0]
    X[hek_idx, 200:240] += rng.normal(loc=4.0, scale=0.5, size=(len(hek_idx), 40))

    # Metadata
    harvest = np.array(["H4"] * (n // 2) + ["H8"] * (n - n // 2))
    culture = np.array(["A", "C", "D"])  # cycle through
    culture = np.array([culture[i % 3] for i in range(n)])
    chip    = np.array(["chip1" if i < n // 2 else "chip2" for i in range(n)])
    is_hek  = labels == "HEK"

    var_names = [f"G{i:04d}" for i in range(n_genes)]
    adata = ad.AnnData(X=X)
    adata.var_names = var_names
    adata.obs["label_ref"] = labels
    adata.obs["harvest"]   = harvest
    adata.obs["culture"]   = culture
    adata.obs["chip"]      = chip
    adata.obs["is_hek"]    = is_hek

    # Faux harmony embedding (random linear proj to 20 dims)
    W = rng.normal(size=(n_genes, 20))
    adata.obsm["X_harmony"] = X @ W

    return adata


@pytest.fixture()
def tmpdir_out(tmp_path):
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture()
def saved_adata_tmp(tmp_path, synthetic_adata):
    path = tmp_path / "toy.h5ad"
    synthetic_adata.write_h5ad(path)
    return path