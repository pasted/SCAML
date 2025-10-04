import sys
from pathlib import Path

import numpy as np

# import path set in conftest
from scaml.features import select_hvgs


def test_subset_split_and_keys(toy_h5ad):
    import scanpy as sc  # lazy import here to keep test output tidy
    adata = sc.read_h5ad(str(toy_h5ad))

    # required keys
    for k in ["label_ref", "harvest", "culture"]:
        assert k in adata.obs.columns
    assert "X_harmony" in adata.obsm_keys()

    # split sizes we encoded in the fixture
    n_train = int((adata.obs["harvest"] == "H4").sum())
    n_test = int((adata.obs["harvest"] == "H8").sum())
    assert n_train == 30
    assert n_test == 30


def test_hvg_selection_on_counts(toy_h5ad):
    import scanpy as sc
    adata = sc.read_h5ad(str(toy_h5ad))

    hv_idx, hv_names = select_hvgs(adata, n_top=10)
    assert 1 <= len(hv_names) <= 10
    # Indices should be valid
    assert np.all((hv_idx >= 0) & (hv_idx < adata.n_vars))
