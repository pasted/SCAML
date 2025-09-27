from __future__ import annotations
import numpy as np
import scanpy as sc
from typing import Optional

def select_hvgs(adata, n_top: int = 3000, layer: Optional[str] = None):
    tmp = adata.copy()
    if layer:
        tmp.X = tmp.layers[layer]
    sc.pp.highly_variable_genes(tmp, n_top_genes=n_top, flavor="seurat_v3")
    hv = tmp.var["highly_variable"].values
    return np.where(hv)[0], tmp.var_names[hv].tolist()

def get_features(adata, embedding: Optional[str] = None, hv_idx=None):
    if embedding:
        if embedding not in adata.obsm:
            raise ValueError(f"Embedding {embedding} not found in adata.obsm")
        X = adata.obsm[embedding]
        return X
    # else: use genes (subset HVGs if provided)
    X = adata.X
    if hv_idx is not None:
        X = X[:, hv_idx]
    return X