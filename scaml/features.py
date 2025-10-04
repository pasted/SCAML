from __future__ import annotations
from typing import Tuple, List, Optional

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData


def _choose_hvg_flavor(preferred: Optional[str] = None) -> str:
    if preferred and preferred != "auto":
        return preferred
    # Prefer seurat_v3 only if scikit-misc is available
    try:
        import skmisc  # noqa: F401
        return "seurat_v3"
    except Exception:
        return "cell_ranger"  # no extra deps, works on counts


def _ensure_counts_in_X(adata: AnnData) -> AnnData:
    """Return a view/copy with counts in .X if available, else a safe numeric X."""
    tmp = adata.copy()
    # If counts layer exists, use it
    if "counts" in tmp.layers:
        tmp.X = tmp.layers["counts"]
        return tmp
    # If X looks log-transformed (floats), move to approx counts domain
    if np.issubdtype(tmp.X.dtype, np.floating):
        tmp.X = np.expm1(tmp.X).astype(np.float32)
    return tmp


def select_hvgs(adata: AnnData, n_top: int = 2000, flavor: str = "auto") -> Tuple[np.ndarray, List[str]]:
    """Select HVGs with a dependency-light fallback.

    - If scikit-misc is present → use seurat_v3
    - Else → use cell_ranger
    """
    hv_flavor = _choose_hvg_flavor(flavor)
    tmp = _ensure_counts_in_X(adata)

    sc.pp.highly_variable_genes(tmp, n_top_genes=n_top, flavor=hv_flavor)
    mask = tmp.var["highly_variable"].to_numpy()
    idx = np.where(mask)[0]
    names = tmp.var_names[mask].tolist()
    return idx, names


def get_features(
    adata: AnnData,
    features: str = "hvg",
    n_hvg: int = 2000,
    embedding: Optional[str] = None,
    flavor: str = "auto",
    dense: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Returns (X, names) where:
      - X is a 2D numpy array of shape (n_cells, n_features) (dense if dense=True)
      - names are feature names (genes or embedding dims)

    Parameters
    ----------
    adata : AnnData
        The AnnData object.
    features : {"hvg","embedding"}
        Whether to use highly-variable genes or an embedding in .obsm.
    n_hvg : int
        Number of HVGs to select if features="hvg".
    embedding : str or None
        Key in adata.obsm to use if features="embedding". If None, tries common keys.
    flavor : {"auto","seurat_v3","cell_ranger"}
        HVG selection flavor. "auto" will pick seurat_v3 if scikit-misc is present, else cell_ranger.
    dense : bool
        If True, convert output to a dense numpy array (recommended for scikit-learn).

    Returns
    -------
    X : np.ndarray
    names : list[str]
    """
    mode = features.lower()

    if mode == "hvg":
        # Select HVGs and slice matrix
        hv_idx, hv_names = select_hvgs(adata, n_top=n_hvg, flavor=flavor)
        X = adata.X[:, hv_idx]
        # Ensure dense for sklearn models that don’t accept sparse
        if dense:
            if sp.issparse(X):
                X = X.toarray()
            else:
                X = np.asarray(X)
        return X, hv_names

    elif mode in {"embedding", "emb", "obsm"}:
        key = embedding
        if not key:
            # pick a sensible default if not provided
            for cand in ("X_harmony", "X_pca", "X_umap"):
                if cand in adata.obsm_keys():
                    key = cand
                    break
        if key not in adata.obsm_keys():
            raise KeyError(
                f"Embedding '{key}' not found in adata.obsm. "
                f"Available: {list(adata.obsm_keys())}"
            )
        X = adata.obsm[key]
        if dense:
            if sp.issparse(X):
                X = X.toarray()
            else:
                X = np.asarray(X)
        names = [f"{key}_{i}" for i in range(X.shape[1])]
        return X, names

    else:
        raise ValueError("features must be 'hvg' or 'embedding'")

