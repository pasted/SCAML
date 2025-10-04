from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from .features import select_hvgs, get_features
from typing import Optional, Tuple
from anndata import AnnData




def novelty_eval(adata, hek_key: str, hek_positive: str | bool = True,
                 features: str = "hvg", n_hvg: int = 3000, embedding: str | None = None,
                 outdir: str | Path = "results/novelty", method: str = "iforest"):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    y = adata.obs[hek_key].astype(str).values
    is_hek = (y == str(hek_positive))

    # Train only on non‑HEK (microglia)
    adata_micro = adata[~is_hek].copy()

    hv_idx, _ = (None, None)
    if features == "hvg":
        hv_idx, _ = select_hvgs(adata_micro, n_top=n_hvg)

    X_train = get_features(adata_micro, embedding=embedding, hv_idx=hv_idx)
    X_all   = get_features(adata,        embedding=embedding, hv_idx=hv_idx)

    if method == "iforest":
        clf = IsolationForest(n_estimators=500, contamination="auto", random_state=7)
    elif method == "ocsvm":
        clf = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    else:
        raise ValueError("method must be 'iforest' or 'ocsvm'")

    clf.fit(X_train)
    # decision_function: higher = more normal; we want outlier score
    scores = -clf.decision_function(X_all)

    auroc = roc_auc_score(is_hek.astype(int), scores)
    aupr  = average_precision_score(is_hek.astype(int), scores)

    pd.DataFrame({"score": scores, "is_hek": is_hek}).to_csv(outdir/"scores.csv", index=False)
    with open(outdir/"metrics.txt", "w") as f:
        f.write(f"AUROC: {auroc:.4f}\nAUPR: {aupr:.4f}\n")


def train_iforest(
    adata: AnnData,
    hek_key: str = "is_hek",
    hek_positive: bool = True,
    features: str = "hvg",
    n_hvg: int = 2000,
    embedding: Optional[str] = None,
    random_state: int = 0,
    n_estimators: int = 200,
    contamination: str | float = "auto",
    max_samples: str | int | float = "auto",
) -> Tuple[IsolationForest, np.ndarray]:
    """
    Train an IsolationForest on *non-HEK* cells and score all cells.

    Parameters
    ----------
    adata : AnnData
        Input AnnData with obs[hek_key] indicating HEK controls.
    hek_key : str
        Column in adata.obs with HEK indicator.
    hek_positive : bool
        If True, obs[hek_key]==True means HEK. If False, invert logic.
    features : {"hvg","embedding"}
        Feature source. If "hvg", uses top n_hvg HVGs; if "embedding", uses `embedding` key in .obsm.
    n_hvg : int
        Number of HVGs when features="hvg".
    embedding : Optional[str]
        Embedding key in .obsm when features="embedding". If None, tries common keys.
    random_state : int
        RNG seed for IsolationForest.
    n_estimators, contamination, max_samples :
        Passed to sklearn.ensemble.IsolationForest.

    Returns
    -------
    model : IsolationForest
        Trained model (fitted on non-HEK cells).
    scores : np.ndarray
        Anomaly scores for all cells (higher == more anomalous).
        Shape: (adata.n_obs,)
    """
    if hek_key not in adata.obs.columns:
        raise KeyError(f"obs lacks '{hek_key}'")

    hek_series = adata.obs[hek_key]
    hek_mask = hek_series.astype(bool).to_numpy(copy=False)
    if not hek_positive:
        hek_mask = ~hek_mask

    # Train on microglia (non-HEK); score everyone
    train_mask = ~hek_mask

    X_all, _names = get_features(
        adata=adata,
        features=features,
        n_hvg=n_hvg,
        embedding=embedding,
        dense=True,
    )
    X_train = X_all[train_mask]

    # Scale on train only, then transform all
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train)
    X_all_s = scaler.transform(X_all)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train_s)

    # Higher score => more anomalous
    scores = -model.score_samples(X_all_s)
    return model, scores
