from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score

from .features import select_hvgs, get_features


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