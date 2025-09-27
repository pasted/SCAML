#!/usr/bin/env python3
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

from scaml.io import load_adata
from scaml.features import select_hvgs, get_features
from scaml.models import make_model
from scaml.plots import plot_learning_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata", required=True)
    ap.add_argument("--label-key", default="label_ref")
    ap.add_argument("--group-key", default="culture")
    ap.add_argument("--features", choices=["hvg","raw"], default="hvg")
    ap.add_argument("--n-hvg", type=int, default=3000)
    ap.add_argument("--embedding", default=None)
    ap.add_argument("--model", default="lr")
    ap.add_argument("--outdir", default="results/learning_curves")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    adata = load_adata(args.adata)

    # Group‑aware CV: iteratively add groups (cultures) to training set
    groups = adata.obs[args.group_key].astype(str).unique().tolist()
    rng = np.random.default_rng(7)
    rng.shuffle(groups)

    hv_idx = None; feat_names = None
    if args.features == "hvg":
        hv_idx, feat_names = select_hvgs(adata, n_top=args.n_hvg)

    X = get_features(adata, embedding=args.embedding, hv_idx=hv_idx)
    y = adata.obs[args.label_key].astype(str).values
    g = adata.obs[args.group_key].astype(str).values

    scores = []
    fracs  = []
    for k in range(1, len(groups)):
        train_groups = groups[:k]
        test_groups  = groups[k:]
        tr_idx = np.isin(g, train_groups)
        te_idx = np.isin(g, test_groups)
        model = make_model(args.model, n_classes=len(np.unique(y)))
        model.fit(X[tr_idx], y[tr_idx])
        y_pred = model.predict(X[te_idx])
        f1 = f1_score(y[te_idx], y_pred, average="macro")
        scores.append(f1)
        fracs.append(tr_idx.mean())

    pd.DataFrame({"train_frac": fracs, "macro_f1": scores}).to_csv(outdir/"learning_curve.csv", index=False)
    plot_learning_curve(fracs, scores, str(outdir/"learning_curve.png"))

if __name__ == "__main__":
    main()