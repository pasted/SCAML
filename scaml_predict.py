#!/usr/bin/env python3
from __future__ import annotations
import argparse
import joblib
import anndata as ad
import pandas as pd

from scaml.io import load_adata
from scaml.features import select_hvgs, get_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--features", choices=["hvg","raw"], default="hvg")
    ap.add_argument("--n-hvg", type=int, default=3000)
    ap.add_argument("--embedding", default=None)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    adata = load_adata(args.adata)
    hv_idx = None
    if args.features == "hvg":
        hv_idx, _ = select_hvgs(adata, n_top=args.n_hvg)

    X = get_features(adata, embedding=args.embedding, hv_idx=hv_idx)
    model = joblib.load(args.model)
    y_pred = model.predict(X)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        df = pd.DataFrame(proba, columns=[f"p_{i}" for i in range(proba.shape[1])])
    else:
        df = pd.DataFrame()
    df.insert(0, "pred", y_pred)
    df.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()