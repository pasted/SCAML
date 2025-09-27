#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import scanpy as sc
from scaml.io import load_adata, subset_split
from scaml.train_eval import train_and_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata", required=True)
    ap.add_argument("--label-key", default="label_ref")
    ap.add_argument("--split-key", default="harvest")
    ap.add_argument("--split-train", nargs="+", default=["H4"]) 
    ap.add_argument("--split-test",  nargs="+", default=["H8"]) 
    ap.add_argument("--features", choices=["hvg","raw"], default="hvg")
    ap.add_argument("--n-hvg", type=int, default=3000)
    ap.add_argument("--embedding", default=None)
    ap.add_argument("--models", nargs="+", default=["lr","rf","xgb","mlp"]) 
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    adata = load_adata(args.adata)
    adata_train, adata_test = subset_split(adata, args.split_key, args.split_train, args.split_test)

    train_and_eval(
        adata_train, adata_test,
        label_key=args.label_key,
        models=args.models,
        features=args.features,
        n_hvg=args.n_hvg,
        embedding=args.embedding,
        outdir=args.outdir,
    )

if __name__ == "__main__":
    main()