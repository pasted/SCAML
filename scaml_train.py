#!/usr/bin/env python
# SCAML/scaml_train.py

import os
import argparse
import scanpy as sc

# If SCAML is installed as a package, this works:
try:
    from scaml.train_eval import train_and_eval
except Exception:
    # Fallback if you're running from a repo layout without installation
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    pkg_root = os.path.join(here, "scaml")
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from train_eval import train_and_eval  # type: ignore


def parse_args():
    p = argparse.ArgumentParser(description="SCAML: train ML models on scRNA-seq AnnData")
    p.add_argument("--adata", required=True, help=".h5ad input with counts in X")
    p.add_argument("--label-key", required=True, help="obs column with labels")
    p.add_argument("--split-key", required=True, help="obs column used to split train/test")
    p.add_argument("--split-train", required=True, help="value of split-key for TRAIN")
    p.add_argument("--split-test", required=True, help="value of split-key for TEST")
    p.add_argument("--features", default="hvg", choices=["hvg"], help="feature mode")
    p.add_argument("--n-hvg", type=int, default=3000, help="number of HVGs when features=hvg")
    p.add_argument("--embedding", default=None, help="obsm key to use instead of HVGs (e.g., X_harmony)")
    p.add_argument("--models", nargs="+", default=["lr"], help="models to run: lr rf xgb mlp")
    p.add_argument("--outdir", required=True, help="output directory")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load AnnData
    adata = sc.read_h5ad(args.adata)

    # Keyword-only call prevents "multiple values for argument 'label_key'"
    train_and_eval(
        adata=adata,
        label_key=args.label_key,
        split_key=args.split_key,
        split_train=args.split_train,
        split_test=args.split_test,
        features=args.features,
        n_hvg=args.n_hvg,
        embedding=args.embedding,
        models=args.models,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
