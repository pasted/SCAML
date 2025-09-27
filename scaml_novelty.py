#!/usr/bin/env python3
from __future__ import annotations
import argparse
from scaml.io import load_adata
from scaml.novelty import novelty_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata", required=True)
    ap.add_argument("--hek-key", default="is_hek")
    ap.add_argument("--hek-positive", default="true")
    ap.add_argument("--features", choices=["hvg","raw"], default="hvg")
    ap.add_argument("--n-hvg", type=int, default=3000)
    ap.add_argument("--embedding", default=None)
    ap.add_argument("--method", choices=["iforest","ocsvm"], default="iforest")
    ap.add_argument("--outdir", default="results/novelty")
    args = ap.parse_args()

    adata = load_adata(args.adata)
    novelty_eval(
        adata,
        hek_key=args.hek_key,
        hek_positive=args.hek_positive,
        features=args.features,
        n_hvg=args.n_hvg,
        embedding=args.embedding,
        outdir=args.outdir,
        method=args.method,
    )

if __name__ == "__main__":
    main()
