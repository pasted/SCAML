from __future__ import annotations
import anndata as ad
import pandas as pd
from pathlib import Path

REQ_OBS_KEYS = ["harvest", "culture", "label_ref", "is_hek"]

def load_adata(path: str | Path) -> ad.AnnData:
    adata = ad.read_h5ad(str(path))
    # normalize obs key names to lowercase variants if present
    obs = adata.obs
    for k in list(obs.columns):
        if k.lower() != k and k.lower() in [x.lower() for x in REQ_OBS_KEYS]:
            obs.rename(columns={k: k.lower()}, inplace=True)
    return adata

def ensure_keys(adata: ad.AnnData, keys: list[str]) -> None:
    missing = [k for k in keys if k not in adata.obs.columns]
    if missing:
        print(f"[WARN] Missing obs keys: {missing}")

def subset_split(adata: ad.AnnData, split_key: str, train_vals: list[str], test_vals: list[str]):
    train = adata[adata.obs[split_key].isin(train_vals)].copy()
    test  = adata[adata.obs[split_key].isin(test_vals)].copy()
    return train, test