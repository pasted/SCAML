from pathlib import Path
import json

import numpy as np

from scaml.train_eval import train_and_eval


def test_train_eval_runs(toy_h5ad, outdir):
    import scanpy as sc
    adata = sc.read_h5ad(str(toy_h5ad))

    res = train_and_eval(
        adata=adata,
        label_key="label_ref",
        split_key="harvest",
        split_train="H4",
        split_test="H8",
        features="hvg",
        n_hvg=10,
        embedding=None,        # HVG path
        models=["lr", "rf", "mlp"],  # keep CI light (no xgb)
        outdir=str(outdir),
    )
    # summary file exists
    summary_path = Path(outdir) / "summary_metrics.json"
    assert summary_path.exists()

    with open(summary_path) as fh:
        summary = json.load(fh)
    # Models present and have accuracy/f1
    for m in ["lr", "rf", "mlp"]:
        assert m in summary
        assert "accuracy" in summary[m]
        assert "f1_macro" in summary[m]
        # AUC may be NaN if only one class in test; don't assert numeric

    # Per-model artifacts
    for m in ["lr", "rf", "mlp"]:
        mdir = Path(outdir) / m
        assert (mdir / "confusion_matrix.csv").exists()
        assert (mdir / "predictions.csv").exists()
        assert (mdir / "metrics.json").exists()
        assert (mdir / "proba.csv").exists()  # probability expansion path
