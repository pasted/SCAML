#!/usr/bin/env python3
"""
SCAML Plotting Utility

Generates:
  - model_compare.png              (accuracy & F1 per model)
  - <model>/confusion_matrix.png   (per-model confusion matrix heatmap)

Inputs expected in --results (created by scaml_train.py / train_eval.py):
  - summary_metrics.json
  - classes.txt
  - <model>/confusion_matrix.csv
  - <model>/classification_report.txt (optional)
  - <model>/predictions.csv (optional)

Usage:
  python scaml_plot.py --results results/baseline_H4train_H8test
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _mkdir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def load_summary(results_dir: str) -> Dict[str, Dict]:
    p = os.path.join(results_dir, "summary_metrics.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing summary_metrics.json at {p}")
    with open(p, "r") as fh:
        return json.load(fh)


def load_classes(results_dir: str) -> Optional[List[str]]:
    p = os.path.join(results_dir, "classes.txt")
    if not os.path.exists(p):
        return None
    classes = []
    with open(p, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # format: "<idx>\t<label>"
            parts = line.split("\t", 1)
            if len(parts) == 2:
                classes.append(parts[1])
            else:
                classes.append(parts[0])
    return classes


def plot_model_compare(summary: Dict[str, Dict], out_path: str) -> None:
    # Build dataframe for plotting
    rows = []
    for model, metr in summary.items():
        if not isinstance(metr, dict):
            continue
        acc = metr.get("accuracy", np.nan)
        f1 = metr.get("f1_macro", np.nan)
        rows.append({"model": model, "metric": "accuracy", "value": acc})
        rows.append({"model": model, "metric": "f1_macro", "value": f1})
    if not rows:
        print("[WARN] No metrics to plot in summary.")
        return
    df = pd.DataFrame(rows)

    # Sort models by accuracy (descending) for nicer ordering
    order = (
        df[df["metric"] == "accuracy"]
        .sort_values("value", ascending=False)["model"]
        .tolist()
    )
    df["model"] = pd.Categorical(df["model"], categories=order, ordered=True)

    # Plot: grouped bars (accuracy vs f1)
    models = list(dict.fromkeys(order))  # preserve order, unique
    metrics = ["accuracy", "f1_macro"]
    width = 0.38
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for i, metric in enumerate(metrics):
        vals = [
            float(df[(df["model"] == m) & (df["metric"] == metric)]["value"].values[0])
            if not df[(df["model"] == m) & (df["metric"] == metric)]["value"].empty
            else np.nan
            for m in models
        ]
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=metric.upper())

        # Annotate
        for xi, v in zip(x + (i - 0.5) * width, vals):
            if np.isfinite(v):
                ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("SCAML: Model Comparison (Accuracy & F1 Macro)")
    ax.legend(frameon=False)
    ax.grid(axis="y", linewidth=0.5, alpha=0.3)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Wrote {out_path}")


def _fmt_percent(v: float) -> str:
    if not np.isfinite(v):
        return ""
    return f"{100.0 * v:.1f}%"


def plot_confusion(cm_csv: str, classes_txt: Optional[List[str]], out_path: str) -> None:
    if not os.path.exists(cm_csv):
        print(f"[WARN] Missing confusion matrix: {cm_csv}")
        return
    cm_df = pd.read_csv(cm_csv, index_col=0)

    # Prefer labels from the CSV (header/index) to guarantee alignment
    row_labels = cm_df.index.tolist()
    col_labels = cm_df.columns.tolist()

    # If classes.txt provided and sizes match, use those (nicer names)
    if classes_txt and len(classes_txt) == len(row_labels) == len(col_labels):
        row_labels = classes_txt
        col_labels = classes_txt

    cm = cm_df.values.astype(float)
    totals = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.divide(cm, totals, where=totals != 0)

    fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    im = ax.imshow(frac, interpolation="nearest", aspect="auto")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Ticks & labels
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalized)")

    # Grid lines
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5, alpha=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate each cell with count and percentage
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            pct = frac[i, j]
            txt = f"{count}\n{_fmt_percent(pct)}" if np.isfinite(pct) else f"{count}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="black")

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Wrote {out_path}")


def infer_models(results_dir: str) -> List[str]:
    # Typical subdirs (lr, rf, xgb, mlp); keep only those that exist
    candidates = ["lr", "rf", "xgb", "mlp"]
    present = [m for m in candidates if os.path.isdir(os.path.join(results_dir, m))]
    if present:
        return present
    # Fallback: any directories with a confusion_matrix.csv
    models = []
    for name in os.listdir(results_dir):
        p = os.path.join(results_dir, name, "confusion_matrix.csv")
        if os.path.isfile(p):
            models.append(name)
    return sorted(models)


def main():
    ap = argparse.ArgumentParser(description="SCAML plotting helper")
    ap.add_argument("--results", required=True, help="Path to a SCAML results folder")
    ap.add_argument("--outdir", default=None, help="Directory to write plots (default: <results>/plots)")
    ap.add_argument("--models", nargs="+", default=None, help="Subset of models to plot (e.g., lr rf xgb mlp)")
    args = ap.parse_args()

    results_dir = os.path.abspath(args.results)
    if not os.path.isdir(results_dir):
        raise NotADirectoryError(f"Results directory not found: {results_dir}")

    outdir = args.outdir or os.path.join(results_dir, "plots")
    _mkdir(outdir)

    # 1) Model comparison
    summary = load_summary(results_dir)
    plot_model_compare(summary, os.path.join(outdir, "model_compare.png"))

    # 2) Per-model confusion matrices
    class_names = load_classes(results_dir)  # may be None; we'll fall back to CSV header
    models = args.models or infer_models(results_dir)
    if not models:
        print("[WARN] No model subdirectories found to plot confusion matrices.")
    for m in models:
        cm_csv = os.path.join(results_dir, m, "confusion_matrix.csv")
        out_png = os.path.join(outdir, f"{m}_confusion_matrix.png")
        plot_confusion(cm_csv, class_names, out_png)


if __name__ == "__main__":
    main()
