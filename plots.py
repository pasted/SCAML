from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_learning_curve(fracs, scores, out_png: str):
    fig = plt.figure(figsize=(5,4))
    plt.plot(fracs, scores, marker="o")
    plt.xlabel("Training fraction (by groups)")
    plt.ylabel("Macro‑F1")
    plt.title("Learning Curve")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)