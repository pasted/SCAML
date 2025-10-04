from __future__ import annotations
import numpy as np
import shap
import pandas as pd
from pathlib import Path


def shap_top_features(model, X, feature_names, outdir):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    explainer = shap.Explainer(model.named_steps.get("clf", model))
    Sh = explainer(X)
    sv = np.abs(Sh.values).mean(axis=0)
    top = pd.Series(sv, index=feature_names).sort_values(ascending=False).head(100)
    top.to_csv(outdir/"shap_top100.csv")