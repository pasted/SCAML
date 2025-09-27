from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

from .features import select_hvgs, get_features
from .models import make_model


def train_and_eval(adata_train, adata_test, label_key: str, models: list[str],
                   features: str = "hvg", n_hvg: int = 3000, embedding: str | None = None,
                   outdir: str | Path = "results"):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    y_train = adata_train.obs[label_key].astype(str).values
    y_test  = adata_test .obs[label_key].astype(str).values
    classes = np.unique(np.concatenate([y_train, y_test]))

    hv_idx, hv_names = (None, None)
    if features == "hvg":
        hv_idx, hv_names = select_hvgs(adata_train, n_top=n_hvg)
        pd.Series(hv_names).to_csv(outdir/"features_hvg.csv", index=False)

    X_train = get_features(adata_train, embedding=embedding, hv_idx=hv_idx)
    X_test  = get_features(adata_test , embedding=embedding, hv_idx=hv_idx)

    results = []
    for m in models:
        model = make_model(m, n_classes=len(classes))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        df_rep = pd.DataFrame(report).T
        df_rep.to_csv(outdir / f"report_{m}.csv")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        fig = plt.figure(figsize=(6,5));
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"Confusion Matrix — {m}")
        fig.tight_layout(); fig.savefig(outdir / f"confusion_{m}.png", dpi=150); plt.close(fig)

        # AUROC (one-vs-rest)
        try:
            y_score = _predict_proba(model, X_test)
            Y = label_binarize(y_test, classes=classes)
            auroc = roc_auc_score(Y, y_score, average="macro", multi_class="ovr")
        except Exception:
            auroc = np.nan

        results.append({"model": m, "macro_f1": df_rep.loc["macro avg", "f1-score"],
                        "accuracy": df_rep.loc["accuracy", "precision"], "auroc": auroc})

        # Save model
        try:
            import joblib
            joblib.dump(model, outdir / f"model_{m}.joblib")
        except Exception:
            pass

    pd.DataFrame(results).to_csv(outdir/"summary.csv", index=False)


def _predict_proba(pipeline, X):
    clf = pipeline.named_steps.get("clf", pipeline)
    # Some models (e.g., RF/XGB/MLP/LR) have predict_proba
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    raise AttributeError("Model lacks predict_proba for AUROC computation")