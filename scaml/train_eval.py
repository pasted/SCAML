# SCAML/scaml/train_eval.py
import os
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

import scanpy as sc  # noqa: F401
from sklearn.preprocessing import LabelEncoder, StandardScaler, MaxAbsScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.base import ClassifierMixin

try:
    from xgboost import XGBClassifier  # type: ignore
    _HAVE_XGB = True
except Exception:
    _HAVE_XGB = False

from .features import select_hvgs  # expects (idx, names)


# ----------------------------- helpers --------------------------------- #

def _ensure_csr(X):
    if sparse.isspmatrix_csr(X):
        return X
    if sparse.issparse(X):
        return X.tocsr()
    return np.asarray(X)


def _matrix_from_genes(adata, genes: List[str]):
    var_index = {g: i for i, g in enumerate(adata.var_names)}
    cols = [var_index[g] for g in genes if g in var_index]
    if len(cols) == 0:
        raise ValueError("None of the requested genes were found in adata.var_names.")
    X = adata.X[:, cols]
    return _ensure_csr(X), [adata.var_names[i] for i in cols]


def _log1p_safe(X):
    if sparse.issparse(X):
        X = X.copy()
        X.data = np.log1p(X.data)
        return X
    return np.log1p(np.asarray(X))


def _prepare_features(
    adata_train,
    adata_test,
    features: str,
    n_hvg: int,
    embedding: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build feature matrices for train and test:
    - If `embedding` is provided and exists in .obsm, use it.
    - Else if `features=="hvg"`, select HVGs on the *train* set and use log1p(counts).
    Returns: X_train, X_test, feature_names
    """
    if embedding:
        if embedding not in adata_train.obsm_keys():
            raise KeyError(f"Embedding '{embedding}' not found in adata_train.obsm.")
        if embedding not in adata_test.obsm_keys():
            raise KeyError(f"Embedding '{embedding}' not found in adata_test.obsm.")
        X_train = np.asarray(adata_train.obsm[embedding])
        X_test = np.asarray(adata_test.obsm[embedding])
        feat_names = [f"{embedding}_{i}" for i in range(X_train.shape[1])]
        return X_train, X_test, feat_names

    if features.lower() == "hvg":
        _, hv_names = select_hvgs(adata_train, n_top=n_hvg)
        Xtr_counts, hv_names_tr = _matrix_from_genes(adata_train, hv_names)
        Xte_counts, hv_names_te = _matrix_from_genes(adata_test, hv_names)

        hv_set = set(hv_names_tr) & set(hv_names_te)
        hv_order = [g for g in hv_names if g in hv_set]
        Xtr_counts, _ = _matrix_from_genes(adata_train, hv_order)
        Xte_counts, _ = _matrix_from_genes(adata_test, hv_order)

        X_train = _log1p_safe(Xtr_counts)
        X_test = _log1p_safe(Xte_counts)
        return X_train, X_test, hv_order

    raise ValueError(f"Unsupported features mode: {features}")


def _build_model(name: str, n_classes: int) -> Optional[ClassifierMixin]:
    name = name.lower().strip()
    if name == "lr":
        return Pipeline([
            ("scale", MaxAbsScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                n_jobs=None,
                multi_class="auto",  # harmless warning on sklearn 1.5
            )),
        ])
    if name == "rf":
        return Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                n_jobs=-1,
                random_state=42,
            )),
        ])
    if name == "mlp":
        to_dense = FunctionTransformer(
            lambda x: x.toarray() if sparse.issparse(x) else np.asarray(x),
            accept_sparse=True
        )
        return Pipeline([
            ("todense", to_dense),
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                max_iter=200,
                random_state=42
            )),
        ])
    if name == "xgb":
        if not _HAVE_XGB:
            return None
        return Pipeline([
            ("scale", MaxAbsScaler()),
            ("clf", XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.8,
                tree_method="hist",
                objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                eval_metric="mlogloss" if n_classes > 2 else "logloss",
                random_state=42,
                n_jobs=0,
            )),
        ])
    raise ValueError(f"Unknown model '{name}'")


def _safe_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    """
    Robust macro OVR ROC-AUC:
      1) If <2 classes present → NaN
      2) Try sklearn multiclass with explicit labels
      3) Fallback: average per-class binary AUC over classes present with both pos/neg
    """
    try:
        present = np.unique(y_true)
        if present.size < 2:
            return float("nan")
        return float(
            roc_auc_score(
                y_true, proba, multi_class="ovr", average="macro", labels=present
            )
        )
    except Exception:
        aucs = []
        for k in np.unique(y_true):
            y_bin = (y_true == k).astype(int)
            if 0 < y_bin.sum() < y_bin.size:
                try:
                    aucs.append(roc_auc_score(y_bin, proba[:, k]))
                except Exception:
                    pass
        return float(np.mean(aucs)) if aucs else float("nan")


def _write_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def _write_json(path: str, obj: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def _final_estimator(model):
    return model.steps[-1][1] if isinstance(model, Pipeline) else model


# ----------------------------- main API -------------------------------- #

def train_and_eval(
    adata,
    label_key: str,
    split_key: str,
    split_train: str,
    split_test: str,
    features: str,
    n_hvg: int,
    embedding: Optional[str],
    models: List[str],
    outdir: str,
) -> Dict[str, Dict]:
    """
    Train and evaluate one or more models on AnnData.
    """
    os.makedirs(outdir, exist_ok=True)

    # Split
    tr_mask = adata.obs[split_key].astype(str) == str(split_train)
    te_mask = adata.obs[split_key].astype(str) == str(split_test)
    if tr_mask.sum() == 0 or te_mask.sum() == 0:
        raise ValueError("Empty train or test split; check split_key and split values.")

    adata_train = adata[tr_mask].copy()
    adata_test = adata[te_mask].copy()

    # Labels: encode to ints; keep mapping for reports
    y_train_str = adata_train.obs[label_key].astype(str)
    y_test_str = adata_test.obs[label_key].astype(str)
    le = LabelEncoder()
    le.fit(pd.Index(y_train_str).append(pd.Index(y_test_str)))  # union for stability
    y_train_global = le.transform(y_train_str)
    y_test = le.transform(y_test_str)
    class_names = le.classes_
    label_indices = np.arange(len(class_names))

    # Features
    X_train, X_test, feat_names = _prepare_features(
        adata_train, adata_test, features=features, n_hvg=n_hvg, embedding=embedding
    )
    X_train = _ensure_csr(X_train)
    X_test = _ensure_csr(X_test)

    # Save mapping & features
    _write_text(os.path.join(outdir, "classes.txt"),
                "\n".join(f"{i}\t{c}" for i, c in enumerate(class_names)))
    _write_text(os.path.join(outdir, "features.txt"),
                "\n".join(map(str, feat_names)))

    results: Dict[str, Dict] = {}
    n_classes_global = len(class_names)

    for name in models:
        model = _build_model(name, n_classes=n_classes_global)
        if model is None:
            warn = f"Model '{name}' skipped (xgboost not installed)."
            print(warn)
            results[name] = {"warning": warn}
            continue

        # --------- Handle XGB contiguous-label requirement ---------
        # XGBoost needs training labels as 0..K-1 without gaps.
        if name.lower() == "xgb":
            present_global = np.unique(y_train_global)
            local_from_global = {g: i for i, g in enumerate(present_global)}
            inv_local_to_global = np.asarray(present_global)  # local idx -> global idx
            y_train_fit = np.vectorize(local_from_global.get)(y_train_global)
            # Fit
            model.fit(X_train, y_train_fit)
            # Predict (local -> global)
            y_pred_local = model.predict(X_test)
            y_pred = inv_local_to_global[y_pred_local]
            trained_global_classes = inv_local_to_global  # for probability expansion
        else:
            # Other models can handle non-contiguous global labels directly
            y_train_fit = y_train_global
            model.fit(X_train, y_train_fit)
            y_pred = model.predict(X_test)
            # Estimator classes_ are the global label indices used during training
            est = _final_estimator(model)
            trained_global_classes = getattr(est, "classes_", None)

        # Classification report with explicit labels to match target_names
        report_txt = classification_report(
            y_test,
            y_pred,
            labels=label_indices,
            target_names=class_names,
            zero_division=0,
        )

        # Metrics
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="macro"))

        model_dir = os.path.join(outdir, name)
        os.makedirs(model_dir, exist_ok=True)

        # Probas / AUC (robust & expanded to full class set)
        auc = float("nan")
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)

            try:
                if isinstance(proba, list):
                    proba = np.vstack([p[:, 1] for p in proba]).T
            except Exception:
                pass

            proba = np.asarray(proba)
            if proba.ndim == 1:
                proba = np.column_stack([1.0 - proba, proba])

            # Determine which *global* classes these columns correspond to
            if name.lower() == "xgb":
                # columns are local 0..K-1 -> map to global via inv map
                global_cols = np.asarray(trained_global_classes)
            else:
                # sklearn keeps .classes_ as the labels used in fit (global indices)
                est = _final_estimator(model)
                if hasattr(est, "classes_"):
                    global_cols = np.asarray(est.classes_)
                else:
                    # fallback: assume 0..C-1 already match global indices
                    global_cols = np.arange(proba.shape[1])

            # Expand to full set of global classes
            full_proba = np.zeros((proba.shape[0], n_classes_global), dtype=float)
            mask = (global_cols >= 0) & (global_cols < n_classes_global)
            full_proba[:, global_cols[mask]] = proba[:, mask]

            # AUC using full matrix (and only across classes present in y_test)
            auc = _safe_auc(y_test, full_proba)

            proba_df = pd.DataFrame(full_proba, columns=list(class_names))
            proba_df.insert(0, "cell", adata_test.obs_names)
            proba_df.to_csv(os.path.join(model_dir, "proba.csv"), index=False)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=label_indices)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

        # Persist outputs
        _write_text(os.path.join(model_dir, "classification_report.txt"), report_txt)
        cm_df.to_csv(os.path.join(model_dir, "confusion_matrix.csv"))

        pred_df = pd.DataFrame({
            "cell": adata_test.obs_names,
            "y_true": y_test_str.values,
            "y_pred": class_names[y_pred],
            "y_true_idx": y_test,
            "y_pred_idx": y_pred,
        })
        pred_df.to_csv(os.path.join(model_dir, "predictions.csv"), index=False)

        metrics = {"accuracy": acc, "f1_macro": f1, "roc_auc_macro_ovr": auc}
        _write_json(os.path.join(model_dir, "metrics.json"), metrics)

        results[name] = metrics

    _write_json(os.path.join(outdir, "summary_metrics.json"), results)
    return results
