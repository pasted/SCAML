from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier


def make_model(name: str, n_classes: int) -> Pipeline:
    name = name.lower()
    if name == "lr":
        clf = LogisticRegression(max_iter=4000, n_jobs=None, multi_class="auto")
        return Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", clf)])
    if name == "rf":
        clf = RandomForestClassifier(n_estimators=400, n_jobs=-1, class_weight="balanced")
        return Pipeline([("clf", clf)])
    if name == "xgb":
        clf = XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.7,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=-1,
        )
        return Pipeline([("clf", clf)])
    if name == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(256,128), activation="relu", batch_size=256, max_iter=60)
        return Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", clf)])
    raise ValueError(f"Unknown model '{name}'")