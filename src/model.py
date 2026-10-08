"""One-click modelling.

Pick a target column; this works out whether it is a classification or regression
problem, trains a baseline AND a real model, and reports which features mattered.

Training always happens on a SAMPLE, so a 10M-row file trains as fast as a small one.
The baseline is the point: a model that cannot beat "always guess the majority class"
is not a model, and reporting both is what separates analysis from a demo.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

MAX_TRAIN_ROWS = 50_000
MAX_CATEGORIES = 30


@dataclass
class ModelReport:
    task: str                     # "classification" | "regression"
    target: str
    n_rows: int
    features: list[str]
    metric_name: str
    model_score: float
    baseline_score: float
    importances: pd.DataFrame
    note: str = ""

    @property
    def beats_baseline(self) -> bool:
        return self.model_score > self.baseline_score


def detect_task(series: pd.Series) -> str:
    """Numeric with many distinct values -> regression, otherwise classification."""
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 20:
        return "regression"
    return "classification"


def candidate_targets(df: pd.DataFrame) -> list[str]:
    """Columns worth predicting: not unique identifiers, not constant."""
    out = []
    for c in df.columns:
        n = df[c].nunique(dropna=True)
        if n < 2:
            continue
        if n / max(len(df), 1) > 0.9 and not pd.api.types.is_numeric_dtype(df[c]):
            continue                       # an ID column
        out.append(c)
    return out


def _prepare(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    data = df.dropna(subset=[target]).copy()

    features = []
    for c in data.columns:
        if c == target:
            continue
        if pd.api.types.is_numeric_dtype(data[c]):
            features.append(c)
        elif data[c].nunique(dropna=True) <= MAX_CATEGORIES:
            features.append(c)             # low-cardinality text -> one-hot below

    X = data[features]
    X = pd.get_dummies(X, drop_first=True, dummy_na=True)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return X, data[target], features


def train(df: pd.DataFrame, target: str) -> ModelReport:
    note = ""
    if len(df) > MAX_TRAIN_ROWS:
        df = df.sample(MAX_TRAIN_ROWS, random_state=42)
        note = f"trained on a {MAX_TRAIN_ROWS:,}-row random sample for speed"

    X, y, features = _prepare(df, target)
    if X.empty or X.shape[1] == 0:
        raise ValueError("no usable feature columns for that target")

    task = detect_task(y)
    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)

    if task == "classification":
        model = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1)
        dummy = DummyClassifier(strategy="most_frequent")
        model.fit(X_tr, y_tr); dummy.fit(X_tr, y_tr)
        if y.nunique() == 2:
            metric, score = "F1", f1_score(y_te, model.predict(X_te), average="binary", pos_label=y.unique()[0])
            base = f1_score(y_te, dummy.predict(X_te), average="binary", pos_label=y.unique()[0], zero_division=0)
        else:
            metric = "accuracy"
            score = accuracy_score(y_te, model.predict(X_te))
            base = accuracy_score(y_te, dummy.predict(X_te))
    else:
        model = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
        dummy = DummyRegressor(strategy="mean")
        model.fit(X_tr, y_tr); dummy.fit(X_tr, y_tr)
        metric = "R²"
        score = r2_score(y_te, model.predict(X_te))
        base = r2_score(y_te, dummy.predict(X_te))
        note += (" | " if note else "") + f"MAE {mean_absolute_error(y_te, model.predict(X_te)):,.2f}"

    imp = (
        pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    return ModelReport(
        task=task, target=target, n_rows=len(X), features=features,
        metric_name=metric, model_score=float(score), baseline_score=float(base),
        importances=imp, note=note,
    )
