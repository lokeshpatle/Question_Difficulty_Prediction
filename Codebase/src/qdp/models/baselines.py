from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logistic(seed=42):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(solver="lbfgs", C=1.0, max_iter=2000, random_state=seed)),
    ])


def make_random_forest(seed=42):
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=1,
        random_state=seed,
    )
