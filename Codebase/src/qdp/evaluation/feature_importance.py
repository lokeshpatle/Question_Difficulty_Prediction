from __future__ import annotations

import numpy as np
from sklearn.inspection import permutation_importance


def permutation(model, X, y, columns, seed=42):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=seed, scoring="f1_macro", n_jobs=1)
    return {column: {"mean": float(mean), "std": float(std)} for column, mean, std in zip(columns, result.importances_mean, result.importances_std)}


def xgb_gain(model, columns):
    if hasattr(model, "feature_importances_"):
        return {column: float(value) for column, value in zip(columns, model.feature_importances_)}
    return {}
