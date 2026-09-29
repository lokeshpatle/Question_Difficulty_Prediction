from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from qdp.data.schema import LABEL_TO_INT, INT_TO_LABEL


def _encode(values):
    return np.asarray([LABEL_TO_INT[value] if isinstance(value, str) else int(value) for value in values], dtype=int)


def multiclass_brier(y_true, probabilities):
    y = _encode(y_true)
    p = np.asarray(probabilities, dtype=float)
    onehot = np.zeros_like(p)
    for i, cls in enumerate(y):
        onehot[i, int(cls)] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def evaluate(y_true, y_pred, probabilities=None, calibrated=False):
    yt = _encode(y_true)
    yp = _encode(y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(yt, yp, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2])
    result = {
        "macro_f1": float(f1_score(yt, yp, labels=[0, 1, 2], average="macro", zero_division=0)),
        "ordinal_mae": float(np.mean(np.abs(yt - yp))),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "accuracy": float(accuracy_score(yt, yp)),
        "per_class": {
            INT_TO_LABEL[i]: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
            for i in range(3)
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_row_normalized": _normalize_rows(cm).tolist(),
        "quadratic_kappa": float(cohen_kappa_score(yt, yp, weights="quadratic")) if len(set(yt.tolist())) > 1 else 1.0,
        "calibrated": bool(calibrated),
    }
    if calibrated and probabilities is not None:
        result["brier_score"] = multiclass_brier(yt, probabilities)
        result["reliability_curve"] = reliability_bins(yt, probabilities, bins=10)
    return result


def _normalize_rows(cm):
    cm = np.asarray(cm, dtype=float)
    denom = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, np.where(denom == 0, 1.0, denom))


def reliability_bins(y_true, probabilities, bins=10):
    y = _encode(y_true)
    p = np.asarray(probabilities, dtype=float)
    confidence = p.max(axis=1)
    prediction = p.argmax(axis=1)
    output = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (confidence >= lower) & ((confidence < upper) if index < bins - 1 else (confidence <= upper))
        if not mask.any():
            output.append({"lower": lower, "upper": upper, "count": 0, "mean_confidence": None, "accuracy": None})
            continue
        output.append({
            "lower": lower,
            "upper": upper,
            "count": int(mask.sum()),
            "mean_confidence": float(confidence[mask].mean()),
            "accuracy": float((prediction[mask] == y[mask]).mean()),
        })
    return output


def selection_key(metrics, run_id):
    return (-metrics["macro_f1"], metrics["ordinal_mae"], -metrics["balanced_accuracy"], str(run_id))
