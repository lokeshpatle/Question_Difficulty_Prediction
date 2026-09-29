from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV


def calibrate_model(model, X_validation, y_validation):
    """Calibrate a frozen fitted estimator on held-out validation data only.

    For modern sklearn releases, FrozenEstimator prevents the base model from
    being refit during calibration. A finite CV value is supplied because the
    FrozenEstimator path in current sklearn still validates the requested fold
    count. The number of folds is bounded by the smallest validation-class count.
    Older sklearn releases fall back to the historical cv='prefit' API.
    """
    import numpy as np

    y = np.asarray(y_validation)
    _, counts = np.unique(y, return_counts=True)
    if len(counts) < 2:
        raise ValueError("calibration_requires_at_least_two_classes")

    min_class_count = int(counts.min())
    if min_class_count < 2:
        raise ValueError("calibration_requires_at_least_two_examples_per_class")

    try:
        from sklearn.frozen import FrozenEstimator
    except ImportError:  # pragma: no cover - legacy sklearn compatibility
        wrapper = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        return wrapper.fit(X_validation, y_validation)

    cv_folds = min(5, min_class_count)
    wrapper = CalibratedClassifierCV(
        estimator=FrozenEstimator(model),
        method="isotonic",
        cv=cv_folds,
        ensemble=False,
    )
    return wrapper.fit(X_validation, y_validation)
