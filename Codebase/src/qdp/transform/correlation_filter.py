from __future__ import annotations

import numpy as np


class FeatureSelector:
    def __init__(self, threshold=0.90):
        self.threshold = float(threshold)
        self.keep_ = None
        self.columns_ = None
        self.variance_mask_ = None
        self.fit_rows_ = None

    def fit(self, X, columns, fit_rows=None):
        A = np.asarray(X, dtype=float)
        self.columns_ = list(columns)
        self.fit_rows_ = list(fit_rows) if fit_rows is not None else None
        variance = np.nanvar(A, axis=0)
        variance_mask = variance > 0.0
        self.variance_mask_ = variance_mask
        if not np.any(variance_mask):
            raise ValueError("all_features_constant")
        B = A[:, variance_mask]
        reduced_columns = [c for c, keep in zip(self.columns_, variance_mask) if keep]
        keep = np.ones(B.shape[1], dtype=bool)
        for i in range(B.shape[1]):
            if not keep[i]:
                continue
            for j in range(i + 1, B.shape[1]):
                if not keep[j]:
                    continue
                r = np.corrcoef(B[:, i], B[:, j])[0, 1]
                if np.isfinite(r) and abs(float(r)) > self.threshold:
                    keep[j] = False
        final_mask = np.zeros(len(self.columns_), dtype=bool)
        final_mask[np.where(variance_mask)[0]] = keep
        self.keep_ = final_mask
        return self

    def transform(self, X):
        if self.keep_ is None:
            raise RuntimeError("feature selector not fitted")
        return np.asarray(X, dtype=float)[:, self.keep_]

    def columns(self):
        if self.keep_ is None:
            raise RuntimeError("feature selector not fitted")
        return [c for c, keep in zip(self.columns_, self.keep_) if keep]


CorrelationFilter = FeatureSelector
