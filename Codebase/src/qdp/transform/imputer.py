from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer


class TrainOnlyImputer:
    def __init__(self):
        self.model = SimpleImputer(strategy="median", add_indicator=False)
        self.fit_rows = None

    def fit(self, frame, fit_indices=None):
        if fit_indices is None:
            fit_frame = frame
            self.fit_rows = None
        else:
            indices = np.asarray(list(fit_indices), dtype=int)
            if indices.size == 0:
                raise ValueError("fit_indices must contain at least one row")
            if np.any(indices < 0) or np.any(indices >= len(frame)):
                raise IndexError("fit_indices contains an out-of-range row")
            fit_frame = frame.iloc[indices]
            self.fit_rows = indices.tolist()
        self.model.fit(fit_frame)
        return self

    def transform(self, frame):
        return np.asarray(self.model.transform(frame), dtype=float)
