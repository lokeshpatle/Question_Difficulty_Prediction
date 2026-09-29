from __future__ import annotations

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from qdp.features.registry import I_TYPES, Q_TYPES


class CategoricalEncoder:
    def __init__(self):
        self.model = OneHotEncoder(
            categories=[Q_TYPES, I_TYPES],
            handle_unknown="ignore",
            sparse_output=False,
            dtype=float,
        )
        self._names = [f"question_type={value}" for value in Q_TYPES] + [f"instruction_type={value}" for value in I_TYPES]

    def fit(self, frame):
        self.model.fit(frame[["F14", "F15"]])
        return self

    def transform(self, frame):
        return np.asarray(self.model.transform(frame[["F14", "F15"]]), dtype=float)

    def names(self):
        return list(self._names)
