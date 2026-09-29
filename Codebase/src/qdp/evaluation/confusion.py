from __future__ import annotations

import numpy as np


def row_normalize(matrix):
    matrix = np.asarray(matrix, dtype=float)
    denom = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, np.where(denom == 0, 1.0, denom))
