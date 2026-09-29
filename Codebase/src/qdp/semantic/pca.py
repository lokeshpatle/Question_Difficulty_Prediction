from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


class SemanticPCA:
    def __init__(self, n_components=48, seed=42):
        self.n_components = int(n_components)
        self.seed = int(seed)
        self.model = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        max_components = min(X.shape[0], X.shape[1])
        if self.n_components > max_components:
            raise ValueError(f"pca_components_exceed_train_dimensions:{self.n_components}>{max_components}")
        if self.n_components < 1:
            raise ValueError("pca_components_must_be_positive")
        self.model = PCA(n_components=self.n_components, svd_solver="full", random_state=self.seed)
        self.model.fit(X)
        return self

    def transform(self, X):
        if self.model is None:
            raise RuntimeError("PCA not fitted")
        return np.asarray(self.model.transform(X), dtype=float)

    @property
    def explained_variance_ratio_(self):
        if self.model is None:
            raise RuntimeError("PCA not fitted")
        return self.model.explained_variance_ratio_
