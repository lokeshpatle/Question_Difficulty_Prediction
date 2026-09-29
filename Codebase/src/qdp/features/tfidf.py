from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def fit(texts, percentile=90.0):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 1),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
        norm="l2",
        stop_words=None,
    )
    matrix = vectorizer.fit_transform(texts)
    data = matrix.data
    threshold = float(np.percentile(data, percentile)) if data.size else 0.0
    return vectorizer, threshold


def extract(texts, vectorizer, threshold):
    matrix = vectorizer.transform(texts)
    output = []
    for row in matrix:
        values = row.data
        output.append({
            "F23": float(values.mean()) if len(values) else np.nan,
            "F24": float(values.max()) if len(values) else np.nan,
            "F25": int((values > threshold).sum()),
        })
    return output
