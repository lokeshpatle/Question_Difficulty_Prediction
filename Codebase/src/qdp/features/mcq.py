from __future__ import annotations

import math
import numpy as np
from qdp.preprocessing.tokenizer import word_tokens_regex


def cosine(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def extract(options, stem, stem_vector=None, option_vectors=None):
    if not options:
        return {"F18": 0, "F19": math.nan, "F20": math.nan, "F21": math.nan, "F22": math.nan}
    lengths = [len(word_tokens_regex(option)) for option in options]
    mean_length = sum(lengths) / len(lengths)
    out = {
        "F18": len(options),
        "F19": mean_length,
        "F20": float(sum((value - mean_length) ** 2 for value in lengths) / len(lengths)) if len(lengths) >= 2 else math.nan,
        "F21": math.nan,
        "F22": math.nan,
    }
    if stem_vector is not None and option_vectors and all(vector is not None for vector in option_vectors):
        out["F21"] = float(sum(cosine(stem_vector, vector) for vector in option_vectors) / len(option_vectors))
        pairs = [
            cosine(option_vectors[i], option_vectors[j])
            for i in range(len(option_vectors))
            for j in range(i + 1, len(option_vectors))
        ]
        if pairs:
            out["F22"] = float(sum(pairs) / len(pairs))
    return out
