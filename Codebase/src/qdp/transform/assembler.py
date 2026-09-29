from __future__ import annotations

import numpy as np
import pandas as pd

from qdp.features.registry import FEATURES, INDICATORS


class FeatureAssembler:
    def build_engineered(self, rows):
        frame = pd.DataFrame(rows)
        ordered = [feature_id for feature_id, _, _ in FEATURES] + list(INDICATORS)
        for column in ordered:
            if column not in frame.columns:
                frame[column] = np.nan if column.startswith("F") and column not in {"F14", "F15"} else 0
        frame["F14"] = frame["F14"].fillna("Other")
        frame["F15"] = frame["F15"].fillna("Other")
        return frame[ordered].copy()
