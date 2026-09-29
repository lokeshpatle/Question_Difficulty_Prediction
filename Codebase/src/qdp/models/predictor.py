from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qdp.artifacts.checksums import verify_checksums
from qdp.artifacts.schema_hashing import make_schema
from qdp.data.schema import INT_TO_LABEL, make_record
from qdp.features.pipeline import FeatureExtractor
from qdp.features.registry import FEATURES, INDICATORS, EXTRACTOR_VERSION
from qdp.transform.assembler import FeatureAssembler
from qdp.utils.hashing import stable_json_hash


class InferencePreprocessor:
    def __init__(self, cfg, lexicons, nlp, embedder, rare_df, tfidf, tfidf_threshold, imputer, onehot, selector, pca, numeric_columns, engineered_columns, final_schema, use_selector):
        self.cfg = cfg
        self.lexicons = lexicons
        self.nlp = nlp
        self.embedder = embedder
        self.rare_df = rare_df
        self.tfidf = tfidf
        self.tfidf_threshold = tfidf_threshold
        self.imputer = imputer
        self.onehot = onehot
        self.selector = selector
        self.pca = pca
        self.numeric_columns = numeric_columns
        self.engineered_columns = engineered_columns
        self.final_schema = final_schema
        self.use_selector = bool(use_selector)

    def transform_records(self, records):
        extractor = FeatureExtractor(self.cfg, self.lexicons, self.nlp, self.embedder)
        extractor.rare_df = self.rare_df
        extractor.tfidf = self.tfidf
        extractor.tfidf_threshold = self.tfidf_threshold
        rows = extractor.extract_many(records)
        frame = FeatureAssembler().build_engineered(rows)
        frame["F14"] = frame["F14"].fillna("Other")
        frame["F15"] = frame["F15"].fillna("Other")
        X_num = self.imputer.transform(frame[self.numeric_columns])
        X_cat = self.onehot.transform(frame[["F14", "F15"]]) if self.onehot is not None else np.empty((len(records), 0), dtype=float)
        X_eng = np.hstack([X_num, X_cat])
        if self.use_selector:
            combined = X_eng
            if self.pca is not None and self.embedder is not None:
                units = [(record.question_id, "question", extractor._make_view(record).nl_view) for record in records]
                combined = np.hstack([combined, self.pca.transform(self.embedder.encode_units(units))])
            X = self.selector.transform(combined)
        else:
            X = X_eng
        if not np.isfinite(X).all():
            raise ValueError("inference_feature_matrix_contains_nan_or_inf")
        saved_versions = self.final_schema.get("extractor_versions", {})
        saved_extractor_version = saved_versions.get("FeatureExtractor")
        if saved_extractor_version != EXTRACTOR_VERSION:
            raise ValueError(
                f"extractor_version_mismatch: expected={saved_extractor_version} current={EXTRACTOR_VERSION}"
            )
        actual_schema = make_schema(
            self.final_schema["columns"],
            self.final_schema["dtypes"],
            missing_sentinels=self.final_schema.get("missing_sentinels"),
            family_map=self.final_schema.get("family_map"),
            extractor_versions={"FeatureExtractor": EXTRACTOR_VERSION},
            profile=self.final_schema.get("profile"),
        )
        if actual_schema["schema_hash"] != self.final_schema["schema_hash"]:
            raise ValueError(f"schema_hash_mismatch: expected={self.final_schema['schema_hash']} actual={actual_schema['schema_hash']}")
        if X.shape[1] != len(self.final_schema["columns"]):
            raise ValueError(f"inference_dimension_mismatch: {X.shape[1]} != {len(self.final_schema['columns'])}")
        return X


class Predictor:
    def __init__(self, preprocessor, model, metadata):
        self.preprocessor = preprocessor
        self.model = model
        self.metadata = metadata

    def _probability_payload(self, probabilities):
        classes = [int(value) if isinstance(value, (int, np.integer)) else value for value in self.model.classes_]
        mapped = {}
        for cls, probability in zip(classes, probabilities):
            if isinstance(cls, str):
                label = cls
            else:
                label = INT_TO_LABEL[int(cls)]
            mapped[label] = float(probability)
        # Every expected class must be represented.
        ordered = {label: float(mapped.get(label, 0.0)) for label in ("Easy", "Moderate", "Hard")}
        return ordered

    def predict_record(self, record):
        X = self.preprocessor.transform_records([record])
        probabilities = np.asarray(self.model.predict_proba(X)[0], dtype=float)
        classes = [int(value) if isinstance(value, (int, np.integer)) else value for value in self.model.classes_]
        best = min(
            range(len(probabilities)),
            key=lambda idx: (-probabilities[idx], int(classes[idx]) if not isinstance(classes[idx], str) else {"Easy": 0, "Moderate": 1, "Hard": 2}[classes[idx]]),
        )
        predicted = INT_TO_LABEL[int(classes[best])] if not isinstance(classes[best], str) else classes[best]
        output = {
            "question_id": record.question_id,
            "predicted_label": predicted,
            "probabilities": self._probability_payload(probabilities),
            "model_confidence": float(probabilities[best]),
        }
        output.update(self.metadata)
        return output

    def predict_batch(self, records):
        X = self.preprocessor.transform_records(records)
        probabilities = np.asarray(self.model.predict_proba(X), dtype=float)
        outputs = []
        for record, row in zip(records, probabilities):
            # Reuse the single-record logic only for label mapping; avoid re-transforming.
            classes = [int(value) if isinstance(value, (int, np.integer)) else value for value in self.model.classes_]
            best = min(range(len(row)), key=lambda idx: (-row[idx], int(classes[idx]) if not isinstance(classes[idx], str) else {"Easy": 0, "Moderate": 1, "Hard": 2}[classes[idx]]))
            predicted = INT_TO_LABEL[int(classes[best])] if not isinstance(classes[best], str) else classes[best]
            outputs.append({
                "question_id": record.question_id,
                "predicted_label": predicted,
                "probabilities": self._probability_payload(row),
                "model_confidence": float(row[best]),
                **self.metadata,
            })
        return outputs
