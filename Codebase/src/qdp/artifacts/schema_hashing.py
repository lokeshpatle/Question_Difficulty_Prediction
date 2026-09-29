from __future__ import annotations

from qdp.features.registry import FEATURE_VERSION, EXTRACTOR_VERSION, FAMILIES, INDICATORS, schema_hash


def make_schema(columns, dtypes=None, missing_sentinels=None, family_map=None, extractor_versions=None, profile=None):
    payload = {
        "version": FEATURE_VERSION,
        "columns": list(columns),
        "dtypes": list(dtypes or ["float64"] * len(columns)),
        "missing_sentinels": missing_sentinels or {feature_id: "NaN" for feature_id, _, _ in __import__("qdp.features.registry", fromlist=["FEATURES"]).FEATURES},
        "family_map": family_map or {family: list(features) for family, features in FAMILIES.items()},
        "indicators": list(INDICATORS),
        "extractor_versions": extractor_versions or {"FeatureExtractor": EXTRACTOR_VERSION},
        "profile": profile,
    }
    payload["schema_hash"] = schema_hash(payload)
    return payload


def verify(expected_hash: str, actual_schema: dict):
    actual_hash = schema_hash(actual_schema)
    if actual_hash != expected_hash:
        raise ValueError(f"schema_hash_mismatch: expected={expected_hash} actual={actual_hash}")
    return True
