from __future__ import annotations

from typing import Any


REQUIRED_LABELS = ("Easy", "Moderate", "Hard")


def validate_config(cfg: dict[str, Any]) -> None:
    required_sections = [
        "corpus", "ingest", "validation", "dedup", "split", "preprocess",
        "features", "semantic", "selection", "training", "interpret", "seed", "runtime",
    ]
    missing = [section for section in required_sections if section not in cfg]
    if missing:
        raise ValueError(f"missing_config_sections:{missing}")
    if int(cfg["corpus"]["min_records"]) < 1:
        raise ValueError("corpus.min_records_must_be_positive")
    ratios = [float(value) for value in cfg["split"]["ratios"]]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split.ratios_must_sum_to_one")
    if not 0 < float(cfg["dedup"]["near_threshold"]) <= 1:
        raise ValueError("dedup.near_threshold_out_of_range")
    if int(cfg["semantic"]["pca_components"]) < 1:
        raise ValueError("semantic.pca_components_must_be_positive")
    if not 0 < float(cfg["selection"]["correlation_threshold"]) < 1:
        raise ValueError("selection.correlation_threshold_out_of_range")
    if int(cfg["preprocess"]["max_chars"]) <= 0:
        raise ValueError("preprocess.max_chars_must_be_positive")
    abbreviations = cfg["preprocess"].get("sentence_abbreviations", [])
    if not isinstance(abbreviations, list) or any(not isinstance(value, str) for value in abbreviations):
        raise ValueError("preprocess.sentence_abbreviations_must_be_string_list")
    if float(cfg["ingest"]["max_reject_ratio"]) < 0 or float(cfg["ingest"]["max_reject_ratio"]) > 1:
        raise ValueError("ingest.max_reject_ratio_out_of_range")
    seed_keys = {"split", "minhash", "pca", "model", "cv", "permutation_importance", "shap_sample"}
    if seed_keys - set(cfg["seed"]):
        raise ValueError("missing_seed_keys")
