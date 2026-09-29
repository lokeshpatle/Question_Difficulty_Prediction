from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml

from qdp.artifacts.checksums import verify_checksums
from qdp.artifacts.save_load import load_joblib, load_json
from qdp.config.schema import validate_config
from qdp.data.schema import make_record
from qdp.models.predictor import InferencePreprocessor, Predictor
from qdp.semantic.encoder import SemanticEncoder


def load_lexicons(root):
    def read(name):
        with (root / "config" / name).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    code = read("code_signatures.yaml")
    concept = read("concept_lexicon.yaml")
    return {
        "task": read("task_lexicon.yaml"),
        "bloom": read("bloom_lexicon.yaml"),
        "concepts": concept["concepts"],
        "technical_terms": concept["technical_terms"],
        "algorithm": read("algorithm_complexity.yaml")["weights"],
        "constraint_patterns": read("constraint_patterns.yaml")["patterns"],
        "negation": read("negation_lexicon.yaml")["terms"],
        "code_signatures": code["patterns"],
        "branch_keywords": code["branch_keywords"],
        "loop_keywords": code["loop_keywords"],
        "mcq_markers": read("mcq_markers.yaml")["markers"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-dir", default="artifacts/latest")
    parser.add_argument("--question", required=True)
    parser.add_argument("--question-id", default=None)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run = (root / args.run_dir).resolve()
    manifest = load_json(run / "run_manifest.json")
    verify_checksums(run, manifest["artifact_checksums"])
    cfg = load_json(run / "resolved_config.json")
    validate_config(cfg)
    schema = load_json(run / "feature_schema.json")
    mode = load_json(run / "production_feature_mode.json")
    rare = load_joblib(run / "rare_word_df.joblib")
    tfidf = load_joblib(run / "tfidf.joblib") if (run / "tfidf.joblib").exists() else None
    threshold = load_json(run / "tfidf_threshold.json")["threshold"] if (run / "tfidf_threshold.json").exists() else 0.0
    imputer = load_joblib(run / "imputer.joblib")
    onehot = load_joblib(run / "onehot.joblib")
    selector = load_joblib(run / "feature_selector.joblib")
    model = load_joblib(run / "calibrator.joblib") if manifest.get("calibrated") else load_joblib(run / "model.joblib")
    nlp = None
    profile = manifest["dependency_profile"]
    if profile not in {"P2", "P4"}:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    embedder = None
    pca = load_joblib(run / "semantic_pca.joblib") if (run / "semantic_pca.joblib").exists() else None
    if pca is not None:
        encoder_meta = load_json(run / "semantic_encoder.json")
        embedder = SemanticEncoder(
            encoder_meta["name"],
            root / "data/features/embedding_cache",
            cfg["runtime"]["embedding_batch_size"],
            max_seq_length=encoder_meta["max_seq_length"],
            revision=encoder_meta.get("revision"),
        ).load()
    preprocessing_meta = load_json(run / "preprocessing_manifest.json")
    preprocessor = InferencePreprocessor(
        cfg,
        load_lexicons(root),
        nlp,
        embedder,
        rare,
        tfidf,
        threshold,
        imputer,
        onehot,
        selector,
        pca,
        preprocessing_meta["numeric_columns"],
        preprocessing_meta["engineered_columns"],
        schema,
        bool(mode["use_final_selector"]),
    )
    metadata = {
        "calibrated": bool(manifest.get("calibrated", False)),
        "model_version": "qdp_xgb_v1" if manifest["selected_hyperparameters"].get("production") else "qdp_model_v1",
        "feature_schema_version": schema["version"],
        "dependency_profile": profile,
    }
    result = Predictor(preprocessor, model, metadata).predict_record(make_record(args.question, None, args.question_id))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
