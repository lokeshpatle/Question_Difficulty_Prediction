from __future__ import annotations

import copy
import importlib
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from qdp.artifacts.checksums import checksum_dir
from qdp.artifacts.save_load import atomic_json, save_joblib
from qdp.artifacts.schema_hashing import make_schema
from qdp.config.loader import load_config
from qdp.data.deduplicator import exact_deduplicate
from qdp.data.loader import CorpusLoader, IngestError
from qdp.data.schema import LABEL_TO_INT, with_split
from qdp.data.splitter import group_split, repeated_group_cv
from qdp.data.validator import validate_splits, validate_training
from qdp.evaluation.feature_importance import permutation, xgb_gain
from qdp.evaluation.metrics import evaluate, selection_key
from qdp.features.pipeline import FeatureExtractor
from qdp.features.registry import FEATURES, FAMILIES, INDICATORS, I_TYPES, Q_TYPES
from qdp.models.baselines import make_logistic, make_random_forest
from qdp.models.calibration import calibrate_model
from qdp.models.xgb import grid, make_xgb
from qdp.preprocessing.normalize import normalize_working_text
from qdp.semantic.encoder import SemanticEncoder
from qdp.semantic.pca import SemanticPCA
from qdp.transform.assembler import FeatureAssembler
from qdp.transform.correlation_filter import FeatureSelector
from qdp.transform.encoder import CategoricalEncoder
from qdp.transform.imputer import TrainOnlyImputer
from qdp.utils.hashing import file_sha256, stable_json_hash
from qdp.utils.logging import setup_logger
from qdp.utils.seeding import seed_everything


class PipelineError(RuntimeError):
    pass


class QDPTrainer:
    def __init__(self, project_root: Path, cfg: dict, run_dir: Path, profile: str = "auto"):
        self.root = Path(project_root).resolve()
        self.cfg = cfg
        self.run_dir = (self.root / run_dir).resolve() if not Path(run_dir).is_absolute() else Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.requested_profile = profile
        self.logger = setup_logger(self.run_dir)

    def _read_yaml(self, name):
        import yaml
        with (self.root / "config" / name).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def load_labels(self):
        return self._read_yaml("label_normalization.yaml")["labels"]

    def load_lexicons(self):
        code = self._read_yaml("code_signatures.yaml")
        concept = self._read_yaml("concept_lexicon.yaml")
        return {
            "task": self._read_yaml("task_lexicon.yaml"),
            "bloom": self._read_yaml("bloom_lexicon.yaml"),
            "concepts": concept["concepts"],
            "technical_terms": concept["technical_terms"],
            "algorithm": self._read_yaml("algorithm_complexity.yaml")["weights"],
            "constraint_patterns": self._read_yaml("constraint_patterns.yaml")["patterns"],
            "negation": self._read_yaml("negation_lexicon.yaml")["terms"],
            "code_signatures": code["patterns"],
            "branch_keywords": code["branch_keywords"],
            "loop_keywords": code["loop_keywords"],
            "mcq_markers": self._read_yaml("mcq_markers.yaml")["markers"],
        }

    def _availability(self):
        packages = ["numpy", "pandas", "sklearn", "joblib", "pyarrow", "yaml", "xgboost", "spacy", "textstat", "sentence_transformers"]
        available = {}
        for package in packages:
            try:
                importlib.import_module(package)
                available[package] = True
            except Exception:
                available[package] = False
        return available

    def runtime_profile(self):
        available = self._availability()
        mandatory = {"numpy", "pandas", "sklearn", "joblib", "yaml", "xgboost"}
        missing = sorted(name for name in mandatory if not available.get(name, False))
        if missing:
            return "BLOCKED", available, missing
        semantic_ok = False
        parser_ok = False
        if available["sentence_transformers"]:
            try:
                SemanticEncoder(
                    self.cfg["semantic"]["encoder"],
                    self.root / "data/features/embedding_cache",
                    self.cfg["runtime"]["embedding_batch_size"],
                    max_seq_length=256,
                    revision=self.cfg["semantic"].get("encoder_revision", "configured"),
                ).load()
                semantic_ok = True
            except Exception:
                semantic_ok = False
        if available["spacy"]:
            try:
                import spacy
                spacy.load("en_core_web_sm", disable=["ner", "textcat"])
                parser_ok = True
            except Exception:
                parser_ok = False
        readability_ok = available["textstat"]
        optional_missing = sum(not value for value in (semantic_ok, parser_ok, readability_ok, available["pyarrow"]))
        feature_missing = (not semantic_ok, not parser_ok, not readability_ok, not available["pyarrow"])
        if feature_missing == (False, False, False, False):
            detected = "P0"
        elif feature_missing == (True, False, False, False):
            detected = "P1"
        elif feature_missing == (False, True, False, False):
            detected = "P2"
        elif feature_missing == (False, False, True, False):
            detected = "P3"
        elif feature_missing[:3].count(True) >= 2:
            # P4 is the explicitly defined floor/smoke profile. It is only selected
            # when multiple primary feature dependencies are unavailable.
            detected = "P4"
        else:
            return "BLOCKED", {**available, "semantic_model_local": semantic_ok, "spacy_model_local": parser_ok}, [
                "unsupported_dependency_combination"
            ]
        # Resolve the requested profile only when the environment satisfies that
        # profile's dependency contract. P0/auto falls back to a supported detected
        # profile; contradictory explicit profiles fail loudly instead of partially
        # running with hidden feature loss.
        requested = self.requested_profile
        if requested in {"auto", None, "", "P0"}:
            profile = detected
        else:
            profile = requested
            valid = {"P0", "P1", "P2", "P3", "P4"}
            if profile not in valid:
                raise PipelineError(f"unknown_dependency_profile:{profile}")
            requirements = {
                "P0": (semantic_ok, parser_ok, readability_ok, available["pyarrow"]),
                "P1": (parser_ok, readability_ok, available["pyarrow"]),
                "P2": (semantic_ok, readability_ok, available["pyarrow"]),
                "P3": (semantic_ok, parser_ok, available["pyarrow"]),
                "P4": (True, True, True, True),
            }
            if not all(requirements[profile]):
                return "BLOCKED", {**available, "semantic_model_local": semantic_ok, "spacy_model_local": parser_ok}, [
                    f"requested_profile_incompatible:{profile}"
                ]
        return profile, {**available, "semantic_model_local": semantic_ok, "spacy_model_local": parser_ok}, []

    def load_nlp(self, profile):
        if profile in {"P2", "P4"}:
            return None
        import spacy
        try:
            return spacy.load("en_core_web_sm", disable=["ner", "textcat"])
        except Exception as exc:
            raise PipelineError(f"spaCy model unavailable for profile {profile}:{exc}") from exc

    def load_embedder(self, profile):
        if profile in {"P1", "P4"} or not self.cfg["features"].get("use_semantic_embedding", True):
            return None
        encoder = SemanticEncoder(
            self.cfg["semantic"]["encoder"],
            self.root / "data/features/embedding_cache",
            self.cfg["runtime"]["embedding_batch_size"],
            max_seq_length=256,
            revision=self.cfg["semantic"].get("encoder_revision", "configured"),
        )
        encoder.load()
        return encoder

    def _active_feature_ids(self, profile):
        active = [fid for fid, _, _ in FEATURES]
        if profile in {"P1", "P4"} or not self.cfg["features"].get("use_semantic_embedding", True):
            active = [fid for fid in active if fid not in {"F21", "F22", "F29"}]
        if profile in {"P2", "P4"}:
            active = [fid for fid in active if fid not in {"F09", "F10", "F11", "F12", "F13"}]
        if profile == "P3":
            active = [fid for fid in active if fid not in {"F06", "F07", "F08"}]
        if not self.cfg["features"].get("use_lexical", True):
            active = [fid for fid in active if fid not in FAMILIES["lexical"]]
        for family, key in {
            "readability": "use_readability", "syntax": "use_syntax", "task": "use_task", "mcq": "use_mcq",
            "tfidf": "use_tfidf", "semantic_scalar": "use_semantic_scalar", "programming": "use_programming",
        }.items():
            if not self.cfg["features"].get(key, True):
                active = [fid for fid in active if fid not in FAMILIES[family]]
        return active

    def _class_weight_policy(self, labels):
        labels = np.asarray(labels, dtype=int)
        counts = np.bincount(labels, minlength=3)
        if counts.min() <= 0:
            raise PipelineError("all_three_classes_required_for_class_weight_policy")
        ratio = float(counts.max() / counts.min())
        if ratio <= 1.5:
            return None, ratio, False
        total = len(labels)
        mapping = {class_id: total / (3.0 * int(count)) for class_id, count in enumerate(counts)}
        return mapping, ratio, True

    def _sample_weights(self, labels):
        mapping, ratio, applied = self._class_weight_policy(labels)
        if not applied:
            return None, ratio, False
        labels = np.asarray(labels, dtype=int)
        return np.asarray([mapping[int(label)] for label in labels], dtype=float), ratio, True

    def _fit_feature_bundle(self, records, profile, active_ids=None):
        lexicons = self.load_lexicons()
        nlp = self.load_nlp(profile)
        encoder = self.load_embedder(profile)
        extractor = FeatureExtractor(self.cfg, lexicons, nlp, encoder, self.root / "data/features/cache")
        train_records = [record for record in records if record.split == "train"]
        extractor.fit_artifacts(train_records)
        rows = extractor.extract_many(records)
        diagnostics = [
            {
                "question_id": record.question_id,
                "semantic_truncated": bool(row.get("semantic_truncated", False)),
                "code_parse_exact": int(row.get("code_parse_exact", 0)),
                "code_detection_tiers": list(row.get("code_detection_tiers", [])),
                "mcq_detection_tier": row.get("mcq_detection_tier"),
            }
            for record, row in zip(records, rows)
        ]
        frame = FeatureAssembler().build_engineered(rows)
        if active_ids is None:
            active_ids = self._active_feature_ids(profile)
        feature_ids = [fid for fid, _, _ in FEATURES]
        task_active = any(fid in active_ids for fid in FAMILIES["task"])
        active_columns = [fid for fid in feature_ids if fid in active_ids]
        category_columns = [fid for fid in ("F14", "F15") if task_active and fid in active_columns]
        # Indicators are retained whenever their governing family remains active.
        active_indicator_names = []
        if any(fid in active_ids for fid in ["F18", "F19", "F20", "F21", "F22"]): active_indicator_names.append("mcq_detected")
        if any(fid in active_ids for fid in FAMILIES["programming"]): active_indicator_names.append("code_detected")
        if "F31" in active_ids or "F32" in active_ids: active_indicator_names.append("input_size_detected")
        if any(fid in active_ids for fid in FAMILIES["syntax"]): active_indicator_names.append("parse_ok")
        if "F29" in active_ids: active_indicator_names.append("multi_segment")
        numeric_columns = [fid for fid in active_ids if fid not in {"F14", "F15"}] + active_indicator_names
        # Enforce deterministic order according to the registry.
        registry_order = [fid for fid, _, _ in FEATURES] + INDICATORS
        numeric_columns = [column for column in registry_order if column in numeric_columns]
        train_idx = np.asarray([i for i, record in enumerate(records) if record.split == "train"], dtype=int)
        numeric_frame = frame[numeric_columns].copy()
        if numeric_frame.shape[1] == 0:
            raise PipelineError("no_active_numeric_features")
        all_nan = np.isnan(numeric_frame.iloc[train_idx].to_numpy(dtype=float)).all(axis=0)
        usable_numeric = [column for column, bad in zip(numeric_columns, all_nan) if not bad]
        if not usable_numeric:
            raise PipelineError("all_numeric_features_missing")
        imputer = TrainOnlyImputer().fit(numeric_frame[usable_numeric], fit_indices=train_idx.tolist())
        X_numeric = imputer.transform(numeric_frame[usable_numeric])
        category_frame = frame[["F14", "F15"]].copy()
        category_frame["F14"] = category_frame["F14"].fillna("Other")
        category_frame["F15"] = category_frame["F15"].fillna("Other")
        onehot = None
        X_cat = np.empty((len(records), 0), dtype=float)
        cat_names = []
        if category_columns:
            onehot = CategoricalEncoder().fit(category_frame.iloc[train_idx])
            X_cat = onehot.transform(category_frame)
            cat_names = onehot.names()
        engineered_columns = usable_numeric + cat_names
        X_eng = np.hstack([X_numeric, X_cat])

        semantic_pca = None
        semantic_embeddings = None
        full_columns = list(engineered_columns)
        combined = X_eng
        if self.cfg["features"].get("use_semantic_embedding", True) and encoder is not None:
            units = [(record.question_id, "question", extractor._make_view(record).nl_view) for record in records]
            semantic_embeddings = encoder.encode_units(units)
            semantic_pca = SemanticPCA(int(self.cfg["semantic"]["pca_components"]), int(self.cfg["seed"]["pca"]))
            semantic_pca.fit(semantic_embeddings[train_idx])
            semantic_block = semantic_pca.transform(semantic_embeddings)
            combined = np.hstack([combined, semantic_block])
            full_columns.extend([f"sem_pca_{i:03d}" for i in range(semantic_block.shape[1])])
        selector = FeatureSelector(float(self.cfg["selection"]["correlation_threshold"])).fit(combined[train_idx], full_columns, fit_rows=train_idx.tolist())
        X_final = selector.transform(combined)
        if not np.isfinite(X_eng).all() or not np.isfinite(X_final).all():
            raise PipelineError("final_feature_matrix_contains_nan_or_inf")
        return {
            "extractor": extractor,
            "encoder": encoder,
            "X_eng_raw": X_eng,
            "engineered_columns": engineered_columns,
            "X_full": combined,
            "full_columns": full_columns,
            "X_final": X_final,
            "final_columns": selector.columns(),
            "imputer": imputer,
            "onehot": onehot,
            "selector": selector,
            "numeric_columns": usable_numeric,
            "semantic_pca": semantic_pca,
            "semantic_embeddings": semantic_embeddings,
            "profile": profile,
            "active_feature_ids": active_ids,
            "feature_diagnostics": diagnostics,
        }

    def _transform_with_bundle(self, records, bundle):
        if not records:
            return (
                np.empty((0, len(bundle["engineered_columns"]))),
                np.empty((0, len(bundle["final_columns"]))),
                np.empty((0, len(bundle["full_columns"]))),
                [],
            )
        rows = bundle["extractor"].extract_many(records)
        diagnostics = [
            {
                "question_id": record.question_id,
                "semantic_truncated": bool(row.get("semantic_truncated", False)),
                "code_parse_exact": int(row.get("code_parse_exact", 0)),
                "code_detection_tiers": list(row.get("code_detection_tiers", [])),
                "mcq_detection_tier": row.get("mcq_detection_tier"),
            }
            for record, row in zip(records, rows)
        ]
        frame = FeatureAssembler().build_engineered(rows)
        frame["F14"] = frame["F14"].fillna("Other")
        frame["F15"] = frame["F15"].fillna("Other")
        numeric = frame[bundle["numeric_columns"]]
        X_num = bundle["imputer"].transform(numeric)
        X_cat = bundle["onehot"].transform(frame[["F14", "F15"]]) if bundle["onehot"] is not None else np.empty((len(records), 0), dtype=float)
        X_eng = np.hstack([X_num, X_cat])
        combined = X_eng
        if bundle["semantic_pca"] is not None:
            units = [(record.question_id, "question", bundle["extractor"]._make_view(record).nl_view) for record in records]
            embeddings = bundle["encoder"].encode_units(units)
            combined = np.hstack([combined, bundle["semantic_pca"].transform(embeddings)])
        X_final = bundle["selector"].transform(combined)
        if not np.isfinite(X_eng).all() or not np.isfinite(X_final).all():
            raise PipelineError("transformed_feature_matrix_contains_nan_or_inf")
        return X_eng, X_final, combined, diagnostics

    def _fit_baseline(self, name, model, X, y, train_idx, val_idx, train_weights=None):
        if name == "baseline_logistic":
            model.fit(X[train_idx], y[train_idx], model__sample_weight=train_weights) if train_weights is not None else model.fit(X[train_idx], y[train_idx])
        else:
            model.fit(X[train_idx], y[train_idx], sample_weight=train_weights)
        train_metrics = evaluate(y[train_idx], model.predict(X[train_idx]))
        val_metrics = evaluate(y[val_idx], model.predict(X[val_idx]))
        return model, train_metrics, val_metrics

    def _fit_xgb_candidate(self, depth, learning_rate, subsample, X, y, train_idx, val_idx, train_weight, val_weight):
        model = make_xgb(depth, learning_rate, subsample, self.cfg["seed"]["model"], n_estimators=2000, early_stopping_rounds=50)
        kwargs = {"eval_set": [(X[val_idx], y[val_idx])], "verbose": False}
        if train_weight is not None:
            kwargs["sample_weight"] = train_weight
            kwargs["sample_weight_eval_set"] = [val_weight]
        model.fit(X[train_idx], y[train_idx], **kwargs)
        best_iteration = getattr(model, "best_iteration", None)
        best_n = int(best_iteration + 1) if best_iteration is not None else 2000
        return model, evaluate(y[train_idx], model.predict(X[train_idx])), evaluate(y[val_idx], model.predict(X[val_idx])), best_n

    def _select_standard(self, bundle, records):
        Xeng, Xfinal = bundle["X_eng_raw"], bundle["X_final"]
        y = np.asarray([LABEL_TO_INT[r.label] for r in records], dtype=int)
        train_idx = np.asarray([i for i, r in enumerate(records) if r.split == "train"], dtype=int)
        val_idx = np.asarray([i for i, r in enumerate(records) if r.split == "validation"], dtype=int)
        train_weights, ratio, weights_applied = self._sample_weights(y[train_idx])
        candidates = []
        for name, factory in (("baseline_logistic", make_logistic), ("baseline_rf", make_random_forest)):
            model, train_metrics, val_metrics = self._fit_baseline(name, factory(self.cfg["seed"]["model"]), Xeng, y, train_idx, val_idx, train_weights)
            candidates.append({"key": selection_key(val_metrics, name), "run_id": name, "model": model, "train": train_metrics, "validation": val_metrics, "matrix": "engineered", "hparams": None})
        best_xgb = None
        for depth, lr, subsample in (grid() if self.cfg["training"]["hyperparameter_search"] else [(6, 0.05, 0.8)]):
            val_counts = np.bincount(y[val_idx], minlength=3)
            mapping = None
            val_weight = None
            if weights_applied:
                mapping = {class_id: len(train_idx) / (3.0 * int(count)) for class_id, count in enumerate(np.bincount(y[train_idx], minlength=3)) if count}
                val_weight = np.asarray([mapping[int(cls)] for cls in y[val_idx]], dtype=float)
            model, train_metrics, val_metrics, best_n = self._fit_xgb_candidate(depth, lr, subsample, Xfinal, y, train_idx, val_idx, train_weights, val_weight)
            rid = f"xgb_d{depth}_l{lr}_s{subsample}"
            item = {"key": selection_key(val_metrics, rid), "run_id": rid, "model": model, "train": train_metrics, "validation": val_metrics, "matrix": "final", "hparams": {"max_depth": depth, "learning_rate": lr, "subsample": subsample, "n_estimators": best_n, "best_iteration": best_n - 1, "early_stopping_rounds": 50}}
            candidates.append(item)
            if best_xgb is None or item["key"] < best_xgb["key"]:
                best_xgb = item
        selected = min(candidates, key=lambda item: item["key"])
        return candidates, selected, best_xgb, train_idx, val_idx, ratio, weights_applied

    def _fit_calibration_base_model(self, selected, bundle, records):
        """Fit the selected classifier on train rows only for post-hoc calibration.

        TDD2 requires the validation set to remain disjoint from the base model
        when calibration is enabled. The uncalibrated production path still
        uses train+validation in _fit_final_model.
        """
        train_idx = np.asarray([i for i, r in enumerate(records) if r.split == "train"], dtype=int)
        y = np.asarray([LABEL_TO_INT[r.label] for r in records], dtype=int)
        train_weight_map, _, weights_applied = self._class_weight_policy(y[train_idx])
        train_weights = (
            np.asarray([train_weight_map[int(label)] for label in y[train_idx]], dtype=float)
            if weights_applied
            else None
        )
        if selected["run_id"].startswith("xgb_"):
            hp = selected["hparams"]
            model = make_xgb(
                hp["max_depth"],
                hp["learning_rate"],
                hp["subsample"],
                self.cfg["seed"]["model"],
                n_estimators=hp["n_estimators"],
                early_stopping_rounds=None,
            )
            kwargs = {"verbose": False}
            if train_weights is not None:
                kwargs["sample_weight"] = train_weights
            model.fit(bundle["X_final"][train_idx], y[train_idx], **kwargs)
            return model, "final"
        model = copy.deepcopy(selected["model"])
        if selected["run_id"] == "baseline_logistic":
            if train_weights is not None:
                model.fit(bundle["X_eng_raw"][train_idx], y[train_idx], model__sample_weight=train_weights)
            else:
                model.fit(bundle["X_eng_raw"][train_idx], y[train_idx])
        else:
            model.fit(bundle["X_eng_raw"][train_idx], y[train_idx], sample_weight=train_weights)
        return model, "engineered"

    def _fit_final_model(self, selected, bundle, records):
        fit_idx = np.asarray([i for i, r in enumerate(records) if r.split in {"train", "validation"}], dtype=int)
        y = np.asarray([LABEL_TO_INT[r.label] for r in records], dtype=int)
        train_idx = np.asarray([i for i, r in enumerate(records) if r.split == "train"], dtype=int)
        final_weight_map, _, final_weights_applied = self._class_weight_policy(y[train_idx])
        final_weights = np.asarray([final_weight_map[int(label)] for label in y[fit_idx]], dtype=float) if final_weights_applied else None
        if selected["run_id"].startswith("xgb_"):
            hp = selected["hparams"]
            model = make_xgb(hp["max_depth"], hp["learning_rate"], hp["subsample"], self.cfg["seed"]["model"], n_estimators=hp["n_estimators"], early_stopping_rounds=None)
            if final_weights is not None:
                model.fit(bundle["X_final"][fit_idx], y[fit_idx], sample_weight=final_weights, verbose=False)
            else:
                model.fit(bundle["X_final"][fit_idx], y[fit_idx], verbose=False)
            return model, "final"
        model = copy.deepcopy(selected["model"])
        if selected["run_id"] == "baseline_logistic":
            if final_weights is not None:
                model.fit(bundle["X_eng_raw"][fit_idx], y[fit_idx], model__sample_weight=final_weights)
            else:
                model.fit(bundle["X_eng_raw"][fit_idx], y[fit_idx])
        else:
            model.fit(bundle["X_eng_raw"][fit_idx], y[fit_idx], sample_weight=final_weights)
        return model, "engineered"

    def _config_fingerprint(self):
        files = {str(path.relative_to(self.root)): file_sha256(path) for path in sorted((self.root / "config").glob("*.yaml")) if path.is_file()}
        return stable_json_hash(files), files

    def _package_versions(self):
        names = ["numpy", "pandas", "scikit-learn", "xgboost", "spacy", "textstat", "sentence-transformers", "pyarrow", "joblib", "PyYAML"]
        output = {}
        for name in names:
            try:
                output[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                output[name] = None
        return output

    def _save_matrix(self, directory: Path, ids, splits, X, columns):
        directory.mkdir(parents=True, exist_ok=True)
        if importlib.util.find_spec("pyarrow") is None:
            return False
        frame = pd.DataFrame(X, columns=columns)
        frame.insert(0, "question_id", ids)
        frame.insert(1, "split", splits)
        for split in ("train", "validation", "test"):
            frame[frame["split"] == split].to_parquet(directory / f"{split}.parquet", index=False)
        return True

    def _interpret(self, model, X_val, y_val, columns):
        payloads = {
            "feature_importance.json": xgb_gain(model, columns),
            "permutation_importance.json": permutation(model, X_val, y_val, columns, self.cfg["seed"]["permutation_importance"]),
        }
        for filename, payload in payloads.items():
            atomic_json(self.run_dir / filename, payload)
        shap_payload = {"status": "skipped", "reason": "shap_unavailable_or_not_installed"}
        try:
            import shap
            sample_size = min(int(self.cfg["interpret"]["shap_sample_size"]), len(X_val))
            rng = np.random.RandomState(self.cfg["seed"]["shap_sample"])
            indices = np.sort(rng.choice(len(X_val), size=sample_size, replace=False)) if sample_size else []
            values = shap.TreeExplainer(model).shap_values(X_val[indices])
            if isinstance(values, list):
                summary = [np.mean(np.abs(np.asarray(value)), axis=0).tolist() for value in values]
            else:
                summary = np.mean(np.abs(np.asarray(values)), axis=0).tolist()
            shap_payload = {"status": "ok", "sample_size": int(sample_size), "mean_abs_values": summary, "columns": columns, "causality_statement": "Feature importance describes model reliance and does not establish causality."}
        except Exception as exc:
            shap_payload["reason"] = str(exc)
        atomic_json(self.run_dir / "shap_summary.json", shap_payload)

    def train(self, corpus_path=None, allow_test_reevaluation=False):
        seed_everything(int(self.cfg["seed"]["model"]))
        if (self.run_dir / "test_evaluated.marker").exists() and not allow_test_reevaluation:
            raise PipelineError("test_already_evaluated_for_run")
        profile, availability, missing = self.runtime_profile()
        if profile == "BLOCKED":
            raise PipelineError(f"missing_mandatory_dependencies:{missing}")
        atomic_json(self.run_dir / "environment_preflight.json", {"dependency_profile": profile, "availability": availability})
        atomic_json(self.run_dir / "resolved_config.json", self.cfg)
        labels = self.load_labels()
        loader = CorpusLoader(labels, self.cfg["ingest"]["max_reject_ratio"], self.cfg["preprocess"]["max_chars"])
        source = Path(corpus_path or self.cfg["corpus"]["path"])
        source = self.root / source if not source.is_absolute() else source
        rejected_path = self.run_dir / "rejected_records.jsonl"
        try:
            records = loader.load(source, rejected_path)
        except IngestError as exc:
            raise PipelineError(str(exc)) from exc
        records, conflicts, duplicate_report = exact_deduplicate(records)
        if conflicts:
            with rejected_path.open("a", encoding="utf-8") as handle:
                for group in conflicts:
                    handle.write(json.dumps({"line": None, "reason": "duplicate_label_conflict", "record": [record.as_dict() for _, record in group]}, ensure_ascii=False) + "\n")
        with (self.run_dir / "duplicate_report.jsonl").open("w", encoding="utf-8") as handle:
            for item in duplicate_report:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        atomic_json(self.run_dir / "duplicate_summary.json", {"groups": len(duplicate_report), "conflicts": len(conflicts)})
        validate_training(records, int(self.cfg["corpus"]["min_records"]), int(self.cfg["validation"]["min_class_count"]))
        records, split_info = group_split(
            records,
            tuple(self.cfg["split"]["ratios"]),
            seed=int(self.cfg["seed"]["split"]),
            minhash_seed=int(self.cfg["seed"]["minhash"]),
            max_ratio_deviation=float(self.cfg["split"]["max_ratio_deviation"]),
            H=int(self.cfg["dedup"]["minhash_permutations"]),
            bands=int(self.cfg["dedup"]["lsh_bands"]),
            threshold=float(self.cfg["dedup"]["near_threshold"]),
            small_dataset_threshold=int(self.cfg["split"]["small_dataset_threshold"]),
            cv_seed=int(self.cfg["seed"]["cv"]),
        )
        if not split_info.get("cv_folds"):
            validate_splits(records, int(self.cfg["validation"]["min_class_count"]))
        else:
            # Small-data fallback reserves a test set and uses repeated group CV in the pool.
            test_rows = [record for record in records if record.split == "test"]
            if {record.label for record in test_rows} != {"Easy", "Moderate", "Hard"}:
                raise PipelineError("test_split_missing_class")
        canonical = self.root / "data/processed/canonical.jsonl"
        canonical.parent.mkdir(parents=True, exist_ok=True)
        with canonical.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")
        atomic_json(self.run_dir / "split_info.json", split_info)

        fit_records = [record for record in records if record.split != "test"]
        test_records = [record for record in records if record.split == "test"]
        bundle = self._fit_feature_bundle(fit_records, profile)
        y_fit = np.asarray([LABEL_TO_INT[r.label] for r in fit_records], dtype=int)
        selection_cv_metrics = None
        if split_info.get("cv_folds"):
            candidates, cv_fold_results = self._select_cv_on_pool(fit_records, profile, split_info["cv_folds"])
            selected = min(candidates, key=lambda item: item["key"])
            best_xgb = min((item for item in candidates if item["run_id"].startswith("xgb_")), key=lambda item: item["key"], default=None)
            selection_cv_metrics = selected["validation"]
            # The full non-test pool is used for the final fit after CV selection.
            train_idx = np.arange(len(fit_records), dtype=int)
            # First fold's validation indices are used only for optional calibration and
            # validation-side interpretability; they never touch the held-out test set.
            first_fold = cv_fold_results[0][0]
            first_val_ids = set(split_info["cv_folds"][0]["validation_ids"])
            val_idx = np.asarray([i for i, r in enumerate(fit_records) if r.question_id in first_val_ids], dtype=int)
            counts = np.bincount(y_fit, minlength=3)
            class_ratio = float(counts.max() / counts.min())
            weights_applied = class_ratio > 1.5
        else:
            candidates, selected, best_xgb, train_idx, val_idx, class_ratio, weights_applied = self._select_standard(bundle, fit_records)
        calibration_enabled = bool(self.cfg["training"].get("calibrate", False))
        if calibration_enabled:
            final_model, production_mode = self._fit_calibration_base_model(selected, bundle, fit_records)
        else:
            final_model, production_mode = self._fit_final_model(selected, bundle, fit_records)

        # Persist preprocessing artifacts needed by inference.
        save_joblib(self.run_dir / "rare_word_df.joblib", bundle["extractor"].rare_df)
        if bundle["extractor"].tfidf is not None:
            save_joblib(self.run_dir / "tfidf.joblib", bundle["extractor"].tfidf)
            atomic_json(self.run_dir / "tfidf_threshold.json", {"threshold": bundle["extractor"].tfidf_threshold})
        save_joblib(self.run_dir / "imputer.joblib", bundle["imputer"])
        save_joblib(self.run_dir / "onehot.joblib", bundle["onehot"])
        save_joblib(self.run_dir / "feature_selector.joblib", bundle["selector"])
        if bundle["semantic_pca"] is not None:
            save_joblib(self.run_dir / "semantic_pca.joblib", bundle["semantic_pca"])
            np.save(self.run_dir / "semantic_embeddings.npy", bundle["semantic_embeddings"], allow_pickle=False)
            atomic_json(self.run_dir / "semantic_embedding_ids.json", {"question_ids": [r.question_id for r in fit_records]})
            atomic_json(self.run_dir / "semantic_encoder.json", {"name": self.cfg["semantic"]["encoder"], "revision": getattr(bundle["encoder"], "revision", None), "dim": int(bundle["semantic_embeddings"].shape[1]), "max_seq_length": 256})

        production_columns = bundle["final_columns"] if production_mode == "final" else bundle["engineered_columns"]
        production_schema = make_schema(production_columns, ["float64"] * len(production_columns), profile=profile)
        production_schema["engineered_columns"] = bundle["engineered_columns"]
        production_schema["full_columns"] = bundle["full_columns"]
        production_schema["active_feature_ids"] = bundle["active_feature_ids"]
        production_schema["production_matrix"] = production_mode
        atomic_json(self.run_dir / "feature_schema.json", production_schema)
        full_schema = make_schema(bundle["full_columns"], ["float64"] * len(bundle["full_columns"]), profile=profile)
        full_schema["engineered_columns"] = bundle["engineered_columns"]
        full_schema["active_feature_ids"] = bundle["active_feature_ids"]
        atomic_json(self.run_dir / "full_feature_schema.json", full_schema)
        atomic_json(self.run_dir / "production_feature_mode.json", {"semantic_embedding": bool(bundle["semantic_pca"] is not None and production_mode == "final"), "use_final_selector": production_mode == "final"})
        atomic_json(
            self.run_dir / "preprocessing_manifest.json",
            {
                "numeric_columns": bundle["numeric_columns"],
                "engineered_columns": bundle["engineered_columns"],
                "full_columns": bundle["full_columns"],
                "final_columns": bundle["final_columns"],
                "active_feature_ids": bundle["active_feature_ids"],
                "profile": profile,
            },
        )
        save_joblib(self.run_dir / "model.joblib", final_model)

        if production_mode == "final":
            X_prod_fit = bundle["X_final"]
            X_prod_test = None
        else:
            X_prod_fit = bundle["X_eng_raw"]
            X_prod_test = None
        # Mark test access before touching the held-out rows. If execution is interrupted after
        # this point, re-evaluation requires the explicit override flag rather than silently
        # consuming the test set a second time.
        marker = self.run_dir / "test_evaluated.marker"
        marker.write_text("started\n", encoding="utf-8")
        # Transforming the test set occurs only now, after selection/final fit.
        test_eng, test_final, test_combined, test_diagnostics = self._transform_with_bundle(test_records, bundle)
        X_prod_test = test_final if production_mode == "final" else test_eng
        y_test = np.asarray([LABEL_TO_INT[r.label] for r in test_records], dtype=int)
        if production_mode == "final":
            val_X = bundle["X_final"][val_idx] if len(val_idx) else bundle["X_final"][train_idx]
        else:
            val_X = bundle["X_eng_raw"][val_idx] if len(val_idx) else bundle["X_eng_raw"][train_idx]
        y_val = y_fit[val_idx] if len(val_idx) else y_fit[train_idx]

        calibration_model = None
        calibrated = False
        if self.cfg["training"].get("calibrate", False):
            calibration_model = calibrate_model(final_model, val_X, y_val)
            save_joblib(self.run_dir / "calibrator.joblib", calibration_model)
            predictor_model = calibration_model
            calibrated = True
        else:
            predictor_model = final_model

        train_fit_idx = np.arange(len(y_fit))
        train_metrics = evaluate(y_fit, predictor_model.predict(X_prod_fit), predictor_model.predict_proba(X_prod_fit), calibrated)
        val_metrics_observed = evaluate(y_val, predictor_model.predict(val_X), predictor_model.predict_proba(val_X), calibrated)
        val_metrics = selection_cv_metrics if selection_cv_metrics is not None else val_metrics_observed
        test_metrics = evaluate(y_test, predictor_model.predict(X_prod_test), predictor_model.predict_proba(X_prod_test), calibrated)
        diagnostics = list(bundle.get("feature_diagnostics", [])) + test_diagnostics
        with (self.run_dir / "feature_diagnostics.jsonl").open("w", encoding="utf-8") as handle:
            for item in diagnostics:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        candidate_rows = [{"run_id": item["run_id"], "train": item["train"], "validation": item["validation"], "selected": item["run_id"] == selected["run_id"]} for item in candidates]
        metrics_payload = {
            "candidate_runs": candidate_rows,
            "selected_run_id": selected["run_id"],
            "production": {"train": train_metrics, "validation": val_metrics, "test": test_metrics},
            "selection_rule": "macro_f1 desc, ordinal_mae asc, balanced_accuracy desc, run_id asc",
            "selection_method": "repeated_stratified_group_cv" if selection_cv_metrics is not None else "single_group_validation_split",
            "observed_first_fold_validation": val_metrics_observed,
            "semantic_pca": None if bundle["semantic_pca"] is None else {
                "explained_variance_ratio": np.asarray(bundle["semantic_pca"].explained_variance_ratio_).tolist(),
                "cumulative_explained_variance": float(np.asarray(bundle["semantic_pca"].explained_variance_ratio_).sum()),
                "warning_below_min": bool(np.asarray(bundle["semantic_pca"].explained_variance_ratio_).sum() < float(self.cfg["semantic"]["min_explained_variance"])),
            },
        }
        atomic_json(self.run_dir / "metrics.json", metrics_payload)
        self._interpret(final_model, val_X, y_val, production_columns)

        saved_parquet = self._save_matrix(self.root / "data/features/full", [r.question_id for r in fit_records], [r.split for r in fit_records], bundle["X_full"], bundle["full_columns"])
        self._save_matrix(self.root / "data/features/final", [r.question_id for r in fit_records], [r.split for r in fit_records], bundle["X_final"] if production_mode == "final" else bundle["X_eng_raw"], production_columns)
        # Test rows are persisted only after the one final evaluation.
        if saved_parquet:
            self._save_matrix(self.root / "data/features/full_test", [r.question_id for r in test_records], [r.split for r in test_records], test_combined, bundle["full_columns"])
            self._save_matrix(self.root / "data/features/final_test", [r.question_id for r in test_records], [r.split for r in test_records], X_prod_test, production_columns)

        config_fingerprint, config_files = self._config_fingerprint()
        git_commit = None
        try:
            git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.root, check=True, capture_output=True, text=True).stdout.strip()
        except Exception:
            pass
        manifest = {
            "dataset_fingerprint": stable_json_hash({"raw_hashes": sorted(record.raw_hash for record in records), "record_count": len(records)}),
            "config_fingerprint": config_fingerprint,
            "config_files": config_files,
            "record_count": len(records),
            "split": {name: [r.question_id for r in records if r.split == name] for name in ("train", "validation", "test")},
            "seeds": self.cfg["seed"],
            "config": self.cfg,
            "dependency_profile": profile,
            "dependency_versions": self._package_versions(),
            "feature_schema_version": production_schema["version"],
            "schema_hash": production_schema["schema_hash"],
            "full_feature_schema_hash": full_schema["schema_hash"],
            "production_feature_mode": {"semantic_embedding": bool(bundle["semantic_pca"] is not None and production_mode == "final"), "use_final_selector": production_mode == "final"},
            "selected_hyperparameters": {"production": selected.get("hparams"), "best_xgb_for_ablations": best_xgb.get("hparams") if best_xgb else None},
            "class_weight_decision": {"training_ratio": class_ratio, "applied": weights_applied},
            "calibrated": calibrated,
            "test_evaluated": True,
            "git_commit": git_commit,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "cache_root": str(self.root / "data/features/cache"),
            "parquet_persisted": bool(saved_parquet),
        }
        atomic_json(self.run_dir / "run_manifest.json", manifest)
        atomic_json(self.run_dir / "ablation_config.json", {"selected_xgb_hyperparameters": best_xgb.get("hparams") if best_xgb else None, "test_ids_are_forbidden": True})
        (self.run_dir / "report.md").write_text(
            "# QDP Run Report\n\n"
            f"- Dependency profile: `{profile}`\n"
            f"- Records: **{len(records)}**\n"
            f"- Selected model: `{selected['run_id']}`\n"
            f"- Validation Macro F1: **{val_metrics['macro_f1']:.6f}**\n"
            f"- Test Macro F1: **{test_metrics['macro_f1']:.6f}**\n"
            f"- Test Ordinal MAE: **{test_metrics['ordinal_mae']:.6f}**\n"
            f"- Test Balanced Accuracy: **{test_metrics['balanced_accuracy']:.6f}**\n"
            "\nInterpretability describes model reliance and does not establish causality.\n",
            encoding="utf-8",
        )
        (self.run_dir / "test_evaluated.marker").write_text("true", encoding="utf-8")
        checksums = checksum_dir(self.run_dir, exclude={"artifact_checksums.json", "run_manifest.json"})
        atomic_json(self.run_dir / "artifact_checksums.json", checksums)
        manifest["artifact_checksums"] = checksums
        atomic_json(self.run_dir / "run_manifest.json", manifest)
        return {"run_dir": str(self.run_dir), "dependency_profile": profile, "selected_run_id": selected["run_id"], "test": test_metrics}

    def _select_cv_on_pool(self, pool_records, profile, folds):
        """Select models with leakage-safe preprocessing refit inside every CV fold."""
        fold_results = []
        id_to_record = {r.question_id: r for r in pool_records}
        for fold in folds:
            train_set = set(fold["train_ids"])
            val_set = set(fold["validation_ids"])
            fold_records = []
            for record in pool_records:
                if record.question_id in train_set:
                    fold_records.append(with_split(record, "train", record.family_id))
                elif record.question_id in val_set:
                    fold_records.append(with_split(record, "validation", record.family_id))
            if not fold_records or not train_set or not val_set:
                raise PipelineError("invalid_cv_fold")
            fold_bundle = self._fit_feature_bundle(fold_records, profile)
            y = np.asarray([LABEL_TO_INT[r.label] for r in fold_records], dtype=int)
            tr = np.asarray([i for i, r in enumerate(fold_records) if r.split == "train"], dtype=int)
            va = np.asarray([i for i, r in enumerate(fold_records) if r.split == "validation"], dtype=int)
            train_weights, ratio, applied = self._sample_weights(y[tr])
            fold_results.append((fold_bundle, y, tr, va, train_weights, ratio, applied))

        candidate_defs = [("baseline_logistic", None, None), ("baseline_rf", None, None)]
        if self.cfg["training"]["classifier"] == "xgboost":
            for depth, lr, subsample in (grid() if self.cfg["training"]["hyperparameter_search"] else [(6, 0.05, 0.8)]):
                candidate_defs.append((f"xgb_d{depth}_l{lr}_s{subsample}", depth, (lr, subsample)))

        candidates = []
        for run_id, depth, rest in candidate_defs:
            fold_train_metrics = []
            fold_val_metrics = []
            best_rounds = []
            for fold_bundle, y, tr, va, train_weights, ratio, applied in fold_results:
                if run_id == "baseline_logistic":
                    model = make_logistic(self.cfg["seed"]["model"])
                    model, tm, vm = self._fit_baseline("baseline_logistic", model, fold_bundle["X_eng_raw"], y, tr, va, train_weights)
                elif run_id == "baseline_rf":
                    model = make_random_forest(self.cfg["seed"]["model"])
                    model, tm, vm = self._fit_baseline("baseline_rf", model, fold_bundle["X_eng_raw"], y, tr, va, train_weights)
                else:
                    lr, subsample = rest
                    val_weight = None
                    if applied:
                        counts = np.bincount(y[tr], minlength=3)
                        mapping = {cls: len(tr) / (3.0 * int(count)) for cls, count in enumerate(counts) if count}
                        val_weight = np.asarray([mapping[int(label)] for label in y[va]], dtype=float)
                    model, tm, vm, best_n = self._fit_xgb_candidate(depth, lr, subsample, fold_bundle["X_final"], y, tr, va, train_weights, val_weight)
                    best_rounds.append(best_n)
                fold_train_metrics.append(tm)
                fold_val_metrics.append(vm)

            avg_train = {key: float(np.mean([item[key] for item in fold_train_metrics])) for key in fold_train_metrics[0] if isinstance(fold_train_metrics[0][key], (int, float))}
            avg_val = {key: float(np.mean([item[key] for item in fold_val_metrics])) for key in fold_val_metrics[0] if isinstance(fold_val_metrics[0][key], (int, float))}
            if run_id.startswith("xgb_"):
                lr, subsample = rest
                n_estimators = int(max(1, round(np.mean(best_rounds))))
                hparams = {"max_depth": depth, "learning_rate": lr, "subsample": subsample, "n_estimators": n_estimators, "early_stopping_rounds": 50}
                model = make_xgb(depth, lr, subsample, self.cfg["seed"]["model"], n_estimators=n_estimators, early_stopping_rounds=None)
                matrix = "final"
            else:
                hparams = None
                model = make_logistic(self.cfg["seed"]["model"]) if run_id == "baseline_logistic" else make_random_forest(self.cfg["seed"]["model"])
                matrix = "engineered"
            candidates.append({"key": selection_key(avg_val, run_id), "run_id": run_id, "model": model, "train": avg_train, "validation": avg_val, "matrix": matrix, "hparams": hparams})
        return candidates, fold_results

