from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qdp.artifacts.save_load import load_json
from qdp.artifacts.checksums import checksum_dir, verify_checksums
from qdp.data.schema import LABEL_TO_INT
from qdp.evaluation.metrics import evaluate
from qdp.models.xgb import make_xgb
from qdp.features.registry import FAMILIES


def load_labels_for_pool(project_root, manifest):
    allowed = set(manifest["split"]["train"]) | set(manifest["split"]["validation"])
    labels = {}
    path = project_root / "data/processed/canonical.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["question_id"] in allowed:
                labels[row["question_id"]] = LABEL_TO_INT[row["label"]]
    if allowed - set(labels):
        raise RuntimeError("missing_train_validation_labels")
    return labels


def plan(active_families):
    ladder = [
        ("A0", []),
        ("A1", ["lexical", "readability"]),
        ("A2", ["lexical", "readability", "syntax"]),
        ("A3", ["lexical", "readability", "syntax", "task"]),
        ("A4", ["lexical", "readability", "syntax", "task", "mcq"]),
        ("A5", ["lexical", "readability", "syntax", "task", "mcq", "tfidf"]),
        ("A6", ["lexical", "readability", "syntax", "task", "mcq", "tfidf", "semantic_scalar"]),
        ("A7", ["lexical", "readability", "syntax", "task", "mcq", "tfidf", "semantic_scalar", "programming"]),
        ("A8", ["lexical", "readability", "syntax", "task", "mcq", "tfidf", "semantic_scalar", "programming", "semantic_embedding"]),
        ("A8b-32", ["lexical", "readability", "syntax", "task", "mcq", "tfidf", "semantic_scalar", "programming", "semantic_embedding"]),
        ("A8b-48", ["lexical", "readability", "syntax", "task", "mcq", "tfidf", "semantic_scalar", "programming", "semantic_embedding"]),
        ("A8b-64", ["lexical", "readability", "syntax", "task", "mcq", "tfidf", "semantic_scalar", "programming", "semantic_embedding"]),
    ]
    standalone = [(f"S_{family}", [family]) for family in active_families]
    return ladder + standalone


def feature_columns(columns, families, include_embedding=True):
    wanted = set()
    for family in families:
        wanted.update(FAMILIES.get(family, []))
    result = []
    for column in columns:
        if column.startswith("sem_pca_"):
            if include_embedding and "semantic_embedding" in families:
                result.append(column)
        elif column in wanted:
            result.append(column)
        elif column.startswith("question_type=") or column.startswith("instruction_type="):
            if "task" in families:
                result.append(column)
        elif column in {"mcq_detected"} and "mcq" in families:
            result.append(column)
        elif column in {"code_detected", "input_size_detected"} and "programming" in families:
            result.append(column)
        elif column == "parse_ok" and "syntax" in families:
            result.append(column)
        elif column == "multi_segment" and "semantic_scalar" in families:
            result.append(column)
    return result


def class_weights(y):
    counts = np.bincount(y, minlength=3)
    ratio = counts.max() / counts.min()
    if ratio <= 1.5:
        return None
    mapping = {cls: len(y) / (3.0 * int(count)) for cls, count in enumerate(counts)}
    return np.asarray([mapping[int(cls)] for cls in y], dtype=float)


def fit_and_evaluate(X, y, train_idx, val_idx, hp, seed):
    model = make_xgb(
        hp["max_depth"], hp["learning_rate"], hp["subsample"], seed,
        n_estimators=hp["n_estimators"], early_stopping_rounds=None,
    )
    weights = class_weights(y[train_idx])
    if weights is not None:
        model.fit(X[train_idx], y[train_idx], sample_weight=weights, verbose=False)
    else:
        model.fit(X[train_idx], y[train_idx], verbose=False)
    return evaluate(y[val_idx], model.predict(X[val_idx]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="artifacts/latest")
    args = parser.parse_args()
    run_dir = (ROOT / args.run_dir).resolve()
    manifest = load_json(run_dir / "run_manifest.json")
    verify_checksums(run_dir, manifest["artifact_checksums"])
    config = load_json(run_dir / "resolved_config.json")
    full_schema = load_json(run_dir / "full_feature_schema.json")
    frame_path = ROOT / "data/features/full"
    train_path = frame_path / "train.parquet"
    val_path = frame_path / "validation.parquet"
    if not train_path.exists() or not val_path.exists():
        raise RuntimeError("ablation_requires_train_and_validation_feature_partitions")
    train_frame = pd.read_parquet(train_path)
    val_frame = pd.read_parquet(val_path)
    frame = pd.concat([train_frame, val_frame], ignore_index=True)
    labels = load_labels_for_pool(ROOT, manifest)
    y = np.asarray([labels[qid] for qid in frame["question_id"]], dtype=int)
    train_idx = np.arange(len(train_frame))
    val_idx = np.arange(len(train_frame), len(frame))
    columns = list(full_schema["columns"])
    X = frame[columns].to_numpy(dtype=float)
    hp = manifest["selected_hyperparameters"].get("best_xgb_for_ablations")
    if not hp:
        raise RuntimeError("missing_best_xgb_hyperparameters")

    feature_family_names = ["lexical", "readability", "syntax", "task", "mcq", "tfidf", "semantic_scalar", "programming", "semantic_embedding"]
    active_families = [family for family in feature_family_names if family == "semantic_embedding" or any(col in FAMILIES.get(family, []) or col.startswith(("question_type=", "instruction_type=")) for col in columns)]
    results = []
    majority = int(np.bincount(y[train_idx], minlength=3).argmax())
    baseline = evaluate(y[val_idx], np.full(len(val_idx), majority, dtype=int))
    results.append({"run_id": "A0", "validation": baseline, "families": []})
    for run_id, families in plan(active_families)[1:]:
        if run_id.startswith("A8b-"):
            if not any(column.startswith("sem_pca_") for column in columns):
                results.append({"run_id": run_id, "status": "unavailable", "reason": "semantic_embedding_not_available"})
                continue
            requested_dim = int(run_id.split("-")[1])
            # The stored 48-D PCA block cannot be safely re-expanded to 64. Re-run PCA from
            # persisted raw embeddings when available; otherwise report unavailable.
            emb_path = run_dir / "semantic_embeddings.npy"
            if not emb_path.exists():
                results.append({"run_id": run_id, "status": "unavailable", "reason": "raw_embeddings_not_persisted"})
                continue
            from qdp.semantic.pca import SemanticPCA
            embeddings = np.load(emb_path, allow_pickle=False)
            embedding_ids = load_json(run_dir / "semantic_embedding_ids.json")["question_ids"]
            embedding_index = {qid: index for index, qid in enumerate(embedding_ids)}
            try:
                row_indices = [embedding_index[qid] for qid in frame["question_id"].tolist()]
            except KeyError as exc:
                raise RuntimeError(f"missing_semantic_embedding_for:{exc}") from exc
            aligned_embeddings = embeddings[row_indices]
            train_embedding_indices = [embedding_index[qid] for qid in train_frame["question_id"].tolist()]
            pca = SemanticPCA(requested_dim, int(config["seed"]["pca"]))
            pca.fit(embeddings[train_embedding_indices])
            engineered_cols = [column for column in columns if not column.startswith("sem_pca_")]
            Xa = np.hstack([frame[engineered_cols].to_numpy(dtype=float), pca.transform(aligned_embeddings)])
            metrics = fit_and_evaluate(Xa, y, train_idx, val_idx, hp, int(config["seed"]["model"]))
            results.append({"run_id": run_id, "validation": metrics, "families": families, "pca_components": requested_dim})
            continue
        if "semantic_embedding" in families and not any(column.startswith("sem_pca_") for column in columns):
            results.append({"run_id": run_id, "status": "unavailable", "reason": "semantic_embedding_not_available"})
            continue
        cols = feature_columns(columns, families, include_embedding=True)
        if not cols:
            results.append({"run_id": run_id, "status": "unavailable", "reason": "no_columns_after_family_filter"})
            continue
        Xa = frame[cols].to_numpy(dtype=float)
        metrics = fit_and_evaluate(Xa, y, train_idx, val_idx, hp, int(config["seed"]["model"]))
        results.append({"run_id": run_id, "validation": metrics, "families": families, "features": cols})

    baseline_f1 = baseline["macro_f1"]
    cumulative_prev = None
    for row in results:
        metrics = row.get("validation")
        if not metrics:
            continue
        row["macro_f1_delta_vs_A0"] = float(metrics["macro_f1"] - baseline_f1)
        if row["run_id"].startswith("A") and row["run_id"] in {f"A{i}" for i in range(1, 9)}:
            row["macro_f1_delta_vs_previous_cumulative"] = None if cumulative_prev is None else float(metrics["macro_f1"] - cumulative_prev)
            cumulative_prev = metrics["macro_f1"]
    (run_dir / "ablation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = ["# Ablation Results", "", "| Run | Macro F1 | Ordinal MAE | Balanced Accuracy | Δ vs A0 |", "|---|---:|---:|---:|---:|"]
    for row in results:
        metrics = row.get("validation")
        if metrics:
            lines.append(f"| {row['run_id']} | {metrics['macro_f1']:.6f} | {metrics['ordinal_mae']:.6f} | {metrics['balanced_accuracy']:.6f} | {row.get('macro_f1_delta_vs_A0', float('nan')):.6f} |")
        else:
            lines.append(f"| {row['run_id']} | {row.get('status','unavailable')} | | | |")
    ablation_markdown = "\n".join(lines) + "\n"
    (run_dir / "ablation_results.md").write_text(ablation_markdown, encoding="utf-8")
    report_path = run_dir / "report.md"
    existing_report = report_path.read_text(encoding="utf-8") if report_path.exists() else "# QDP Run Report\n"
    if "## Ablation Results" not in existing_report:
        report_path.write_text(existing_report.rstrip() + "\n\n## Ablation Results\n\n" + ablation_markdown, encoding="utf-8")
    # Refresh the run integrity manifest after adding ablation artifacts.
    checksums = checksum_dir(run_dir, exclude={"artifact_checksums.json", "run_manifest.json"})
    (run_dir / "artifact_checksums.json").write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["artifact_checksums"] = checksums
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
