# Question Difficulty Prediction (QDP)

This repository implements the architecture and contracts defined by `QDP_Final_TDD2.md`.

## What is implemented

- Leakage-safe ingestion, exact deduplication, and global MinHash/LSH family grouping.
- 70/15/15 group-safe splitting for large corpora and held-out-test + repeated group CV for small corpora.
- The complete 35-feature logical schema plus model missingness indicators.
- Deterministic preprocessing, MCQ/code detection, programming complexity features, train-only TF-IDF/rare-word/PCA/imputer/one-hot/correlation selection, and the frozen semantic branch when P0 assets are available.
- Baseline Logistic Regression and Random Forest models plus the specified XGBoost candidate grid and deterministic four-level selection rule.
- Optional validation-only post-hoc calibration.
- Single-use held-out test evaluation with an explicit override required for re-evaluation.
- Versioned feature schemas, artifact checksums, run manifests, feature diagnostics, deterministic inference, and ablation execution.

## Important prerequisites

The full research path (P0) requires the dependencies and local model assets specified by the TDD. The repository performs environment preflight and does not silently substitute dependencies or fabricate missing corpus or lexicon inputs.

If the environment supports only a named degradation profile, the run records that profile explicitly. P4 is a smoke-test floor and is not sufficient to answer the complete research question.

## Repository

`src/qdp/` contains the implementation. `config/` contains versioned feature semantics and defaults. `artifacts/<run_id>/` contains run outputs. `scripts/` provides training, evaluation, prediction, and ablation entry points. `tests/` contains regression tests aligned with the TDD.

The current corrected feature/extractor schema is `fs_v3` / `extractor_v3`. Artifacts produced with older schema versions must be regenerated.

## Run tests

```bash
python -m pytest -q
```

## Run training

```bash
python scripts/train.py --root . --run-dir artifacts/latest
```

The checked-in `data/raw/questions.jsonl` is intentionally a five-record development fixture and will fail the production `corpus.min_records` gate. Replace it with the authentic labelled corpus built according to the QDP data specification before a research run.

## Run inference

After a successful training run:

```bash
python scripts/predict.py --root . --run-dir artifacts/latest --question "Your question here"
```

Inference loads the saved artifacts, verifies their checksums and feature schema, and never refits a transformer.

## Audit outputs

Each completed run contains, among other artifacts:

- `feature_schema.json`
- `full_feature_schema.json`
- fitted preprocessing/model artifacts
- `feature_diagnostics.jsonl` for audit-only diagnostics such as semantic truncation and parser/code-detection status
- `metrics.json`
- `run_manifest.json`
- `artifact_checksums.json`
- `test_evaluated.marker`

The diagnostics file is never included in the model feature matrix.
