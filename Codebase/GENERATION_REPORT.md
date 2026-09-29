# QDP Final Codebase Verification Report

## Status

The repository has been corrected against the finalized `QDP_Final_TDD2.md` contract. The correction pass addressed the previously identified preprocessing, leakage, splitting, feature-semantics, artifact-integrity, calibration, inference, and test-set isolation issues.

## Final verification performed

- Python compilation: **passed** for `src/`, `scripts/`, and `tests/`.
- Full local regression suite: **29 passed**.
- Standard 70/15/15 group split smoke check: **42/9/9**, with all classes represented in train/validation/test.
- Repeated group-CV validation-fold class coverage: **passed**.
- P4 end-to-end smoke run on a synthetic 60-record, three-class corpus: **completed** through validation, deterministic model selection, final test evaluation, artifact creation, and inference.
- Calibration-enabled P4 smoke run: **completed**, including calibrated inference output and calibration metrics.
- Test-set reuse protection: **verified**; a second training/test attempt against the same run directory is blocked unless the explicit override flag is supplied.
- Audit diagnostics: **persisted** to `feature_diagnostics.jsonl` and kept outside the model matrix.
- Final artifact checksum verification: **passed** during evaluation/inference smoke runs.

## Important verification limitation

The current execution environment does not contain the complete P0 research stack (`textstat`, `sentence-transformers`, `pyarrow`, and the local `en_core_web_sm` spaCy model). The corrected repository therefore has not been represented as having completed a full P0 run on the 100,000-question research corpus. The P4 smoke path was explicitly used to verify the core orchestration and contracts under the available environment.

## Research-data limitation

The checked-in `data/raw/questions.jsonl` remains a tiny development fixture and intentionally fails the production `corpus.min_records` gate. It must be replaced by the authentic labelled corpus built under the project's data specification before research training.

## Schema/version note

The final hardening pass increments the feature/extractor schema to `fs_v3` / `extractor_v3`. Any artifacts generated under earlier schema versions must be regenerated before use.
