# Engineering Decisions

## Decision: Final TDD is the implementation authority

The repository implements `QDP_Final_TDD2.md` as the sole authoritative TDD. The older `QDP_Final_TDD_Ready_For_Codegen.md` reference was removed to prevent contradictory specifications from entering the implementation.

## Decision: Runtime environment is detected before training

The trainer performs dependency and local-model preflight and records the resolved profile in `environment_preflight.json` and `run_manifest.json`. P0 is used only when every dependency/model asset required by P0 is available. Unsupported combinations are blocked rather than silently substituting libraries or dropping features.

## Decision: Small datasets use leakage-safe repeated group CV

When the corpus is below the TDD small-dataset threshold, a deterministic held-out test set is reserved first. The remaining pool is evaluated with repeated stratified group CV, and feature preprocessing is refit inside each CV fold. The external test set is not used during model selection.

## Decision: Global near-duplicate families never cross splits

Family identifiers are global rather than source-prefixed. Provenance may be used for grouping/audit, but it cannot cause the same near-duplicate family to appear in different splits.

## Decision: Feature schemas distinguish logical features from final columns

The repository stores both the full transformed feature schema and the production model schema. Variance and correlation selection are fitted on training data only. Schema hashes are verified during inference.

## Decision: Semantic embedding cache is role-aware

Embedding cache keys include question id, text role, text hash, encoder identity/revision, and truncation length so question, stem, option, and sentence embeddings cannot collide.

## Decision: Class weights are derived only from the training partition

When the training class-count ratio exceeds 1.5, weights use `N_train / (3 * class_count)`. The same training-derived policy is used when the final model is fit on train+validation and is never recomputed from validation/test data.

## Decision: Test data is evaluated only after selection

Candidate models and ablations use train/validation or repeated group CV only. The held-out test set is transformed and evaluated only for the final selected production model.

## Decision: Ablations use frozen selected main-model hyperparameters

Ablation runs do not perform another hyperparameter search. They use the selected main XGBoost hyperparameters so feature-family comparisons are attributable to the feature subset rather than re-tuning.

## Decision: Artifact integrity is refreshed after ablation generation

Ablation outputs are added to the run artifact checksum manifest after they are created so later evaluation/integrity checks remain consistent.

## Decision: Starter lexicons remain explicit configuration

The codebase ships deterministic lexicon/config files required to execute the feature extractors. These are configuration inputs, not hidden model logic. Their final research coverage should be reviewed against the authentic target corpus before scientific conclusions are drawn.

## Current verification environment

The current execution environment does not contain `textstat`, `sentence-transformers`, `pyarrow`, or the `en_core_web_sm` spaCy model, so a full P0 research run cannot be claimed here. A P4 smoke training/inference run was executed successfully using a small synthetic development fixture, while the checked-in authentic-data fixture remains intentionally below the production record-count gate.

## Decision: Final correctness hardening (feature-schema v2)

**Context**

The codebase audit identified several concrete implementation mismatches: working-text normalization was not consistently applied, degraded sentence segmentation used line counts instead of the specified sentence rule, F17 counted non-sequential enumerators, F31 mis-parsed `10^6`, fenced code markers could reach `ast.parse`, task-family ablations could still retain F14/F15, and semantic loading contained an unsafe network fallback.

**Decision**

These behaviors were corrected while preserving the TDD2 feature formulas and execution contract. The feature/extractor schema version was incremented to `fs_v2` / `extractor_v2` because the preprocessing semantics changed.

**Reason**

The TDD requires deterministic, leakage-safe, reproducible feature semantics and an offline core execution path.

**Impact**

Artifacts generated with the older schema are intentionally incompatible with the corrected code and must be regenerated.

## Decision: Final train-only imputer fitting

The numeric imputer is fitted strictly on `train_idx` via row selection before transforming any validation or test rows. The previously stored fit-row metadata was not sufficient by itself; the implementation now enforces the data boundary in the actual `SimpleImputer.fit` call.

## Decision: Test access marker is written before first test read

The held-out test marker is written immediately before the first test transformation. This makes accidental reruns after a partial test-stage failure detectable and requires the explicit override flag for any re-evaluation.

## Decision: Current sklearn calibration API

Calibration uses `FrozenEstimator` when available and falls back to `cv="prefit"` only for older sklearn versions. The base estimator is never retrained during calibration.


## Decision: Feature schema version increment after final semantic fixes

The feature/extractor schema version is now `fs_v3` / `extractor_v3`. The increment records the finalized F30 semantics (algorithm lexicon is evaluated over `nl_view + code_spans`) and the hardened degraded-profile group-CV behavior. Older `fs_v2` artifacts must be regenerated.

## Decision: F30 searches both prose and code

TDD2 defines F30 over `nl_view + code_spans`. The implementation now concatenates both views and treats whitespace, underscore, and hyphen as deterministic phrase separators so configured techniques such as `binary search` are detected in prose and conventional code identifiers.

## Decision: Group-CV validation folds require all three labels

Repeated stratified group CV now seeds validation folds with groups covering `Easy`, `Moderate`, and `Hard` whenever the available group capacity permits it. A fold that cannot cover all three classes fails rather than producing undefined multiclass metrics.

## Decision: Persist audit-only feature diagnostics

The final run writes `feature_diagnostics.jsonl` containing `semantic_truncated`, `code_parse_exact`, code-detection tiers, and MCQ-detection tier. These fields are audit-only and are never passed to the model matrix.
