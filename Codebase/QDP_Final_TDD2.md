# Question Difficulty Prediction (QDP) — Final Technical Design Document

**Document type:** Authoritative, implementation-ready Technical Design Document
**Status:** Final for code-generation input
**Task:** Three-class question-difficulty classification — `Easy | Moderate | Hard`
**Model input contract:** question text only. No learner-response data, no external metadata, no answer key.
**Derived from:** `QDP_100K_Data_Spec.txt`, `Question_Difficulty_Prediction_End_to_End_TDD.docx` (draft), `Question_Difficulty_Prediction_Chat_Context.docx`

This document supersedes the draft TDD where the two disagree. Every such divergence is recorded in §24 (Decision Log) with reason and impact. Conventions used throughout:

- **SPEC** — an explicit requirement taken from the supplied context. Not negotiable by the implementer.
- **DECISION** — an engineering resolution made here to remove ambiguity. Binding, but traceable as a design choice rather than an upstream requirement.
- **UNRESOLVED** — a genuine external dependency that this document cannot settle. Listed in §25. Must be confirmed before or during implementation; must not be silently guessed.
- **HYPOTHESIS** — an expectation that is *not* established by the supplied research and must not be reported as a finding.

---

## Table of Contents

1. Purpose, Scope, and Non-Goals
2. Terminology
3. Environment and Dependency Contract
4. Assumptions
5. Upstream Corpus Contract (the 100,000-question dataset)
6. Model-Facing Data Contract and Ingestion
7. Validation, Deduplication, and Splitting
8. Preprocessing and Text Views
9. Feature Specification (35 logical features)
10. Semantic Representation Branch
11. Feature Transformation and Final Matrix
12. Model Design and Training
13. Evaluation
14. Experiments and Ablations
15. Interpretability
16. Inference
17. Artifacts and Reproducibility
18. Configuration
19. Error Handling and Failure Behaviour
20. Performance and Scalability
21. Security and Data Integrity
22. Repository Layout
23. Testing Specification
24. Decision Log
25. Unresolved Dependencies

---

## 1. Purpose, Scope, and Non-Goals

### 1.1 System objective

Predict whether a question is `Easy`, `Moderate`, or `Hard` using only information intrinsically present in the question text. The system must:

1. Ingest and validate a minimal `{question, label}` corpus.
2. Derive reproducible numeric features from question text alone.
3. Train a multiclass classifier with no data leakage.
4. Evaluate with both nominal and ordinal error measures.
5. Quantify which feature families carry difficulty information (a first-class research goal, not an afterthought).
6. Expose a deterministic inference path reusing the exact fitted artifacts from training.

### 1.2 Research question

*How much information about question difficulty is contained in the intrinsic linguistic, structural, semantic, and programming properties of the question itself?* (SPEC — chat context §4.)

The intended scientific output is **both** a predictor **and** evidence about feature-family contribution. §14 therefore specifies the ablation programme as a required deliverable, not an optional extra.

### 1.3 In scope

- Text-only three-class difficulty prediction with ordered labels.
- 35 logical engineered features (§9) plus a reduced sentence-embedding branch (§10).
- MCQ structural features when options are detectable **in the question text**, using **answer-key-independent** definitions only.
- Programming features when code or constraints are detectable in the question text. No separate code column.
- Leakage-safe splitting, training, validation, single-use test evaluation.
- Ablations, feature importance, artifact versioning, deterministic inference.

### 1.4 Explicitly out of scope (non-goals)

| Excluded | Basis |
|---|---|
| Learner-response signals: success rate, attempts, response time, submission history, student performance | SPEC — data spec §13, chat §3, draft §1 |
| External metadata as model input: subject, exam level, author, semester, year, source platform | SPEC — data spec §13, chat §3 |
| Any use of a correct answer / answer key / post-administration statistics as a feature | SPEC — data spec §8 |
| Transformer fine-tuning as the default production path | SPEC — draft §1; retained only as a documented future extension; not implemented in the current codebase |
| Continuous difficulty score as a product output | SPEC — draft §2. The `{0,1,2}` mapping is internal to ordinal metrics only |
| Distributed compute, microservices, model-serving infrastructure, databases | Prompt §5 — not required by the project |
| Multimodal / image questions, knowledge-graph features, response-time prediction | Future extensions (draft §17), not implemented now |

### 1.5 Design principles

1. **Correctness before complexity.** A leakage-safe deterministic pipeline is established before any model sophistication.
2. **Compactness.** ~35 engineered scalars plus a reduced embedding, not hundreds of correlated columns (SPEC — chat §10, draft §1).
3. **Determinism over "smart" behaviour.** Every feature has one exact formula. No implementation-time choice, no LLM-in-the-loop feature derivation (SPEC — chat §22, §24).
4. **Graceful degradation.** Absent MCQ structure or code must never crash extraction; it produces an explicit missing value plus an indicator.
5. **Ablation-ready.** Every feature family is independently switchable.
6. **Reproducibility.** Every fitted transformer is versioned with the feature schema and configuration that produced it.

---

## 2. Terminology

| Term | Definition in this document |
|---|---|
| **Logical feature** | One conceptual feature as specified in §9 (F01–F35). Exactly 35 exist. |
| **Final model column** | One column of the numeric matrix fed to the learner. Count ≠ 35 (see §11). |
| **Model-facing data** | `question` + `label` only. The sole input to the feature pipeline. |
| **Audit data** | Provenance, licence, rejection, duplicate, statistics records. Never reaches the feature pipeline (§5.4). |
| **Fitted artifact** | Any object whose parameters are estimated from data (TF-IDF vocabulary, IDF, rare-word table, TF-IDF threshold, imputer, encoder, PCA, correlation mask, model). Fitted on **train split only**. |
| **Frozen artifact** | Pretrained object used without estimation from project data (sentence encoder weights, spaCy model). Not a leakage vector; see §7.5. |
| **Family** | A group of exact- or near-duplicate questions that must not be split across train/validation/test (§7.3). |
| **NL view** | Natural-language projection of question text used for linguistic features (§8.2). |
| **Missingness indicator** | Binary column recording that a feature was structurally unavailable, as distinct from genuinely zero. |

**Label order (SPEC, draft §2):** `Easy → 0`, `Moderate → 1`, `Hard → 2`. Used for ordinal metrics only; never emitted as a score.

---

## 3. Environment and Dependency Contract

### 3.1 Constraint

The project owner has directed that **no library outside the existing environment may be installed or used**. The environment inventory was not verified while authoring this document, so availability must be checked at implementation start. This is a **pre-implementation environment check**, not permission to silently substitute libraries.

Before implementation begins, the code generator must inspect the actual environment, record the installed versions of the required dependencies in `run_manifest.json` and the implementation decision log, and select the applicable profile. The frozen `en_core_web_sm` package and `all-MiniLM-L6-v2` model assets must also be locally available because the core execution path does not download resources from the network. If a required dependency or frozen model asset is unavailable, use only one of the explicitly defined degradation profiles in §3.3. If the full research design requires a missing dependency, the system must report that limitation rather than silently replacing the library.

This section therefore specifies (a) the required toolchain, (b) exactly which features are unavailable without each dependency, and (c) a stdlib-only floor.

### 3.2 Required dependencies

Every dependency below is named in the supplied context documents. No dependency has been introduced that the context does not already establish.

| Dependency | Version policy | Justified by | Features it enables |
|---|---|---|---|
| `numpy` | use installed exact version; record it | numeric core | all numeric work |
| `scikit-learn` | use installed exact version; record it | draft §6.1, §7 | TF-IDF, one-hot, imputation, PCA, baselines, metrics |
| `xgboost` | use installed exact version; record it | SPEC draft §8.1 | main learner |
| `spacy` + `en_core_web_sm` | use installed exact versions; record both | draft §6.1 | F01–F05, F09–F13, F14–F17, F26–F28 |
| `textstat` | use installed exact version; record it | draft §6.1 | F06–F08 |
| `sentence-transformers` + `all-MiniLM-L6-v2` | use installed exact versions/revision; record both | SPEC draft §14 config | F21, F22, F29, semantic branch |
| `pandas` | use installed exact version; record it | feature table handling | tabular assembly |
| `pyarrow` | use installed exact version; record it | chat §8 (Parquet) | feature-matrix persistence |
| `joblib` | use installed exact version; record it | draft §14 | artifact serialisation |

**DECISION 3.2.1 — `tree-sitter` is removed.** The draft (§6.1) proposed Tree-sitter for code parsing.
*Reason:* it is a heavyweight non-Python-native dependency requiring per-language grammar builds, it is not needed for F33–F35, and the Python standard library `ast` module plus a specified lexical fallback (§9.8) covers the requirement deterministically.
*Impact:* no loss of specified features; multi-language AST precision is reduced to the lexical fallback. Listed as a future extension.

**DECISION 3.2.2 — no near-duplicate-detection library.** §7.3 specifies MinHash/Jaccard mathematically so it is implementable with `numpy` + stdlib `hashlib` alone.
*Reason:* honours the dependency constraint; avoids adding `datasketch`.
*Impact:* the implementer writes ~40 lines rather than importing a library. Algorithm is fully specified, so behaviour is unambiguous.

### 3.3 Degradation profiles

If a dependency is unavailable and cannot be installed, the system must run a **named profile** and record that profile name in `run_manifest.json`. Cross-profile metric comparison is invalid and must be refused by the evaluation reporter.

| Profile | Missing dependency | Features dropped | Consequence |
|---|---|---|---|
| **P0 — Full** | none | none | The design as specified. Required for the research question to be answered. |
| **P1 — No semantic** | `sentence-transformers` | F21, F22, F29, entire semantic/PCA branch | Final matrix falls from 104 to 53 columns (§11.3). Ablations A6, A8 and the hybrid production model become unrunnable. Research question only partially answerable. |
| **P2 — No parser** | `spacy` | F09–F13 (syntax), degraded F01–F05, F14–F17, F26–F28 | Syntax family eliminated; ablation A2 unrunnable. Tokenisation falls back to the regex tokeniser of §8.3. |
| **P3 — No readability** | `textstat` | F06–F08 | Readability family eliminated; ablation A1 partially unrunnable. |
| **P4 — Floor** | only stdlib + `numpy` | F06–F13, F21, F22, F26–F29, semantic branch | ~15 of 35 features survive. **Not sufficient to answer the research question.** Acceptable only as a smoke-test path. |

**SPEC:** the system must fail loudly with a named missing dependency and the profile it would degrade to. It must **never** silently substitute a different library or silently emit zeros for an unavailable feature family (prompt §11, §18).

---

## 4. Assumptions

| # | Assumption | If false |
|---|---|---|
| A1 | The corpus satisfies §5 — authentic source labels, deduplicated, exactly three label values. | Ingestion gate (§7.1) rejects the run. |
| A2 | Question text is English. | Readability and the parser model become invalid; non-English detection is out of scope. Record as a corpus defect. |
| A3 | All three classes are present with sufficient count for stratification. | Training fails with a diagnostic (SPEC draft §2). |
| A4 | Code, when present, is embedded in the `question` string. | Programming branch yields `code_detected = 0` for all records; F33–F35 are constant and removed by the variance filter (§11.2). |
| A5 | Question length is bounded such that parsing is tractable. Cap specified in §8.5. | Oversized records are truncated for the semantic branch and flagged; see §8.5. |
| A6 | Compute is a single machine, CPU sufficient, GPU optional. | §20 budgets change; no design change. |

---

## 5. Upstream Corpus Contract (the 100,000-question dataset)

The corpus is produced by a **separate upstream process**. This section is the contract the model pipeline enforces at its boundary. The model pipeline does not build the corpus and never reads audit files.

### 5.1 Target and schema

**SPEC (data spec §1, §15):** exactly 100,000 valid records. Final model-facing schema contains exactly two logical fields:

```
{"question": "...", "label": "Easy"}
```

Allowed labels: `Easy`, `Moderate`, `Hard`. Preferred container: JSONL, one object per line, UTF-8. A JSON array is acceptable only where a downstream consumer requires it (§6.2).

### 5.2 Label authenticity — the governing constraint

**SPEC (data spec §3, §19):** the difficulty label must originate from the source dataset or a documented source-provided difficulty scheme.

Forbidden without exception:
- LLM-invented labels for unlabelled questions
- inferring difficulty from question length or apparent difficulty
- heuristic conversion of an unlabelled source into a labelled one
- fabricating labels to reach the 100,000 target

**SPEC (data spec §19.10):** if 100,000 authentic labelled questions cannot be reached under source and licensing constraints, **report the achievable count**. Do not fabricate the remainder.

**DECISION 5.2.1 — the 100,000 count is a target subordinate to authenticity.** Where the data spec's "exactly 100,000" (§1, §17) conflicts with authenticity (§3, §19.10), authenticity wins.
*Reason:* §19 is declared non-negotiable and §19.10 explicitly contemplates a shortfall; a fabricated label destroys the research validity that the entire project exists to establish.
*Impact:* the ingestion gate (§7.1) validates *schema, label validity, and duplication* as hard failures, but treats record count as a **reported metric with a configurable minimum** (`corpus.min_records`, default 10,000), not a hard equality check. A run on fewer than 100,000 authentic records is valid and must state the actual count in every report.

### 5.3 Label normalisation

**SPEC (data spec §4):** normalise source schemes to the three final labels only where the mapping is explicit and defensible. Never silently alter or erase the original source label during collection.

Mapping table is **configuration, not code** (`config/label_normalization.yaml`), so it is auditable and versioned:

```
EASY | Easy | easy | 1            -> Easy
MEDIUM | Medium | Moderate | 2    -> Moderate
HARD | Hard | 3                   -> Hard
VERY_HARD | Very-Hard             -> Hard
```

**DECISION 5.3.1 — `Medium-Hard` resolves to `Hard`.** The data spec (§4) leaves this to "a predefined, documented rule" without fixing one.
*Reason:* a documented deterministic rule is required; mapping upward is the conservative choice for an ordinal scale because it preserves the upper boundary of the source's own ordering rather than collapsing a distinction downward.
*Impact:* affects only sources using a 5-level scheme. The rule is recorded in the config file and in `dataset_statistics.json`, so its effect on class balance is auditable and reversible. Any unmapped source value is a **hard rejection**, never a guess (SPEC data spec §10).

### 5.4 Audit data — strict separation

**SPEC (data spec §5, §16, §13; prompt §14):** the collection pipeline maintains internal provenance, but it is **not** part of the model-facing data.

| Audit file | Contents |
|---|---|
| `provenance.jsonl` | `source_dataset`, `source_question_id`, `original_label`, `normalized_label`, `source_reference`, `license_basis`, `internal_domain` |
| `rejected_records.jsonl` | record + rejection reason + source |
| `duplicate_report.jsonl` | duplicate group, similarity method, retained record |
| `dataset_statistics.json` | totals, counts by label / source / domain, duplicate and rejection counts, final count |

**SPEC:** none of these fields may become a feature column. `internal_domain` in particular must not reach the model — it is exactly the "topic → difficulty shortcut" the project forbids (chat §3).

**DECISION 5.4.1 — provenance may be read by the splitter, never by the feature pipeline.** Data spec §14 requires avoiding same-family leakage across splits, which needs source knowledge; §13 forbids source metadata as model input. These are reconcilable because **splitting is not modelling**.
*Reason:* using `source_dataset` as a grouping key affects only which partition a record lands in. It cannot influence the learned function, because it is never a column in `X`.
*Impact:* the splitter (§7.3) accepts an optional provenance sidecar joined on `question_id`. The feature pipeline's interface physically cannot accept it — it receives only `(question_id, question, label, split)`. A test enforces this (§23, T-LEAK-04).

### 5.5 Corpus quality requirements

**SPEC (data spec §6, §10):** exclude empty questions, malformed records, missing or unmappable labels, corrupted text, metadata-only records, duplicates, and licence violations. Retain meaningful question text, code, and math content where present in the source.

**SPEC (data spec §10):** do **not** remove difficult or unusual questions merely because their text is atypical.

Balance target is ~1/3 per class (data spec §11) — **a target, not permission to relabel**.

---

## 6. Model-Facing Data Contract and Ingestion

### 6.1 Canonical internal record

| Field | Type | Required in input? | Derivation |
|---|---|---|---|
| `question` | string | **Yes** | verbatim from corpus |
| `label` | enum | **Yes** (training) | `Easy \| Moderate \| Hard` after §5.3 normalisation |
| `question_id` | string | **No** | if absent, `sha256(normalized_question)` truncated to 16 hex chars |
| `raw_hash` | string | derived | `sha256(normalized_question)`, full |
| `split` | enum | derived | `train \| validation \| test`, assigned in §7.3 |

**DECISION 6.1.1 — `question_id` is optional on input, required internally.** The draft TDD (§4) marks `question_id` as `Required: Yes`, contradicting the data spec's two-field schema (§1, §15).
*Reason:* the data spec is authoritative on the corpus contract and states no additional fields are required. But stable identifiers are needed for split reproducibility, embedding caches, and duplicate reports.
*Impact:* ingestion generates a deterministic content hash when the field is absent. Consequence to enforce: because the identifier is content-derived, **two records with identical normalised text receive the same `question_id`** — which the exact-duplicate stage (§7.2) resolves before any identifier is used as a key. If the corpus supplies `question_id`, it is used as-is and validated for uniqueness.

### 6.2 Ingestion behaviour

**SPEC (draft §4):** accept both JSONL and a JSON array; normalise both to the identical internal record structure. A test asserts byte-identical downstream results from equivalent inputs in either container (§23, T-SCHEMA-03).

Ingestion order of operations — **fixed**:

1. Read bytes, decode UTF-8 strictly. A decode error is a record-level rejection with reason `encoding_error`.
2. Detect container (JSONL vs array) by first non-whitespace character: `[` → array, else JSONL.
3. Parse each record. Malformed JSON → rejection `malformed_json`.
4. Assert `question` present, string, non-empty after whitespace strip → else rejection `missing_question`.
5. Normalise label case and surrounding whitespace only (SPEC draft §2): `easy`/`EASY`/`Easy` → `Easy`. Any value not in the mapping → rejection `unknown_label`. **Never** map an unknown label to a nearest neighbour.
6. Compute `raw_hash` and `question_id`.
7. Emit the canonical record.

Rejections are written to `rejected_records.jsonl` with reason and input line number. Ingestion reports counts by reason and **fails the run** if the rejection rate exceeds `ingest.max_reject_ratio` (default `0.02`) — a high rate signals corpus corruption rather than isolated bad records.

### 6.3 Inference-time contract

At inference, `label` is absent and `question_id` is optional:

```
{"question_id": "NEW-001", "question": "Given a graph..."}
```

**SPEC (chat §19):** the caller supplies the same raw structure used in training and never computes features manually.

---

## 7. Validation, Deduplication, and Splitting

Ordering is load-bearing. **SPEC (data spec §14, draft §7, chat §12):** validate → deduplicate → group into families → split → *then* fit any data-dependent transform.

### 7.1 Validation gate

Hard failures (abort the run, non-zero exit, diagnostic naming the violated rule):

| Check | Rule |
|---|---|
| Label domain | every label ∈ {Easy, Moderate, Hard} |
| Label presence | no missing/empty labels in training data |
| Question presence | no missing/empty question text |
| Class coverage | all three classes present (SPEC draft §2 — fail rather than pretend to support 3-class) |
| Class minimum | each class ≥ `validation.min_class_count` (default 30) so stratification is meaningful |
| Encoding | valid UTF-8 throughout |
| Identifier uniqueness | `question_id` unique after deduplication |
| Record count | ≥ `corpus.min_records`; **actual count reported, not asserted equal to 100,000** (DECISION 5.2.1) |

Reported, not fatal: class balance ratio, length distribution, MCQ-detected rate, code-detected rate, per-source counts when provenance is present.

### 7.2 Exact deduplication

**SPEC (data spec §9A):** normalise whitespace and case **for detection only**; retain one record per exact normalised match. The retained record keeps its original text unmodified.

Detection key:
```
norm(q) = NFKC(q) → collapse all whitespace runs to single space → strip → casefold
raw_hash = sha256(norm(q))
```

Retention rule (deterministic): among records sharing a `raw_hash`, retain the one with the lexicographically smallest original `question_id`; if none supplied, retain the earliest input position.

**Label-conflict handling — SPEC (data spec §9):** where duplicates carry conflicting labels, do **not** arbitrarily choose. Behaviour:

1. Record the conflict in `duplicate_report.jsonl`.
2. If a documented source-priority rule exists in configuration, apply it and log which rule fired.
3. Otherwise **discard the entire conflicting group** and count it under `rejected_records.jsonl` reason `duplicate_label_conflict`.

**DECISION 7.2.1 — discard is the default for unresolved label conflicts.** The data spec permits "resolve only when there is a documented source-priority rule **or** discard".
*Reason:* discarding is the only option that introduces no label noise. Majority-vote or first-wins would silently fabricate a label decision, violating §5.2.
*Impact:* small reduction in corpus size; conflict count is reported so the loss is visible. No silent label invention.

### 7.3 Near-duplicate families and group-safe splitting

**SPEC (data spec §9B, §14; draft §7):** detect lightly edited copies; do not place near-identical questions from the same family in both train and test.

**Algorithm (fully specified — implementable with `numpy` + stdlib `hashlib`, per DECISION 3.2.2):**

1. **Shingling.** From `norm(q)`, produce the set of character 5-grams. Character-level shingling is chosen over word-level because it is robust to the punctuation and code fragments common in this corpus.
2. **MinHash signature.** `H = 128` hash permutations. Permutation *i* uses `sha256(shingle || i)` reduced to a 64-bit integer; signature component *i* is the minimum over shingles. Seed fixed at `42`.
3. **Jaccard estimate.** `Ĵ(a,b) = (1/H) · |{i : sig_a[i] == sig_b[i]}|`.
4. **Candidate generation via LSH banding.** 32 bands × 4 rows. Two records are candidates if any band matches exactly. This bounds cost to ~O(N) expected rather than O(N²).
5. **Family formation.** Candidate pairs with `Ĵ ≥ dedup.near_threshold` (default `0.85`) become graph edges. **Families are connected components** of this graph.
6. **Group key.** `group_key = family_id` **globally across all sources**. Provenance may be used for reporting and audit, but `source_dataset` must never be added to the grouping key. A near-duplicate family must remain entirely within one split even when its members originate from different sources.

**DECISION 7.3.3 — family grouping is global, not source-scoped.**
*Reason:* the family graph is built across the complete corpus. Adding `source_dataset` to the group key can split the same underlying question across train/validation/test when identical or near-identical records come from different sources, directly violating the leakage constraint.
*Impact:* provenance remains available to the splitter for diagnostics and reporting, but never changes family membership or split assignment.

**Splitting.** Stratified **group** split, 70/15/15 (SPEC data spec §14, draft §8.2), `random_seed = 42`:

- Whole groups are assigned; a group never spans splits.
- Stratification target is the group's **majority label**, which keeps class proportions near-constant while respecting group integrity.
- Assignment is deterministic: groups sorted by `group_key`, then allocated by a seeded greedy pass that assigns each group to whichever split is furthest below its target quota for that group's label.
- Because groups are indivisible, realised split ratios may deviate slightly from 70/15/15. Deviation is reported; a deviation above `split.max_ratio_deviation` (default `0.02` absolute) fails the run so a pathological giant group cannot silently distort the design.

**DECISION 7.3.1 — group split replaces the draft's plain stratified split.** Draft §8.2 specifies "stratified 70/15/15 at question level" while also requiring near-duplicate groups not to cross splits — these are incompatible when a family contains more than one record.
*Reason:* per-question stratification cannot honour group integrity. Group-aware assignment satisfies both requirements, with stratification approximate rather than exact.
*Impact:* exact per-class proportions are no longer guaranteed; they are measured and bounded instead. This is the correct trade — leakage is a correctness failure, a 1% class-proportion drift is not.

**DECISION 7.3.2 — small-dataset fallback threshold is fixed.** Draft §8.2 says "if the dataset is small, use repeated stratified K-fold" without defining *small*.
*Reason:* prompt §11 forbids leaving an implementation-determining choice open.
*Impact:* if `N < 5000`, use repeated stratified group K-fold (5 folds × 3 repeats, seed 42) for model selection and retain a 15% group-held-out test set. At the specified corpus scale (10⁴–10⁵) the single split is used and is statistically ample.

### 7.4 Leakage controls — complete enumeration

**SPEC (draft §11, chat §14, prompt §13).** Fitted **on train split only**, without exception:

| Artifact | Why it leaks if fitted on all data |
|---|---|
| TF-IDF vocabulary + IDF | test term distribution informs train features |
| Rare-word frequency table (F05) | corpus statistics encode test vocabulary |
| High-TF-IDF threshold (F25) | a learned percentile over test data |
| Numeric imputer medians | test central tendency enters train |
| One-hot category vocabulary | test-only categories become known |
| Correlation-filter mask | selection informed by test structure |
| PCA components (semantic branch) | test variance structure shapes the basis |
| Model parameters, early-stopping round | validation may inform stopping; test may not |
| Hyperparameter choice | test must never inform selection |

Also forbidden (SPEC):
- Any label-derived feature. No feature may read `label`.
- Any source-difficulty value re-encoded as an input (data spec §13).
- Any learner-response or performance statistic.
- Any generated difficulty explanation.
- Test set touched more than once, and only after model selection is final (SPEC data spec §14, draft §9).

### 7.5 What is *not* leakage — explicit clarification

The pretrained sentence encoder and the pretrained spaCy model are **frozen artifacts** (§2). Using them on all splits is **not** leakage: no parameter is estimated from project data, so no test information flows into train-time decisions. Only the **PCA fitted on top of the embeddings** is a fitted artifact and is therefore train-only.

This distinction is stated because a naive reading of "fit only on training data" would wrongly forbid embedding the test set at all.

---

## 8. Preprocessing and Text Views

**SPEC (draft §5):** preprocessing is deterministic and preserves information downstream features need. Do **not** aggressively strip punctuation, casing, operators, or code — these carry difficulty signal.

### 8.1 Normalisation

Applied to produce the working text; the original is always retained:

1. Unicode NFKC normalisation.
2. Line-ending normalisation to `\n`.
3. Collapse runs of 3+ blank lines to 2. Trailing whitespace per line stripped.
4. **No** case folding, **no** punctuation removal, **no** stop-word removal at this stage.

Case folding and whitespace collapsing are used **only** inside `norm(q)` for duplicate detection (§7.2) and inside the TF-IDF analyser (§9.6).

### 8.2 Text views — the allocation table

Features are computed from different projections of the text. Mixing these up is the most likely silent-correctness bug in this pipeline, so the allocation is fixed here.

| View | Construction | Consumed by |
|---|---|---|
| `raw_text` | normalised text per §8.1, otherwise untouched | provenance, hashing, code/MCQ detection |
| `code_spans` | ordered list of detected code regions (§8.6) | F30–F35 |
| `nl_view` | `raw_text` with every code span replaced by a single space, and every math span replaced by the literal token `MATHEXPR` | F01–F17, F23–F29 |
| `option_block` | the detected MCQ option region (§8.7) | F18–F22 |
| `nl_stem` | `nl_view` with `option_block` removed | F21 (stem side) |
| `sentences` | sentence segmentation of `nl_view` | F02, F29 |

**Rationale for masking:** raw source code and LaTeX destroy readability formulas (syllable counting over `for(i=0;i<n;i++)` is meaningless) and corrupt dependency parses. Math becomes a single pronounceable token so sentence structure survives without absurd syllable counts. MCQ options are *retained* in `nl_view` because they are genuine natural language and part of the reading burden.

### 8.3 Tokenisation and parsing

**DECISION 8.3.1 — the parser is fixed to spaCy `en_core_web_sm`, dependency parse only.** Draft §6.1 offered "spaCy or a tested equivalent" and "spaCy + a parser or Stanza; constituency parser optional". Prompt §11 forbids this; chat §22 requires a fixed default.
*Reason:* a single pinned parser is required for reproducibility. `en_core_web_sm` is the small English pipeline — adequate for the tokenisation, POS, dependency, and noun-chunk needs of F01–F17 at 10⁵-document scale, and materially cheaper than a transformer-based pipeline.
*Impact:* **constituency parsing is removed from the design.** F11 is redefined as *dependency* tree depth (§9.3). Any future parser change is a breaking feature-schema change requiring a new `feature_schema_version`.

Fallback tokeniser (profile P2/P4 only): split on Unicode word boundaries; sentence split on `[.!?]` followed by whitespace-and-uppercase, with abbreviation exceptions listed in config. This fallback changes feature values and therefore requires a distinct `feature_schema_version`.

Parsed documents are cached (§20.2) because parsing dominates extraction cost.

### 8.4 Determinism requirements

- Single-threaded parsing, or `n_process` fixed in config with results reassembled in input order.
- No randomness in any extractor. Every feature is a pure function of its view plus fitted artifacts.
- Dictionary/set iteration must never influence output ordering; all lexicon scans iterate a sorted key list.

### 8.5 Length handling

**SPEC-adjacent, resolved here.** The sentence encoder `all-MiniLM-L6-v2` truncates at 256 word-pieces. Long programming problems will exceed this.

**DECISION 8.5.1 — truncation is explicit, asymmetric, and flagged.**
*Reason:* silent truncation would make the semantic branch quietly ignore most of a long question while engineered features saw all of it — an inconsistency invisible in metrics.
*Impact:*
- Engineered features (F01–F35) always use the **full** text. No truncation.
- The semantic branch encodes the **first 256 word-pieces** of `nl_view`.
- A `semantic_truncated` flag is recorded per record in the feature table for diagnostics. It is **not** a model column (it correlates with length, which F01 already carries).
- Records exceeding `preprocess.max_chars` (default 20,000) are rejected as `oversized` rather than processed, guarding against pathological inputs.

### 8.6 Code detection

Deterministic cascade; first match wins, and all matches are collected as spans:

1. **Fenced blocks** — ``` ``` ``` or ``` ~~~ ``` delimited regions. Highest confidence.
2. **Indented blocks** — ≥ 4 consecutive lines each beginning with ≥ 4 spaces or a tab, and containing ≥ 1 code signature.
3. **Syntax signatures** — a line matching any configured pattern: `def `, `class `, `function `, `public static`, `#include`, `import `, `for(`, `while(`, `if(`, `=>`, `;` at line end, balanced `{}` blocks.

`code_detected = 1` iff at least one span is found. Language is inferred only as `python | other` (§9.8). Confidence tier (fenced / indented / signature) is logged for audit, not used as a feature.

### 8.7 MCQ detection

Option markers, scanned line-anchored (SPEC draft §5):

```
(A) (a) (1)      A) a) 1)      A. a. 1.      Option A    [A]
```

Rules:
- Require **≥ 2** distinct sequential markers to declare MCQ. A single `A)` is prose, not an option list.
- Markers must be **sequentially ordered** from the first (A,B,C… or 1,2,3…). A non-sequential set is rejected as a false positive.
- `option_block` spans the first marker to the end of the last option's text.
- `mcq_detected = 1` iff the above holds. Parser confidence tier recorded for audit.

**Explicit constraint (SPEC data spec §8):** MCQ parsing yields options only. It **never** identifies a correct answer, because no answer key exists in the contract. This directly constrains §9.5.

---

## 9. Feature Specification (35 logical features)

**SPEC (draft §6, chat §9):** exactly 35 logical features, fixed in the schema so training and inference cannot drift.

Every feature below specifies: view, formula, dtype, range, and missing-value behaviour. Where a feature is unavailable it emits the stated **missing sentinel** and the relevant indicator (§9.9) — **never a fabricated value** (SPEC draft §6).

Missing sentinel convention: numeric features emit `NaN`, resolved to the train-set median by the imputer (§11.2). This keeps "unavailable" distinguishable from "genuinely zero" — a distinction that matters because `option_count = 0` (not MCQ) and `constraint_count = 0` (MCQ with no constraints) are different facts.

### 9.1 Lexical / length (F01–F05) — view: `nl_view`

A **word token** is a token where `is_punct = false`, `is_space = false`, and the token contains ≥ 1 alphanumeric character.

| ID | Feature | Formula | Type | Range | Missing |
|---|---|---|---|---|---|
| F01 | `word_count` | count of word tokens | int | ≥ 0 | never |
| F02 | `avg_sentence_length` | `F01 / max(1, n_sentences)` | float | ≥ 0 | never |
| F03 | `avg_word_length` | `Σ len(token.text) / max(1, F01)` | float | ≥ 0 | never |
| F04 | `type_token_ratio` | `|distinct casefolded word tokens| / max(1, F01)` | float | (0, 1] | never |
| F05 | `rare_word_ratio` | `|{t : df_train(casefold(t)) < τ_rare}| / max(1, F01)` | float | [0, 1] | never |

`df_train` is the document frequency of a token across the **training split only**. `τ_rare = features.rare_word_df_threshold`, default `5`. Tokens absent from the training vocabulary count as rare (`df = 0`), which is the correct treatment at inference for genuinely unseen words.

**Fitted artifact:** `rare_word_df.joblib` — the training-split document-frequency table.

### 9.2 Readability (F06–F08) — view: `nl_view`

| ID | Feature | Source | Type | Range |
|---|---|---|---|---|
| F06 | `flesch_reading_ease` | Flesch Reading Ease | float | unbounded, typically [-50, 120] |
| F07 | `flesch_kincaid_grade` | Flesch–Kincaid grade level | float | typically [0, 30] |
| F08 | `gunning_fog` | Gunning Fog index | float | typically [0, 30] |

Computed via `textstat` on `nl_view`. If `F01 < 3` or `n_sentences < 1` the formulas are numerically unstable (division by near-zero): emit `NaN` and rely on imputation. Values are **not** clipped — an extreme negative Flesch score is real signal about a dense technical sentence.

**Interpretation constraint (SPEC chat §10):** readability is one signal among several. It must not be presented as equivalent to difficulty.

### 9.3 Syntax (F09–F13) — view: `nl_view`, dependency parse

| ID | Feature | Formula | Type | Missing |
|---|---|---|---|---|
| F09 | `clause_count` | count of tokens whose `dep_` ∈ clause-head set | int | `NaN` if parse fails |
| F10 | `avg_noun_phrase_length` | mean token count over `noun_chunks`; `0` if none | float | `NaN` if parse fails |
| F11 | `parse_tree_depth` | `max` over tokens of hop count to the sentence root, maximised over sentences | int | `NaN` if parse fails |
| F12 | `dependency_complexity` | Mean Dependency Distance (below) | float | `NaN` if parse fails or `N < 2` |
| F13 | `negation_count` | count of **unique token positions** satisfying `dep_ = 'neg'` OR matching a configured negation cue | int | `NaN` if parse fails |

Clause-head dependency set (configurable, default): `{ROOT, ccomp, xcomp, advcl, relcl, acl, csubj, csubjpass}`.

**DECISION 9.3.1 — F12 is Mean Dependency Distance, exactly.** Draft §6.1 offered "average dependency distance **or** a calibrated tree-complexity score"; chat §22 demands one formula.

For a sentence with tokens indexed by surface position, over all non-root tokens *t*:

```
MDD(sentence) = ( Σ |index(t) − index(head(t))| ) / (N − 1)
F12 = mean over sentences of MDD(sentence)
```

*Reason:* MDD is a standard, parameter-free, deterministic measure of syntactic integration cost. A "calibrated complexity score" would require calibration data that does not exist in this project.
*Impact:* one formula, no tuning, no ambiguity. Sentences with `N < 2` are skipped in the mean; if no sentence qualifies, F12 is `NaN`.

**DECISION 9.3.2 — F11 is dependency depth, not constituency depth.** Follows DECISION 8.3.1 (no constituency parser).
*Reason:* `en_core_web_sm` provides no constituency parse; requiring one would add a dependency and a second parser.
*Impact:* F11 semantics change from the draft's "constituency/dependency" to strictly dependency-tree depth. Recorded in the feature schema description so it cannot be misread later.

Negation lexicon (config): `no, not, never, none, neither, nor, without, cannot, n't, unless, except`. A token that satisfies both the dependency and lexicon tests counts **once**, by token position.

**DECISION 9.3.3 — F13 uses a union, not arithmetic addition.**
*Reason:* a token such as `not` can satisfy both the dependency relation and the lexicon cue, so simple addition double-counts the same linguistic event.
*Impact:* F13 is deterministic and avoids parser/lexicon overlap inflating the feature.

### 9.4 Question / cognitive structure (F14–F17) — view: `nl_view`

**F14 `question_type`** — categorical, **closed set of 8**: `What, Why, How, Which, When, Where, YesNo, Other`.

Deterministic precedence — first rule that matches wins, evaluated in this exact order:

1. Leading interrogative word of the first question-bearing sentence ∈ {what, why, how, which, when, where} → that class.
2. Sentence begins with an auxiliary or modal (`is, are, was, were, do, does, did, can, could, will, would, should, has, have, had`) **and** ends with `?` → `YesNo`.
3. Any of the six interrogatives appears anywhere in the text → that class (earliest surface position wins; ties broken by the fixed order above).
4. Otherwise → `Other`.

Imperative prompts ("Design an algorithm…") legitimately fall to `Other`; their task nature is captured by F15, not F14.

**F15 `instruction_type`** — categorical, **closed set of 10**: `Define, Explain, Calculate, Compare, Analyze, Design, Prove, Describe, Identify, Other`.

Matching: a configured verb lexicon maps lemmas to classes. Scan lemmas of `nl_view` in surface order; the **first** lemma matching any class assigns that class; no match → `Other`. Lexicon lives in `config/task_lexicon.yaml`.

**DECISION 9.4.1 — F14 and F15 vocabularies are closed and fixed in configuration.** The draft wrote "Define/Explain/Calculate/Compare/Analyze/Design/Prove/**etc.**" — an open set cannot be one-hot encoded reproducibly.
*Reason:* an unbounded category set makes the final matrix width non-deterministic and breaks schema validation at inference.
*Impact:* exactly 8 and 10 one-hot columns respectively (§11.3). An unrecognised pattern maps to `Other`, never to a new category. Adding a category is a breaking schema change.

**F16 `cognitive_demand`** — ordinal integer 1–6 (Bloom): `Remember=1, Understand=2, Apply=3, Analyze=4, Evaluate=5, Create=6`.

**DECISION 9.4.2 — F16 uses a deterministic verb lexicon; no LLM, no classifier.** SPEC chat §22 requires a controlled deterministic mapping or omission.
*Reason:* the explicit prohibition on implementations silently using different LLM classifications. An LLM call would also break determinism, reproducibility, and the offline constraint of §3.
*Impact:* F16 = **maximum** Bloom level over all matched verb lemmas, `1` if none match. Rationale for max rather than first-match: a question asking the learner to "define X and then evaluate Y" is governed by its most demanding operation. Retained as a single ordinal integer column (not one-hot) because the levels are genuinely ordered.
*Status:* **HYPOTHESIS** — that lexicon-derived Bloom level correlates with measured difficulty is an expectation, not an established finding. It is tested by ablation A3, and must be reported as such.

**F17 `subquestion_count`** — integer. `count of line-anchored enumerators` + `max(0, count('?') − 1)`.

Enumerator patterns (config): `(i)`, `(ii)`, `(a)`, `(b)`, `1.`, `2.` at line start, requiring ≥ 2 sequential markers, and **excluding** any span inside `option_block` so MCQ options are not double-counted as subquestions. Value `0` means a single undivided question.

### 9.5 MCQ structure (F18–F22) — view: `option_block`, `nl_stem`

All five require `mcq_detected = 1`. When `mcq_detected = 0`: F18 = `0` (a real fact — there are no options), F19–F22 = `NaN`.

| ID | Feature | Formula | Type |
|---|---|---|---|
| F18 | `option_count` | number of detected options | int |
| F19 | `avg_option_length` | mean word-token count per option | float |
| F20 | `option_length_variance` | population variance of option word counts | float |
| F21 | `stem_option_similarity_mean` | `mean_i cos( E(nl_stem), E(option_i) )` | float [-1, 1] |
| F22 | `option_option_similarity_mean` | `mean_{i<j} cos( E(option_i), E(option_j) )` | float [-1, 1] |

`E(·)` is the frozen sentence encoder (§10.1) applied to raw (un-reduced) embeddings.

**DECISION 9.5.1 — F21 and F22 are redefined to be answer-key-independent. This is the single most important correction in this document.**

The draft TDD defines:
- F21 `stem_answer_similarity` — "similarity between stem and candidate **correct answer**"
- F22 `distractor_answer_similarity` — "mean **distractor-to-answer** similarity"

Both require knowing which option is correct. **No answer key exists in the data contract** (SPEC data spec §8: "The final dataset MUST NOT add a separate correct-answer field… The feature-extraction pipeline therefore must NOT depend on knowing the correct answer"). Chat §22 flags this as a blocking issue. The draft features are therefore **not computable** and, if implemented as written, would force the coding stage either to invent an answer-identification heuristic (fabricating a label-adjacent signal) or to emit these features as permanently missing.

*Reason:* the intent behind the draft features is to capture *option-set discriminability* — how hard it is to tell the options apart, and how closely they relate to the stem. That intent is fully recoverable without an answer key, exactly as chat §9 already specified ("answer-key independent" for both).
*Impact:*
- F21 becomes mean stem-to-option similarity — measures how on-topic the option set is relative to the question.
- F22 becomes mean pairwise option-to-option similarity — high values mean confusable options, the property "distractor quality" was reaching for.
- Both are computable for every MCQ under the real contract.
- `distractor` terminology is removed from the codebase entirely, because identifying a distractor presupposes identifying the answer.
- The research intent (MCQ structure contributes information) is preserved and remains testable via ablation A4.

**DECISION 9.5.2 — F20 uses population variance and is `NaN` for `option_count < 2`.** Prevents an undefined sample variance at *n* = 1 from becoming a silent zero.

### 9.6 Lexical importance / TF-IDF (F23–F25) — view: `nl_view`

**Critical scope clarification:** the full TF-IDF sparse matrix is **not** a model input. Only three scalar summaries enter the model (SPEC draft §6 F23–F25, chat §9 "Lexical Importance (3)"). This is what keeps the feature set compact.

Vectoriser configuration — fixed: lowercase, word 1-grams, `min_df = 2`, `max_features = 50000`, sublinear TF, L2 norm, English stop words **retained** (stop-word density is plausibly informative and removal would discard it).

Let `v` be the document's TF-IDF vector and `NZ(v)` its non-zero entries.

| ID | Feature | Formula | Missing |
|---|---|---|---|
| F23 | `mean_tfidf` | `mean(NZ(v))` | `NaN` if `NZ(v)` empty |
| F24 | `max_tfidf` | `max(NZ(v))` | `NaN` if `NZ(v)` empty |
| F25 | `high_tfidf_term_count` | `|{w ∈ NZ(v) : w > τ_tfidf}|` | `0` if `NZ(v)` empty |

**Fitted artifacts:** `tfidf.joblib` (vocabulary + IDF) and `τ_tfidf`.

**DECISION 9.6.1 — `τ_tfidf` is the 90th percentile of all non-zero TF-IDF weights in the training split.** The draft left it as an unspecified "configured threshold".
*Reason:* an absolute constant is meaningless because TF-IDF magnitudes depend on corpus size and vocabulary; a data-derived percentile is scale-free. It must be fitted on train only, and is therefore a persisted artifact, not a config constant.
*Impact:* `τ_tfidf` is stored in `tfidf_threshold.json` and reused verbatim at inference. Percentile is configurable via `features.tfidf_high_percentile` (default 90).

### 9.7 Semantic / conceptual (F26–F29) — view: `nl_view`

| ID | Feature | Formula | Type | Missing |
|---|---|---|---|---|
| F26 | `concept_count` | count of **distinct** concept-lexicon terms matched | int | never |
| F27 | `concept_density` | `F26 / max(1, F01)` | float | never |
| F28 | `technical_term_ratio` | `|technical token matches| / max(1, F01)` | float | never |
| F29 | `semantic_similarity` | `mean_{i<j} cos( E(s_i), E(s_j) )` over sentences | float | `NaN` if `n_sentences < 2` |

**DECISION 9.7.1 — F26/F28 use a fixed configured lexicon, not open-ended extraction.** The draft proposed "domain lexicon + noun-phrase/keyphrase extraction", which is two different mechanisms with different outputs.
*Reason:* keyphrase extraction is non-deterministic across library versions and would make `concept_count` unstable; the project explicitly prefers deterministic configurable definitions (chat §11, §24).
*Impact:* `config/concept_lexicon.yaml` holds two sorted term sets — `concepts` (domain entities and named techniques) and `technical_terms` (broader technical vocabulary). Matching is case-insensitive whole-token or multi-token phrase match, scanned in sorted key order. F26 counts distinct matches; multi-token phrases are matched greedily longest-first. **The lexicon's coverage bounds these features** — this is a known limitation (§25, UNRESOLVED-3), not a hidden one.

**DECISION 9.7.2 — F29 is mean pairwise inter-sentence cosine.** The draft said only "aggregate semantic coherence/similarity among question segments".
*Reason:* "aggregate" over "segments" is undefined; sentences are the one segmentation already available and deterministic.
*Impact:* exact formula above. Single-sentence questions yield `NaN` plus the `multi_segment = 0` indicator — not a fabricated `1.0`, which would falsely assert perfect coherence. Cost is bounded: `O(k²)` cosines over `k` sentence embeddings, `k` capped at `features.max_sentences_for_f29` (default 40) taking the first 40.

### 9.8 Programming-specific (F30–F35)

**Applicability asymmetry — important.** F30–F32 derive from *natural language* (algorithm names, constraint statements) and are computable for a prose programming problem with no code. Only F33–F35 require actual code. The draft implicitly treated all six as code-gated; they are not.

| ID | Feature | View | Formula | Missing |
|---|---|---|---|---|
| F30 | `algorithm_or_ds_complexity` | `nl_view` + `code_spans` | `max` weight over matched lexicon entries; `0` if none | never |
| F31 | `max_input_size_log` | `nl_view` | `log10(1 + max magnitude)` from constraint patterns | `NaN` if none detected |
| F32 | `constraint_count` | `nl_view` | count of constraint-pattern matches | never (`0` valid) |
| F33 | `code_length` | `code_spans` | count of non-blank code lines | `NaN` if `code_detected = 0` |
| F34 | `cyclomatic_complexity` | `code_spans` | see below | `NaN` if `code_detected = 0` |
| F35 | `loop_nesting_complexity` | `code_spans` | see below | `NaN` if `code_detected = 0` |

**DECISION 9.8.1 — F30 is a configured lexicon maximum, with fixed integer weights.** SPEC chat §11 and §22: algorithmic complexity scores "must be defined deterministically in configuration; they should not be invented ad hoc by the generated code."
*Reason:* this is an explicit upstream requirement, and it is the difference between a reproducible feature and one that varies per implementation.
*Impact:* `config/algorithm_complexity.yaml` maps terms to weights 1–5, for example: `array, string → 1`; `hash map, sorting, stack, queue → 2`; `binary search, tree, heap, recursion → 3`; `graph, BFS, DFS, dynamic programming, topological sort → 4`; `max-flow, suffix automaton, segment tree, FFT, NP-hard → 5`. F30 is the **maximum** matched weight because a problem's difficulty is governed by its hardest required technique, not the average of everything it mentions. `0` means no known technique named — a real observation, not missingness.
*Status:* **HYPOTHESIS** — these weights are an engineering ordering, not an empirically calibrated difficulty scale. They must never be reported as validated.

**F31 / F32 constraint patterns** (config, `constraint_patterns.yaml`): expressions of the form `n <= NUM`, `1 ≤ N ≤ NUM`, `N < NUM`, `up to NUM`, `at most NUM`, with `NUM` allowing `10^k`, `1e6`, and digit-group separators. F31 takes the maximum parsed magnitude; F32 counts matches. `log10(1 + ·)` compresses the 10⁰–10¹⁸ range into a well-scaled feature.

**DECISION 9.8.2 — F34/F35 use stdlib `ast` for Python, and a specified lexical fallback otherwise.** Replaces the draft's Tree-sitter proposal (DECISION 3.2.1).

- **Python path** — attempt `ast.parse` on the concatenated code spans. On success:
  - `F34 = 1 + count of {If, For, AsyncFor, While, ExceptHandler, BoolOp, IfExp, comprehension-if} nodes` — standard cyclomatic counting over the decision nodes available in the AST.
  - `F35 = Σ over loop nodes of depth(loop)`, where `depth` is 1-based nesting among loop ancestors. A doubly-nested loop contributes 1 + 2 = 3. Recursion (a function whose body calls its own name) adds `features.recursion_weight` (default 2).
- **Non-Python / parse-failure path** — lexical fallback:
  - `F34 = 1 + count of branch keywords` from a config list (`if, else if, elif, for, while, case, catch, &&, ||, ?`).
  - For brace-delimited code, scan characters left-to-right; increment `brace_depth` after `{`, decrement it before processing `}`, and evaluate each loop-keyword occurrence at the current non-negative depth. For indentation-based code, expand tabs to `features.indent_width` spaces (default `4`), compute `indent_level = leading_spaces // features.indent_width`, and evaluate each loop-keyword occurrence at that line's `indent_level`. `F35` is the sum of the depth/indent level at each detected loop occurrence.
  - The fallback sets `code_parse_exact = 0` in the audit log. This flag is **not** a model column; it is diagnostic, because fallback quality varies by language and should not be learned from.

**DECISION 9.8.3 — lexical nesting fallback is explicitly defined.**
*Reason:* the previous “brace- or indentation-derived nesting” wording permitted materially different implementations.
*Impact:* the non-Python fallback is deterministic; changing `indent_width` changes the feature schema and therefore requires a new `feature_schema_version`.

*Impact:* no external parser dependency; deterministic for Python (the common case in labelled programming corpora) and bounded-quality elsewhere. Multi-language exact AST parsing is a documented future extension.

### 9.9 Missingness indicators

Five binary columns make structural unavailability explicit (SPEC draft §6, §7.4):

| Indicator | `1` means | Governs |
|---|---|---|
| `mcq_detected` | options found | F18–F22 |
| `code_detected` | code found | F33–F35 |
| `input_size_detected` | a constraint magnitude parsed | F31 |
| `parse_ok` | dependency parse succeeded | F09–F13 |
| `multi_segment` | `n_sentences ≥ 2` | F29 (and F12 stability) |

These are **model columns**, unlike the diagnostic flags (`semantic_truncated`, `code_parse_exact`, parser confidence tiers) which are audit-only. The distinction: an indicator is included when the *fact of absence* is plausibly informative about difficulty; a diagnostic is excluded when it reflects *our tooling's* limitation rather than a property of the question.

### 9.10 Feature-family map (for ablation switching)

| Family | Feature IDs | Config flag |
|---|---|---|
| lexical | F01–F05 | `use_lexical` |
| readability | F06–F08 | `use_readability` |
| syntax | F09–F13 | `use_syntax` |
| task | F14–F17 | `use_task` |
| mcq | F18–F22 | `use_mcq` |
| tfidf | F23–F25 | `use_tfidf` |
| semantic_scalar | F26–F29 | `use_semantic_scalar` |
| programming | F30–F35 | `use_programming` |
| semantic_embedding | PCA block (§10) | `use_semantic_embedding` |

Disabling a family drops its columns **and** its exclusive indicators. Indicators shared across families (`parse_ok`, `multi_segment`) are retained while any dependent family is active.

---

## 10. Semantic Representation Branch

**SPEC (draft §7 item 8, chat §13):** a sentence-embedding branch reduced by PCA, kept separately comparable to the engineered features. The literature does **not** establish that raw high-dimensional embeddings are optimal (SPEC chat §3, §13) — hence reduction plus explicit ablation rather than assumption.

### 10.1 Encoder

**DECISION 10.1.1 — encoder fixed to `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions.** Taken from the draft's own config block (§14), promoted from default to specification.
*Reason:* the design requires a "lightweight" encoder (draft §12); leaving the choice open would make embeddings, PCA basis, and all similarity features irreproducible.
*Impact:* embedding dimension 384, max sequence 256 word-pieces (see DECISION 8.5.1). The encoder is a **frozen artifact** (§7.5) — no fine-tuning. Model name and revision are recorded in `semantic_encoder.json`; changing either invalidates the feature schema.

The **same** encoder instance serves F21, F22, F29, and the branch below. Embeddings are computed once per text unit and cached (§20.2). A text unit is uniquely identified by its role and content, so question, stem, option, and sentence embeddings cannot collide in the cache.

### 10.2 Reduction

- Input: 384-dim L2-normalised embedding of `nl_view` (truncated per §8.5).
- **PCA fitted on the training split only**, `n_components = semantic.pca_components`, default **48** (draft §14 config; within the 32–64 range of chat §13).
- `svd_solver = 'full'`, `random_state = 42` for determinism.
- Output: 48 columns named `sem_pca_000 … sem_pca_047`.
- Explained-variance ratio is recorded in `metrics.json`. If cumulative explained variance is below `semantic.min_explained_variance` (default 0.60), the run emits a **warning**, not a failure — low variance capture is a finding about the embedding space, not a pipeline fault.

**DECISION 10.2.1 — 48 components is a configured default, not a validated optimum.** Chat §13 explicitly says the dimension should be "configured and validated rather than hard-coded as a scientific truth."
*Reason:* honours the explicit instruction not to present a hyperparameter as a finding.
*Impact:* 48 is the default; `{32, 48, 64}` is a defined sweep in §14 (run A8b). Any claim about the best dimension must cite that sweep's validation results, never the default.

### 10.3 Leakage position

PCA is fitted **after** splitting, on train rows only, then applied to validation and test. The encoder itself is applied to all rows (§7.5 — not leakage). A test enforces that PCA's `fit` is never called with non-train indices (§23, T-LEAK-02).

---

## 11. Feature Transformation and Final Matrix

### 11.1 Pipeline order — fixed

1. Extract 35 logical features + 5 indicators per record (§9).
2. Drop **constant** columns (zero variance on train).
3. Impute numeric `NaN` → **train-split median**. Persist the imputer.
4. One-hot encode F14 (8) and F15 (10), categories fixed by config, `handle_unknown = 'ignore'` (unknown cannot occur, since both vocabularies are closed and `Other` is a member — the setting is a defensive guard).
5. Append the 48 PCA columns (§10).
6. Apply the **correlation filter** (§11.2).
7. Emit `X_final` in the exact persisted column order.

**No scaling by default.** The main learner is tree-based and scale-invariant. Scaling is applied **only** for the logistic-regression baseline, inside that baseline's own pipeline, fitted on train only — so it can never contaminate the shared matrix.

### 11.2 Redundancy removal — and what is *not* used

**SPEC (draft §7 item 6):** remove redundant engineered features using a training-only correlation threshold `|r| > 0.90`, retaining the more stable feature.

Deterministic procedure: compute the Pearson matrix over train rows; scan column pairs in fixed schema order; for a pair exceeding the threshold, drop the **later** column in schema order. Fixed ordering makes the outcome reproducible, which a greedy "keep the more interpretable one" rule would not be. The retained mask is persisted as `feature_selector.joblib`.

**DECISION 11.2.1 — mutual information, permutation importance, and SHAP are analysis-only. They are not in the production selection path.**

The draft (§7 item 7) says to "rank remaining features with mutual information and model-based importance… selection is a learned artifact." The chat context (§14) says the recommendation is to use MI, permutation importance, and SHAP "primarily for analysis/ablation rather than forcing an unstable multi-stage feature-selection procedure into the production baseline", and §22 requires a stable baseline procedure not dependent on "a vague combination of MI/SHAP/RFE."

*Reason:* the chat context is the later and more specific instruction, and it is technically correct — multi-stage selection over ~100 columns on this corpus adds variance and a leakage surface for negligible benefit, while tree ensembles already handle redundant features.
*Impact:* production selection = **variance filter + correlation filter only**. MI, permutation importance, and SHAP are computed in the interpretability stage (§15) and reported, but never gate a column. This removes an entire class of tuning-leakage risk.

### 11.3 Exact dimensionality

The draft's "35 engineered features after selection" and "approximately 67–99 numeric columns" are mutually inconsistent and neither is correct. Resolved arithmetic (profile P0):

| Block | Columns |
|---|---|
| Numeric logical features (35 total − F14 − F15) | **33** |
| F14 `question_type` one-hot | **8** |
| F15 `instruction_type` one-hot | **10** |
| Missingness indicators | **5** |
| **Engineered subtotal** | **56** |
| Semantic PCA | **48** |
| **`X_final` before correlation filter** | **104** |

**DECISION 11.3.1 — the final width is 104 pre-filter, and the post-filter width is data-dependent and recorded, not predicted.**
*Reason:* prompt §8 requires distinguishing logical features from final columns, and forbids stating a logical count as the numeric dimensionality. The post-correlation-filter count cannot be known before seeing the training split, so asserting any specific number would be fabrication.
*Impact:* `feature_schema.json` records the realised column list, order, dtypes, and a schema hash. Inference **rejects** a schema-hash mismatch rather than reordering or padding (SPEC draft §7 item 10). Per-profile pre-filter widths: P0 = 104, P1 = 53, P2 = 99, P3 = 101, P4 ≈ 20.

`X_engineered` (56 columns) is retained as a separate matrix for ablation and interpretability, per SPEC draft §7 item 9.

### 11.4 Persistence

Feature matrices are written as **Parquet** (SPEC chat §8), partitioned by split, keyed by `question_id`. Raw corpus stays **JSONL**. CSV is import/export convenience only and is never canonical.

---

## 12. Model Design and Training

### 12.1 Task

Three-class single-label classification. Classes are ordinal **for analysis only**; the learner is a plain probabilistic multiclass classifier.

**SPEC (chat §6):** the baseline must **not** be described as a true ordinal classifier unless an ordinal-specific method is actually implemented. None is implemented here. Ordinal-specific models are a future extension (draft §17).

### 12.2 Models

| Run | Learner | Input matrix | Purpose |
|---|---|---|---|
| Baseline 1 | Multinomial Logistic Regression | `X_engineered` (scaled within its own pipeline) | interpretable sanity check |
| Baseline 2 | Random Forest | `X_engineered` | nonlinear tabular baseline |
| **Main** | **XGBoost** multiclass | `X_final` | production candidate |
| Future extension | Transformer fine-tuning | raw text | **not implemented in the current codebase** |

**SPEC (draft §15 quality gates):** Baseline 1 must train successfully before main-model training is attempted. A failing baseline indicates a pipeline fault, and proceeding would hide it.

**DECISION 12.2.1 — Baseline 2 is Random Forest, not "Random Forest / HistGradientBoosting".** Prompt §11 forbids the disjunction.
*Reason:* RF is the more distinct comparator — HistGradientBoosting is algorithmically close to the XGBoost main model, so it would be a weaker contrast. SVM (chat §15 "optional") is omitted: it scales poorly past ~10⁴ rows with an RBF kernel and adds no distinct information beyond the two retained baselines.
*Impact:* three mandatory runs are implemented; transformer fine-tuning is documented only as a future extension. Fewer, more informative comparisons.

### 12.3 Hyperparameters

**XGBoost fixed configuration** (DECISION 12.3.1 — engineering defaults chosen to be conservative and reproducible, not tuned claims):

```
objective            = multi:softprob
num_class            = 3
eval_metric          = mlogloss
tree_method          = hist
n_estimators         = 2000        (upper bound; early stopping governs)
early_stopping_rounds = 50         (monitored on validation mlogloss)
learning_rate        = 0.05
max_depth            = 6
min_child_weight     = 1
subsample            = 0.8
colsample_bytree     = 0.8
reg_lambda           = 1.0
random_state         = 42
n_jobs               = 1
```

**Bounded search** (SPEC draft §12 "bounded hyperparameter search"): grid over `max_depth ∈ {4, 6, 8}` × `learning_rate ∈ {0.03, 0.05, 0.1}` × `subsample ∈ {0.8, 1.0}` = 18 configurations, each early-stopped on validation. Enumerated in fixed sorted order so the search is reproducible. **Selection uses validation only** (§12.5).

Baseline 1: `multi_class='multinomial'`, `solver='lbfgs'`, `C=1.0`, `max_iter=2000`, `random_state=42`, standardised inputs.
Baseline 2: `n_estimators=500`, `max_depth=None`, `min_samples_leaf=2`, `n_jobs=1`, `random_state=42`.

### 12.4 Class imbalance

**DECISION 12.4.1 — weights apply only when imbalance is material, with "material" defined numerically.** The draft said "if class imbalance is material" without a threshold.
*Reason:* prompt §11. The corpus targets ~1/3 per class (data spec §11), so weighting may be unnecessary; applying it unconditionally would distort a balanced problem.
*Impact:* compute `ratio = max(class_count) / min(class_count)` on the **training split**. If `ratio > 1.5`, apply `sample_weight[i] = N_train / (3 × count(class(i)))`. Otherwise no weighting. The decision and the ratio are logged in `run_manifest.json`.

Macro and per-class metrics are always reported so a dominant class cannot mask minority-class failure (SPEC draft §8.3).

### 12.5 Model selection — fully deterministic

**SPEC (chat §15):** select best validation **macro F1**; tie-break on validation **ordinal MAE**; evaluate the chosen model **once** on the untouched test set.

Complete ordering, applied in sequence until one discriminates:

1. Highest validation macro F1
2. Lowest validation ordinal MAE
3. Highest validation balanced accuracy
4. Lexicographically smallest `run_id`

**DECISION 12.5.1 — criteria 3 and 4 are added.** The supplied two-level rule can still tie on floating-point equality.
*Reason:* prompt §12 requires a tie-breaking rule that fully determines the outcome. Without a terminal criterion, selection is non-deterministic under ties.
*Impact:* selection is now total. Criterion 4 guarantees termination.

### 12.6 Calibration

**DECISION 12.6.1 — probability calibration is OFF by default and is an explicit opt-in stage.** The draft is ambiguous: the header advertises "calibrated probabilities", the architecture shows a calibration layer, `calibration.py` exists, yet §10 says raw probabilities may be exposed if calibration "is not yet trustworthy".
*Reason:* calibration needs held-out data that is not the test set. Fitting it on the validation split consumes the same data used for early stopping and model selection, which risks optimistic calibration estimates. It is not required by any stated objective — the deliverable is a class label plus probabilities, and no downstream consumer with a stated decision threshold exists.
*Impact:* default output is raw `predict_proba`, and the output contract labels it **`model confidence`, not calibrated probability** (§16.2). When `training.calibrate = true`, fit `CalibratedClassifierCV(method='isotonic', cv='prefit')` on the validation split, record `calibrated: true` in the manifest, and report Brier score and a reliability curve. The test set is never used for calibration.

### 12.7 Test-set policy

**SPEC (data spec §14, draft §9, prompt §16).** The test set is read **exactly once**, after all model, hyperparameter, and research-selection decisions that affect the production run are final. Candidate models, hyperparameter searches, ablations, and dimension sweeps use **train + validation only**. The final selected production configuration is evaluated once on test.

Enforcement: the training/evaluation entry point writes a `test_evaluated` marker into the run directory; a second attempt to evaluate the same run on test fails unless explicitly overridden with a flag that is recorded in the manifest.

**DECISION 12.7.1 — test data is excluded from all candidate-run comparisons.**
*Reason:* the previous text allowed per-run test reporting while also declaring test single-use. Those requirements conflict.
*Impact:* train/validation metrics are available for candidate runs; test metrics exist only for the final selected production configuration. This preserves a genuinely untouched final test estimate.

---

## 13. Evaluation

### 13.1 Metrics

| Metric | Role |
|---|---|
| **Macro F1** | **primary** — model selection (SPEC chat §15) |
| **Ordinal MAE** | **primary tie-break** — mean `|y_true − y_pred|` under Easy=0/Moderate=1/Hard=2 |
| Balanced accuracy | secondary; second tie-break |
| Accuracy | secondary headline |
| Per-class precision / recall / F1 | diagnostic (SPEC chat §16) |
| Confusion matrix (3×3, counts + row-normalised) | error-structure diagnosis |
| Quadratic-weighted Cohen's κ | optional, reports ordering-aware agreement |
| Brier score + reliability curve | only when `calibrate = true` |

**SPEC (chat §16):** ordinal MAE is an **analysis** metric. It does not change the training objective, which remains multiclass cross-entropy. Confusing Easy with Hard costs 2 while Easy with Moderate costs 1 — this asymmetry is what MAE captures and accuracy hides.

**SPEC (draft §9):** do not optimise a single metric. Reports always present macro F1, ordinal MAE, balanced accuracy, accuracy, and the confusion matrix together.

### 13.2 Protocol

- Validation: used for early stopping, hyperparameter search, model selection, and optional calibration.
- Candidate runs (baselines, hyperparameter trials, ablations, PCA-dimension sweeps): train and validation metrics only. **They never read the test set.**
- Final selected production configuration: evaluated on the test set exactly once (§12.7).
- Train metrics may be reported for diagnostics; they never drive selection.
- `metrics.json` holds metrics for every candidate run on train/validation and the final production run on train/validation/test; `report.md` is the human-readable summary.
- **Cross-profile comparison is refused.** If two runs have different `dependency_profile` or `feature_schema_version`, the reporter marks the comparison invalid — their feature spaces differ, so the metric difference is not attributable to the model.

---

## 14. Experiments and Ablations

**SPEC (chat §4, §17; draft §16):** quantifying feature-family contribution is a primary objective, not an optional extra.

The source documents specify a cumulative ladder and standalone family runs. These answer **different** questions and both are retained. Transformer fine-tuning (A9) is retained only as a documented future extension and is **not part of the current codebase** because its fine-tuning protocol and additional runtime requirements are not fully specified by this TDD.

### 14.1 Cumulative ladder — *marginal* contribution

| Run | Features | Question |
|---|---|---|
| A0 | majority-class predictor | floor / sanity |
| A1 | lexical + readability | surface complexity |
| A2 | A1 + syntax | syntactic contribution |
| A3 | A2 + task/cognitive | cognitive-structure contribution |
| A4 | A3 + MCQ | option-structure contribution |
| A5 | A4 + TF-IDF | lexical-importance contribution |
| A6 | A5 + semantic scalars | conceptual contribution |
| A7 | A6 + programming | programming contribution |
| A8 | A7 + semantic embedding | **full hybrid production model** |
| A8b | A8 with PCA ∈ {32, 48, 64} | dimension sweep (DECISION 10.2.1) |
| A9 | Transformer fine-tuning | future extension; **not implemented in the current codebase** |

### 14.2 Single-family runs — *standalone* power

One run per family in §9.10, each family alone. Answers "how much does this family know by itself", which the ladder cannot reveal because earlier families mask later ones.

### 14.3 Protocol

- Identical splits and seeds across all runs.
- Baseline models use their fixed hyperparameters from §12.3.
- For XGBoost ablations, the **selected main-model hyperparameters are frozen once from the primary A8/full-feature validation search** and reused unchanged across all cumulative and single-family XGBoost runs. This makes the ablation comparison isolate the feature subset rather than a second tuning process.
- No ablation run performs an independent hyperparameter search unless a separate research experiment is explicitly added to the configuration.
- Candidate runs use train + validation metrics only. The test set is never read by an ablation.
- A8b (PCA 32/48/64) is a validation-only dimensionality study. It does not access test data and does not silently replace the production default of 48 components. A different PCA dimension may become the production setting only through an explicit prior configuration change followed by a fresh final model-selection run.
- A9 is documented for future work only and is excluded from the current ablation runner.
- Results table plus per-family delta, written to `ablation_results.json` and `report.md`.

**DECISION 14.3.1 — freeze selected main XGBoost hyperparameters for ablations.**
*Reason:* the earlier wording “only the feature subset varies” was incompatible with an independent hyperparameter search for every ablation. Freezing the selected configuration isolates the research variable of interest and avoids multiplying validation searches.
*Impact:* ablations are directly comparable and substantially cheaper; the selected XGBoost configuration itself remains a validation-derived result.

### 14.4 Interpretation constraints — binding

**SPEC (draft §16, chat §18):**

1. Ablation measures **predictive contribution within this model and pipeline**. It does not establish that a feature family causes human difficulty.
2. A performance drop on removal does not make a family "intrinsically causal".
3. Results are conditional on this corpus, this encoder, this parser, and this lexicon set. Lexicon coverage (UNRESOLVED-3) bounds F26/F28/F30 and therefore bounds any claim about the concept and programming families.
4. No run may be reported as state-of-the-art; no comparison to external published numbers is valid, because the corpus differs.

---

## 15. Interpretability

**SPEC (chat §18, draft §17; prompt §17).**

| Method | Scope | Purpose |
|---|---|---|
| XGBoost gain importance | global | fast global view |
| Permutation importance (validation) | global, model-agnostic | robust to gain's cardinality bias |
| SHAP | global + per-question | detailed attribution, where computationally practical |

Requirements:

- Importances map back to **human-readable feature names** from `feature_schema.json`, including expanded one-hot names (`question_type=Why`) and PCA components (`sem_pca_017`).
- PCA components are **not** individually interpretable. Reports must state this and present the semantic block's aggregate contribution rather than implying `sem_pca_017` has meaning.
- **SPEC:** every interpretability output carries the statement that importance indicates the model relied on a feature and **does not prove** the feature causes human difficulty.
- Permutation importance uses `n_repeats = 10`, `random_state = 42`.
- SHAP is computed on a seeded sample of `interpret.shap_sample_size` (default 2000) validation rows when the full set is impractical; the sample size is reported alongside.
- Outputs: `feature_importance.json`, `permutation_importance.json`, `shap_summary.json`.

---

## 16. Inference

### 16.1 Path

**SPEC (chat §19, draft §10):** the caller passes the same raw JSON contract used in training; the service computes everything.

```
{"question_id": "...", "question": "..."}
    → §8 preprocessing (identical code path as training)
    → §9 extraction, using loaded fitted artifacts
    → §10 encode + loaded PCA transform
    → assemble columns in persisted order; verify schema hash
    → model.predict_proba
    → argmax → label
```

Requirements:
- **Identical code path.** Training and inference call the same extraction functions. No re-implementation.
- All fitted artifacts are **loaded, never refitted**. Any `fit` call at inference is a defect (test T-INF-03).
- Schema-hash mismatch → hard failure naming both hashes. Never reorder, pad, or drop columns.
- Batch and single-record inference must produce identical output for the same record (SPEC draft §15; test T-INF-01).

### 16.2 Output contract

```json
{
  "question_id": "NEW-001",
  "predicted_label": "Hard",
  "probabilities": { "Easy": 0.08, "Moderate": 0.27, "Hard": 0.65 },
  "model_confidence": 0.65,
  "calibrated": false,
  "model_version": "qdp_xgb_v1",
  "feature_schema_version": "fs_v1",
  "dependency_profile": "P0"
}
```

**DECISION 16.2.1 — the field is `model_confidence`, not `confidence`, and `calibrated` is always present.**
*Reason:* the draft (§10) warns that uncalibrated probability must be labelled as model confidence rather than certainty. Naming the field `confidence` invites exactly the misreading the draft warns against.
*Impact:* consumers cannot mistake an uncalibrated softmax output for a calibrated probability. `probabilities` always sums to 1.0 within floating-point tolerance. Ties in argmax resolve to the **lowest ordinal class** (Easy < Moderate < Hard), deterministically.

No continuous difficulty score is emitted (SPEC draft §2).

---

## 17. Artifacts and Reproducibility

### 17.1 Required artifacts

| Artifact | Contents |
|---|---|
| `feature_schema.json` | column names, order, dtypes, missing sentinels, extractor versions, family map, **schema hash** |
| `rare_word_df.joblib` | training document-frequency table (F05) |
| `tfidf.joblib` | fitted vocabulary + IDF |
| `tfidf_threshold.json` | fitted `τ_tfidf` (F25) |
| `imputer.joblib` | train-median numeric imputer |
| `onehot.joblib` | fitted encoder, closed category vocabularies |
| `feature_selector.joblib` | variance + correlation mask, retained order |
| `semantic_encoder.json` | encoder name, revision, dim, max_seq_length |
| `semantic_pca.joblib` | fitted PCA |
| `model.joblib` | selected classifier |
| `calibrator.joblib` | only when `calibrate = true` |
| `metrics.json` | candidate train/validation metrics plus final selected production train/validation/test metrics and confusion matrices |
| `ablation_results.json` | §14 results |
| `run_manifest.json` | see §17.2 |

### 17.2 `run_manifest.json`

**SPEC (draft §11, §14; chat §20):**

- dataset fingerprint: `sha256` over sorted `raw_hash` values, plus record count
- **actual record count** (DECISION 5.2.1)
- explicit `train` / `validation` / `test` `question_id` lists (or a content-addressed reference)
- all random seeds, individually named (§17.3)
- full resolved configuration, post-defaults
- pinned package versions of every dependency in §3.2
- `dependency_profile` (P0–P4)
- `feature_schema_version` and schema hash
- selected hyperparameters, early-stopping round, class-weight decision + ratio
- `calibrated` flag
- `test_evaluated` marker and, once consumed, the final production test metric record
- timestamp, git commit if available

### 17.3 Seeds — enumerated

| Seed | Governs | Default |
|---|---|---|
| `seed.split` | group assignment | 42 |
| `seed.minhash` | MinHash permutations | 42 |
| `seed.pca` | PCA solver | 42 |
| `seed.model` | learner | 42 |
| `seed.cv` | K-fold fallback | 42 |
| `seed.permutation_importance` | permutation repeats | 42 |
| `seed.shap_sample` | SHAP row sample | 42 |

Each is separately configurable so one stage can be varied without disturbing the others — a single global seed would make "same split, different model seed" experiments impossible.

### 17.4 Determinism

Given identical corpus, configuration, artifact versions, dependency profile, and runtime device class, the pipeline must produce **numerically reproducible** feature matrices and predictions within documented tolerances. No wall-clock, hostname, PID, or unordered-iteration dependence. No network call in the core path (§21).

For the reproducibility profile, model parallelism is fixed to `n_jobs = 1`; parser processing is single-process by default; dictionary/set traversal is explicitly ordered.

Acceptance tolerances are:
- engineered numeric features: maximum absolute difference `<= 1e-7`;
- semantic embeddings/PCA outputs: maximum absolute difference `<= 1e-5`;
- class probabilities: maximum absolute difference `<= 1e-5`;
- predicted class must be identical.

**DECISION 17.4.1 — replace “bit-identical” with numerical reproducibility within explicit tolerances.**
*Reason:* strict bit identity is not guaranteed across BLAS, tokenizer, transformer-runtime, and hardware implementations even when seeds are fixed. Requiring it would make a correct research pipeline fail for irrelevant floating-point differences.
*Impact:* reproducibility remains strict enough for research comparison while being technically achievable across the supported single-machine environments.

---

## 18. Configuration

**SPEC (prompt §21):** fix what must not vary; expose what genuinely varies. Not every constant becomes configuration.

### 18.1 Fixed by specification (not freely configurable)

The three-class contract; the 35-feature identity and formulas; exact-dedup normalisation; the fit-on-train-only rule; test-set single-use; primary metric and tie-break order; the closed F14/F15 vocabularies.

The source-label mapping table is stored in versioned `label_normalization.yaml` so the dataset-building process is auditable, but its semantics are **specification-governed**: changing a mapping is a dataset-contract change that requires a new dataset/feature schema version and an entry in `DECISION.md`. It is not an implementation-time free choice.

### 18.2 Configurable

```yaml
corpus:
  min_records: 10000
  path: data/raw/questions.jsonl
  provenance_path: null          # optional sidecar; splitter-only (DECISION 5.4.1)

ingest:
  max_reject_ratio: 0.02

validation:
  min_class_count: 30

dedup:
  near_threshold: 0.85
  minhash_permutations: 128
  lsh_bands: 32

split:
  ratios: [0.70, 0.15, 0.15]
  max_ratio_deviation: 0.02
  small_dataset_threshold: 5000

preprocess:
  max_chars: 20000

features:
  use_lexical: true
  use_readability: true
  use_syntax: true
  use_task: true
  use_mcq: true
  use_tfidf: true
  use_semantic_scalar: true
  use_programming: true
  use_semantic_embedding: true
  rare_word_df_threshold: 5
  tfidf_high_percentile: 90
  max_sentences_for_f29: 40
  recursion_weight: 2
  indent_width: 4

semantic:
  encoder: sentence-transformers/all-MiniLM-L6-v2
  pca_components: 48
  min_explained_variance: 0.60

selection:
  correlation_threshold: 0.90

training:
  classifier: xgboost
  calibrate: false
  hyperparameter_search: true
  ablation_hyperparameters: fixed_selected_main

interpret:
  shap_sample_size: 2000

seed: { split: 42, minhash: 42, pca: 42, model: 42, cv: 42,
        permutation_importance: 42, shap_sample: 42 }

runtime:
  dependency_profile: P0
  spacy_n_process: 1
  embedding_batch_size: 64
```

### 18.3 Lexicon files — versioned configuration, not code

`label_normalization.yaml`, `task_lexicon.yaml` (F15), `bloom_lexicon.yaml` (F16), `concept_lexicon.yaml` (F26/F28), `algorithm_complexity.yaml` (F30), `constraint_patterns.yaml` (F31/F32), `negation_lexicon.yaml` (F13), `code_signatures.yaml` (§8.6), `mcq_markers.yaml` (§8.7).

**SPEC (chat §11, §22, §24):** these define feature semantics and must be data, not hard-coded literals. A lexicon change alters feature values and therefore requires a new `feature_schema_version`.

---

## 19. Error Handling and Failure Behaviour

**SPEC (prompt §18):** "handle errors gracefully" is not a specification. Every stage declares skip / fallback / fail.

| Stage | Condition | Behaviour |
|---|---|---|
| Ingest | invalid UTF-8, malformed JSON, missing question, unknown label | **skip record**, log to `rejected_records.jsonl` with reason |
| Ingest | reject ratio > threshold | **fail run** |
| Validate | missing class, class below minimum, duplicate ID post-dedup | **fail run** |
| Dedup | label-conflicting duplicate group | **skip group** (DECISION 7.2.1), log |
| Split | ratio deviation > threshold | **fail run** |
| Preprocess | text > `max_chars` | **skip record**, reason `oversized` |
| Parse | spaCy failure/timeout on a record | **fallback**: `parse_ok = 0`, F09–F13 = `NaN`, continue |
| MCQ parse | no valid option set | **not an error**: `mcq_detected = 0` |
| Code parse | `ast.parse` fails | **fallback** to lexical path (DECISION 9.8.2), `code_parse_exact = 0` |
| Readability | `F01 < 3` | **fallback**: F06–F08 = `NaN` |
| Embedding | encoder load failure | **fail run** with the named profile it would degrade to (§3.3) |
| Embedding | single-record failure | **fail run** — a silent zero vector would corrupt PCA and similarity features |
| PCA | components > train rows or train features | **fail run** with a diagnostic |
| Train | baseline 1 fails | **fail run** before main training (SPEC draft §15) |
| Train | all-NaN or inf in `X_final` | **fail run** (SPEC draft §15 gate) |
| Select | all candidates fail | **fail run** |
| Test | second evaluation attempt | **fail** unless explicitly overridden and recorded (§12.7) |
| Inference | schema-hash mismatch | **fail** with both hashes |
| Inference | missing artifact | **fail** naming the artifact |
| Artifact I/O | write failure | **fail run** — a partial artifact bundle is worse than none |

**Logging.** Structured, one record per event: stage, `question_id` where applicable, reason code, severity. Reason codes are a closed enumerated set so rejection reports are aggregatable. Per-record skips are counted and summarised per stage; a stage whose skip rate exceeds its configured tolerance escalates to a run failure rather than silently producing a degraded dataset.

---

## 20. Performance and Scalability

Scale: ~10⁵ questions, single machine, CPU sufficient, GPU optional.

### 20.1 Cost ranking

Most to least expensive: **sentence embedding** → **dependency parsing** → TF-IDF fit → PCA fit → model training → scalar feature arithmetic.

These are **HYPOTHESIS**-level estimates from the known algorithmic cost of each stage, not measured benchmarks. No performance claim in this document is empirical; §23 requires an actual smoke test to establish real numbers.

### 20.2 Caching — required

| Cache | Key | Reason |
|---|---|---|
| Parsed docs | `question_id` + spaCy model version | parsing dominates repeated extraction |
| Embeddings | `question_id` + `text_role` + `text_hash` + encoder name/revision + truncation length | the same question produces multiple units (question/stem/option/sentence); role + content hash prevents collisions and allows ablations to reuse embeddings |
| Code AST results | `question_id` + code-span hash | repeated experiment runs |

**SPEC (draft §12):** extraction must not re-run for every hyperparameter trial. The fused feature table is built **once** per feature configuration; the 18-point search reads it repeatedly.

Cache keys include artifact versions so a dependency upgrade cannot silently serve stale vectors. For embeddings, the key is `(question_id, text_role, text_hash, encoder_revision, truncation_length)`.

**DECISION 10.1.2 — embedding cache keys include text role and text hash.**
*Reason:* F21/F22/F29 reuse the encoder over stems, options, and sentences in addition to whole-question embeddings. A question-only key can overwrite one unit with another.
*Impact:* embeddings are addressed by `(question_id, text_role, text_hash, encoder_revision, truncation_length)`, making the cache collision-safe for the specified text views.

### 20.3 Batching and memory

- spaCy via `nlp.pipe` with batching; `n_process` from config, results reassembled in input order (§8.4).
- Embeddings batched at `embedding_batch_size` (default 64); GPU used when available, CPU fallback supported. Batch size must not alter results beyond floating-point tolerance — a test asserts this (T-PERF-02).
- Feature matrix at 10⁵ × 104 float64 ≈ 83 MB — comfortably in memory. Embeddings at 10⁵ × 384 float32 ≈ 154 MB, cached to disk rather than held.
- Parquet columnar reads let ablations load only needed columns.
- Streaming ingestion; the full raw corpus is never required in memory at once.

### 20.4 Budgets

These are **resource-design targets**, not fixed empirical service-level objectives. Actual timings and peak memory must be measured by T-PERF-01 on the target machine. Optional hardware-specific limits may be supplied through configuration; when no limit is configured, the benchmark remains informational and is not a correctness gate.

| Stage | Target |
|---|---|
| Validation + dedup | small fraction of total; LSH keeps it ~O(N) |
| Scalar features | vectorised, CPU-bound |
| Parsing | batched, cached, one pass |
| Embedding | batched, cached, one pass |
| Training | early-stopped, bounded 18-point search, reuses the fused table |
| Inference | single record, artifacts loaded once, one feature pass, no refit |

---

## 21. Security and Data Integrity

Scoped to this system's real surface (prompt §25 — no generic enterprise-security content).

| Concern | Control |
|---|---|
| **Untrusted code parsing** | Question text contains arbitrary code. `ast.parse` **only** — it builds a tree without executing. `eval`, `exec`, `compile` in executable mode, and subprocess invocation are **forbidden** in the code-analysis path. |
| **Unsafe deserialisation** | `joblib`/pickle artifacts execute arbitrary code on load. Load **only** artifacts produced by this pipeline from a trusted local path. Never load an artifact from an untrusted or network source. Each artifact carries a checksum recorded in the manifest and verified on load. |
| **Input validation** | Enforced at ingest (§6.2): encoding, structure, label domain, size cap (`max_chars`) to bound resource use on pathological input. |
| **Regex safety** | MCQ, constraint, and code-signature patterns must avoid nested unbounded quantifiers (catastrophic backtracking). Patterns are anchored and bounded; a per-record regex timeout guard applies where the runtime supports it. |
| **Path handling** | All paths resolved against a configured root; `..` traversal rejected. Artifact writes are atomic (temp file + rename) so an interrupted run cannot leave a half-written artifact that later loads as valid. |
| **Licence compliance** | Corpus-level, upstream (SPEC data spec §10, §19.8). Records violating source usage conditions are excluded; `license_basis` is retained per record in `provenance.jsonl`. The model pipeline does not re-verify licences but must not strip provenance. |
| **Source reproducibility** | `provenance.jsonl` retains `source_reference` so any retained record is traceable to its origin (SPEC data spec §5). |
| **Network isolation** | The core training and inference paths make **no network calls**. Pretrained models must be pre-fetched to a local cache; runtime download is forbidden, both for determinism and per the project's offline constraint (§3.1). |
| **No PII expected** | Corpus is academic/assessment questions. If a source is later found to contain personal data, handling is an upstream corpus concern, not addressed here. |

---

## 22. Repository Layout

**DECISION 22.1 — the `src/` layout from chat §23 is adopted, with the module breakdown from draft §13, and artifacts moved out of the package.** The two documents specify different trees.
*Reason:* chat §23 is labelled "desired final repository shape" and is the more standard packaging. The draft placed `artifacts/` **inside** the package directory, which mixes generated run output with source code and breaks clean packaging and versioning.
*Impact:* generated artifacts live in a top-level per-run directory. Every module below exists because §s above assign it a responsibility (prompt §20 — no generic template).

```
project/
├── config/
│   ├── default.yaml
│   ├── label_normalization.yaml
│   ├── task_lexicon.yaml
│   ├── bloom_lexicon.yaml
│   ├── concept_lexicon.yaml
│   ├── algorithm_complexity.yaml
│   ├── constraint_patterns.yaml
│   ├── negation_lexicon.yaml
│   ├── code_signatures.yaml
│   └── mcq_markers.yaml
├── data/
│   ├── raw/                     # corpus JSONL (+ optional provenance sidecar)
│   ├── processed/               # canonical records with split assignment
│   └── features/                # Parquet matrices, embedding cache
├── artifacts/
│   └── <run_id>/                # §17.1 bundle, one directory per run
├── src/qdp/
│   ├── config/          loader, schema, validation of config itself
│   ├── data/            schema, loader, validator, deduplicator, splitter
│   ├── preprocessing/   normalize, views, tokenizer, mcq_parser, code_detector
│   ├── features/        lexical, readability, syntax, task, mcq, tfidf,
│   │                    semantic_scalar, programming, registry, pipeline
│   ├── semantic/        encoder, pca
│   ├── transform/       imputer, encoder, correlation_filter, assembler
│   ├── models/          baselines, xgb, calibration, predictor
│   ├── evaluation/      metrics, confusion, ablation, feature_importance
│   ├── artifacts/       save/load, checksums, schema hashing
│   └── utils/           logging, seeding, hashing, caching
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── run_ablations.py
├── tests/               §23
└── README.md
```

`features/registry.py` owns the single source of truth for the 35-feature identity, family map, and schema-hash computation — so a feature cannot be added in one place and missed in another.

---

## 23. Testing Specification

**SPEC (draft §15, chat §21).** Every test below has a pass/fail condition, not a description.

### 23.1 Schema and ingestion

| ID | Assertion |
|---|---|
| T-SCHEMA-01 | Valid JSONL accepted; counts match input lines minus rejections |
| T-SCHEMA-02 | Missing/empty `question`, unknown label, malformed JSON each rejected with the correct reason code |
| T-SCHEMA-03 | A JSON array and the equivalent JSONL yield byte-identical canonical records |
| T-SCHEMA-04 | Absent `question_id` → deterministic content hash; identical text → identical id |
| T-SCHEMA-05 | Label case/whitespace normalised; `Medium-Hard` maps per DECISION 5.3.1; unmapped value rejected, never guessed |
| T-SCHEMA-06 | Reject ratio above threshold fails the run |

### 23.2 Features

| ID | Assertion |
|---|---|
| T-FEAT-01 | Hand-computed fixtures match exactly for F01–F05 |
| T-FEAT-02 | F12 equals hand-computed MDD on a fixed 2-sentence fixture |
| T-FEAT-03 | F14/F15 produce only in-vocabulary values across a fixture covering every class plus unmatched input → `Other` |
| T-FEAT-04 | F16 returns the **maximum** Bloom level on a multi-verb fixture |
| T-FEAT-05 | F18–F22 correct on a 4-option MCQ; F19–F22 = `NaN` and F18 = 0 on prose |
| T-FEAT-06 | **No feature function accepts or references `label`** (signature + source inspection) |
| T-FEAT-07 | F34/F35 match hand-computed values for a nested-loop Python fixture via `ast`; lexical fallback fires on a Java fixture with `code_parse_exact = 0` |
| T-FEAT-08 | F31 parses `10^6`, `1e6`, `1,000,000` identically; `NaN` + `input_size_detected = 0` when absent |
| T-FEAT-09 | F29 = `NaN` with `multi_segment = 0` on a single-sentence question — **not** 1.0 |
| T-FEAT-10 | Unicode, emoji, heavy punctuation, 10k-char, and 1-word inputs do not raise |
| T-FEAT-11 | Views are correctly allocated: code excluded from `nl_view`, options retained, math masked |

### 23.3 Missingness

| ID | Assertion |
|---|---|
| T-MISS-01 | Non-MCQ question completes extraction without exception |
| T-MISS-02 | Non-code question completes extraction without exception |
| T-MISS-03 | Every indicator matches its governed features' missingness exactly |
| T-MISS-04 | Imputation fills only `NaN`, leaves observed values untouched |

### 23.4 Leakage — highest priority

| ID | Assertion |
|---|---|
| T-LEAK-01 | TF-IDF, rare-word table, `τ_tfidf`, imputer, one-hot, correlation mask are each fitted on train indices only (assert on the index set passed to every `fit`) |
| T-LEAK-02 | PCA `fit` receives only train rows |
| T-LEAK-03 | No `question_id` appears in more than one split; no family spans splits |
| T-LEAK-04 | The feature pipeline's interface **cannot** accept provenance fields (DECISION 5.4.1) — passing them raises |
| T-LEAK-05 | Exact and near-duplicate groups are wholly within one split |
| T-LEAK-06 | Corrupting test labels leaves train-fitted artifacts byte-identical |
| T-LEAK-07 | Test set is not read during training or selection (I/O access assertion) |

### 23.5 Model and artifacts

| ID | Assertion |
|---|---|
| T-MODEL-01 | Baseline 1 trains before main model; baseline failure aborts the run |
| T-MODEL-02 | `X_final` contains no NaN/inf at train time |
| T-MODEL-03 | Class-weight decision fires iff ratio > 1.5, with correct weights |
| T-MODEL-04 | Selection follows the full 4-level tie-break; a constructed 3-way tie resolves deterministically |
| T-MODEL-05 | Artifact save/load round-trip reproduces identical predictions |
| T-MODEL-06 | Schema-hash mismatch fails loudly, naming both hashes |
| T-MODEL-07 | Second test evaluation blocked without explicit override |

### 23.6 Inference and determinism

| ID | Assertion |
|---|---|
| T-INF-01 | Single-record prediction == batch prediction for the same record |
| T-INF-02 | Same record + same artifacts → feature values and probabilities remain within the §17.4 tolerances across two processes, with identical predicted class |
| T-INF-03 | **No `fit` is called anywhere in the inference path** |
| T-INF-04 | Output contract validates: keys present, probabilities sum to 1.0 ± 1e-6, `calibrated` present, argmax tie → lowest ordinal class |
| T-INF-05 | Missing artifact fails with the artifact named |

### 23.7 End-to-end and performance

| ID | Assertion |
|---|---|
| T-E2E-01 | Small fixture corpus → train → evaluate → predict completes; report contains macro F1, ordinal MAE, balanced accuracy, accuracy, confusion matrix |
| T-E2E-02 | Ablation runner produces one result row per configured run with identical splits |
| T-PERF-01 | Representative-subset extraction and training execute successfully and **record actual timings and peak memory**. This is a benchmark/measurement test, not a correctness gate unless explicit hardware-specific budgets are supplied in configuration. |
| T-PERF-02 | Changing `embedding_batch_size` does not change results beyond 1e-5 |
| T-PERF-03 | Cache hit produces identical vectors to a cold run |

### 23.8 Quality gates

**SPEC (draft §15).** A run is invalid unless: no unknown-label samples survive validation; no `question_id` crosses splits; no NaN/inf in the final matrix; all three classes present in train and evaluation splits; baseline trains before main; the final report includes accuracy, macro F1, balanced accuracy, confusion matrix, and ordinal MAE.

---

## 24. Decision Log

Consolidated index of every engineering resolution. SPEC items are not listed — only decisions made in this document.

| # | Decision | Reason | Impact |
|---|---|---|---|
| 3.2.1 | Remove `tree-sitter` | heavyweight, non-native, unnecessary | stdlib `ast` + lexical fallback; no feature lost |
| 3.2.2 | No near-dup library | dependency constraint | MinHash specified mathematically |
| 5.2.1 | Authenticity outranks the 100,000 count | data spec §19 is non-negotiable; §19.10 contemplates shortfall | count reported, not asserted; `min_records` gate |
| 5.3.1 | `Medium-Hard` → `Hard` | a documented deterministic rule is mandatory | config-driven, auditable |
| 5.4.1 | Provenance may reach the splitter, never the features | splitting ≠ modelling | reconciles data spec §13 with §14 |
| 6.1.1 | `question_id` optional in, required internally | data spec is authoritative on the model-facing contract | content hash when absent |
| 7.2.1 | Discard unresolved label conflicts | only option with no label invention | small loss, reported |
| 7.3.1 | Group split replaces per-question stratified split | the two draft requirements are incompatible | stratification approximate, bounded, measured |
| 7.3.2 | “Small dataset” = `N < 5000` | remove undefined term | K-fold fallback path defined |
| 7.3.3 | Use global `family_id` as split group key | source-scoping could split one near-duplicate family across partitions | provenance never changes family integrity |
| 8.3.1 | Parser fixed: spaCy `en_core_web_sm`, dependency only | reproducibility; chat §22 | constituency parsing removed |
| 8.5.1 | Explicit asymmetric truncation | silent truncation is invisible in metrics | engineered = full text; semantic = 256 pieces + flag |
| 9.3.1 | F12 = Mean Dependency Distance | one formula required | parameter-free, exact |
| 9.3.2 | F11 = dependency depth | no constituency parser | semantics restated in schema |
| 9.3.3 | F13 counts the union of negation signals | prevents double-counting one token | deterministic negation count |
| 9.4.1 | F14/F15 vocabularies closed (8, 10) | open sets break one-hot reproducibility | fixed matrix width |
| 9.4.2 | F16 = deterministic Bloom lexicon, max level | chat §22 forbids LLM variance | single ordinal column; flagged HYPOTHESIS |
| 9.5.1 | F21/F22 redefined answer-key-independent | draft features need an answer key that the contract forbids | design intent preserved without answer labels |
| 9.5.2 | F20 = population variance, `NaN` at n<2 | avoid undefined → silent zero | explicit missingness |
| 9.6.1 | τ_tfidf = train 90th percentile | absolute constants are scale-dependent | fitted artifact, train-only |
| 9.7.1 | F26/F28 = fixed lexicon, not keyphrase extraction | determinism across versions | coverage bound made explicit |
| 9.7.2 | F29 = mean pairwise inter-sentence cosine | “aggregate over segments” undefined | exact formula; `NaN` at k<2 |
| 9.8.1 | F30 = configured lexicon max, weights 1–5 | explicit deterministic requirement | reproducible; flagged HYPOTHESIS |
| 9.8.2 | F34/F35 use stdlib `ast` plus lexical fallback | removes Tree-sitter without losing specified features | deterministic Python path; bounded-quality fallback elsewhere |
| 9.8.3 | Lexical nesting fallback fully specified | previous wording was underspecified | exact non-Python F35 and `indent_width` |
| 10.1.1 | Encoder fixed: `all-MiniLM-L6-v2` | open choice breaks reproducibility | 384-dim, 256-piece limit, frozen |
| 10.1.2 | Embedding cache keys include role + text hash | one question produces multiple text units | prevents stem/option/sentence cache collisions |
| 10.2.1 | PCA 48 = default, not an optimum | chat §13 forbids hard-coding as truth | sweep A8b is the validation study |
| 11.2.1 | MI / permutation / SHAP are analysis-only | avoids unstable selection and additional leakage surface | production selection = variance + correlation only |
| 11.3.1 | Final width 104 pre-filter; post-filter recorded | logical feature count ≠ numeric matrix width | schema hash enforced at inference |
| 12.2.1 | Baseline 2 = Random Forest; drop SVM | remove disjunction; RF is the stronger contrast | three mandatory model families |
| 12.3.1 | XGBoost defaults + 18-point bounded grid | remove invented/ambiguous hyperparameters | fixed, enumerable, validation-only |
| 12.4.1 | Class weights iff ratio > 1.5 | define “material” numerically | logged decision |
| 12.5.1 | Four-level selection tie-break | two-level rule can still tie | total deterministic selection |
| 12.6.1 | Calibration OFF by default | draft self-contradictory; held-out calibration not required | raw probabilities labelled `model_confidence` |
| 12.7.1 | Candidate runs never read test; final production run reads it once | prior test-reporting text contradicted single-use policy | clean final estimate |
| 14.3.1 | Freeze selected main XGBoost hyperparameters for ablations | “only feature subset varies” is incompatible with independent tuning | fair, efficient ablations |
| 17.4.1 | Use numerical reproducibility tolerances, not bit identity | strict bit identity is not portable across numerical runtimes | explicit achievable reproducibility contract |
| 20.4.1 | Performance budgets are measurement-based unless explicitly configured as gates | hardware target is unknown | avoids arbitrary runtime failures |
| 22.1 | `src/` repository layout with top-level run artifacts | separates source code from generated outputs | clean packaging and versioning |
| 25.1 | Resolve dependency inventory during implementation preflight | exact installed environment was unavailable during authoring | generator inspects and records the real environment rather than guessing |

## 25. Unresolved Dependencies

**SPEC (prompt §28):** genuine uncertainty is stated, not hidden and not fabricated.

### UNRESOLVED-1 — Installed library inventory *(pre-implementation check)*

The project owner requires that no library outside the existing environment be installed or used, and the installed inventory was not verified during authoring. §3.2 lists the required dependencies; §3.3 specifies degradation profiles P0–P4.

**Required action at implementation start:** inspect the actual environment, record the exact installed versions, and select the applicable profile. The code generator must not add dependencies or silently substitute libraries. The local availability of the frozen `en_core_web_sm` package and the `all-MiniLM-L6-v2` model assets must also be verified because the core path contains no network download step.

**Consequence:** if the environment cannot supply `spacy`, `sentence-transformers`, or `textstat`, or the required frozen model assets are unavailable, the implementation must use the explicitly named degradation profile and report which research components are unavailable. Profile P4 remains a smoke-test floor only and is not sufficient to answer the full research question in §1.2. No hidden substitution is permitted.

### UNRESOLVED-2 — Corpus availability and achievable count

Whether 100,000 questions with **authentic source labels** are obtainable under source and licensing constraints is not established by the supplied context. Data spec §19.10 anticipates this and requires reporting the achievable count.

**Required:** the actual corpus and its `dataset_statistics.json`.
**Consequence:** the realised class balance may deviate from ~1/3 each, which changes the class-weight decision (§12.4) and the reliability of macro metrics. No relabelling is permitted to close a gap.

### UNRESOLVED-3 — Lexicon content and coverage

Six features (F15, F16, F26, F28, F30, and F13's cue list) depend on lexicons that the supplied context requires to exist but does not populate. §18.3 fixes their location, format, and versioning; their **contents** are a domain-expertise input.

**Required:** populated lexicon files, reviewed by someone with domain knowledge of the corpus.
**Consequence:** these features' recall is bounded by lexicon coverage. A sparse lexicon depresses F26/F28/F30 toward zero and would make ablations A3, A6, and A7 understate those families' contribution. §14.4 item 3 requires any such claim to be reported as lexicon-conditional.

### UNRESOLVED-4 — Ground-truth semantics of the labels

The corpus aggregates multiple sources whose difficulty schemes were constructed by different processes (item-response statistics, author judgement, contest tiers). Normalisation (§5.3) makes the *strings* consistent; it cannot make the *underlying constructs* identical.

**Consequence:** `Hard` may not denote the same thing across sources. This limits achievable accuracy for reasons unrelated to feature quality, and is a confound that no modelling choice in this document can remove. Per-source metrics should be reported where provenance is available, so cross-source inconsistency is visible rather than absorbed into the error rate. Do not attribute a metric ceiling to feature weakness without first examining this.

### UNRESOLVED-5 — Performance budgets

§20.4 states targets, not measurements; §20.1 estimates are algorithmic, not empirical. T-PERF-01 exists to establish real figures.

**Consequence:** concrete hardware requirements cannot be committed until that test runs on the actual corpus and machine.

---

## Implementation Summary

```
corpus JSONL / JSON array (question + label, authentic labels only)
  → ingest: decode, parse, normalise label, hash, reject invalid
  → validate: label domain, class coverage, counts          [gate: fail]
  → exact dedup (sha256 of normalised text)
  → near-dup families (MinHash 128 / LSH 32×4 / Jaccard ≥ 0.85, connected components)
  → group-stratified split 70/15/15, seed 42                [nothing fitted before here]
  ─────────────────────── train split only, from here ───────────────────────
  → fit: rare-word df, TF-IDF + IDF, τ_tfidf, imputer, one-hot, correlation mask
  → extract 35 logical features + 5 indicators  (views per §8.2)
  → encode nl_view with frozen MiniLM (384-d), fit PCA → 48 dims
  → assemble X_final: 56 engineered + 48 semantic = 104 columns pre-filter
  → apply variance + correlation filter; persist schema + hash
  ───────────────────────────────────────────────────────────────────────────
  → train Baseline 1 (LogReg)                               [gate: must pass]
  → train Baseline 2 (RF), then XGBoost + 18-point bounded search
  → select on validation: macro F1 → ordinal MAE → balanced acc → run_id
  → optional isotonic calibration (off by default)
  → evaluate the final selected production configuration on test ONCE
  → report candidate train/validation metrics; final test metrics only for the selected production run
  → ablations: cumulative A0–A8b + single-family runs (shared splits/seeds; selected main XGBoost hyperparameters frozen)
  → interpretability: gain, permutation, SHAP — association, not causation
  → persist artifact bundle + run_manifest (fingerprint, split IDs, seeds, versions, profile)
  → inference: same code path, artifacts loaded never refitted, schema hash enforced
```

**End of document.**
