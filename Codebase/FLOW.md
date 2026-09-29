# Execution Flow

## Training

```mermaid
flowchart TD
  A[JSONL / JSON corpus] --> B[UTF-8 ingest + label normalization]
  B --> C[Exact deduplication + conflict rejection]
  C --> D[Validation gates]
  D --> E[MinHash / LSH near-duplicate families]
  E --> F{Corpus size}
  F -->|large| G[70/15/15 group-safe split]
  F -->|small| H[held-out test + repeated group CV]
  G --> I[Fit train-only feature artifacts]
  H --> I2[Fit fold-local artifacts during CV]
  I --> J[35 logical features + indicators]
  I2 --> J2[Fold-local features]
  J --> K[Impute + one-hot + variance/correlation selection]
  J2 --> K2[Fold-local transformation]
  K --> L[Optional frozen semantic encoder + train-only PCA]
  K2 --> L2[Fold-local model selection]
  L --> M[Baseline + XGBoost validation selection]
  L2 --> M2[CV model selection]
  M --> N[Final fit on train+validation]
  M2 --> N2[Final fit on full non-test pool]
  N --> O[Single final test evaluation]
  N2 --> O2[Single final test evaluation]
  O --> P[Artifacts + diagnostics + report + checksums]
  O2 --> P
```

## Inference

```mermaid
flowchart LR
  Q[Raw question] --> R[Canonical record]
  R --> S[Same feature extractor]
  S --> T[Saved imputer / encoder / selector / PCA]
  T --> U[Saved classifier]
  U --> V[Easy / Moderate / Hard + probabilities]
  V --> W[Schema + artifact integrity checks]
```

The implementation reuses the same feature-extraction code for training and inference. Fitted artifacts are loaded rather than refit. Test data is never used by inference.
