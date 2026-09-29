# TDD Test Matrix

The tests in this repository map the TDD's required test IDs to implementation areas. The full TDD checklist remains authoritative; environment- and corpus-dependent tests are exercised in P0 when the required assets exist, while profile-dependent smoke tests verify graceful degradation.

| TDD ID | Coverage area |
|---|---|
| T-SCHEMA-01..06 | loader tests + validator |
| T-FEAT-01..11 | feature modules |
| T-MISS-01..04 | extraction + imputation |
| T-LEAK-01..07 | fit-on-train contracts and split integrity |
| T-MODEL-01..07 | model/artifact layer |
| T-INF-01..05 | predictor/artifact layer |
| T-E2E-01..02 | trainer/ablation scripts |
| T-PERF-01..03 | performance/caching under configured environment |
