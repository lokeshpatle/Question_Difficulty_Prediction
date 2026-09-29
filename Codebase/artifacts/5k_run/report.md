# QDP Run Report

- Dependency profile: `P1`
- Records: **5000**
- Selected model: `xgb_d6_l0.1_s1.0`
- Validation Macro F1: **0.791260**
- Test Macro F1: **0.491471**
- Test Ordinal MAE: **0.618667**
- Test Balanced Accuracy: **0.497333**

Interpretability describes model reliance and does not establish causality.

## Ablation Results

# Ablation Results

| Run | Macro F1 | Ordinal MAE | Balanced Accuracy | Δ vs A0 |
|---|---:|---:|---:|---:|
| A0 | 0.166500 | 0.667111 | 0.333333 | 0.000000 |
| A1 | 0.410286 | 0.753662 | 0.417928 | 0.243786 |
| A2 | 0.425662 | 0.740346 | 0.431251 | 0.259162 |
| A3 | 0.432127 | 0.737683 | 0.437907 | 0.265627 |
| A4 | 0.419849 | 0.744341 | 0.427240 | 0.253349 |
| A5 | 0.413240 | 0.756325 | 0.420595 | 0.246740 |
| A6 | 0.462091 | 0.683089 | 0.469827 | 0.295591 |
| A7 | 0.485640 | 0.629827 | 0.495134 | 0.319139 |
| A8 | unavailable | | | |
| A8b-32 | unavailable | | | |
| A8b-48 | unavailable | | | |
| A8b-64 | unavailable | | | |
| S_lexical | 0.418794 | 0.744341 | 0.428526 | 0.252293 |
| S_readability | 0.342555 | 0.848202 | 0.342199 | 0.176054 |
| S_syntax | 0.384716 | 0.812250 | 0.386040 | 0.218216 |
| S_task | 0.331655 | 0.908123 | 0.356622 | 0.165155 |
| S_mcq | 0.209400 | 0.978695 | 0.336090 | 0.042900 |
| S_tfidf | 0.386796 | 0.832224 | 0.400526 | 0.220296 |
| S_semantic_scalar | 0.383026 | 0.782956 | 0.403097 | 0.216526 |
| S_programming | 0.440759 | 0.696405 | 0.464436 | 0.274259 |
| S_semantic_embedding | unavailable | | | |
