# Error Analysis Results — ReDial Full Test

Configuration:
- Dataset: ReDial
- Format: 1
- System: KBRD + Qwen reranker
- Evaluated conversations: 1301
- Evaluation instances: 3898
- Error analysis file: experiments/error_analysis/error_analysis_redial_format1_reranked_20260518_021231.jsonl
- MLflow run ID: 7646fda4c753499baa3849031440dac8

## Retrieval Analysis

| Metric | Count | Rate |
|---|---:|---:|
| Gold in top-1 | 176 | 4.52% |
| Gold in top-10 | 649 | 16.65% |
| Gold in top-50 | 1272 | 32.63% |
| Gold not in top-50 | 2626 | 67.37% |

## Reranking Analysis

| Metric | Count | Rate |
|---|---:|---:|
| Qwen selected correct movie | 555 | 14.24% |
| Qwen success when gold is available | 555 / 1272 | 43.63% |
| Qwen failed while gold was available | 717 | 18.39% |
| Reranker fallbacks | 0 | 0% |

## Interpretation

The main bottleneck is candidate retrieval. The correct movie is missing from KBRD top-50 in 67.37% of recommendation instances, so Qwen cannot select it in those cases.

However, when the correct movie is available in the candidate list, Qwen selects it in 43.63% of cases. This explains why the proposed KBRD + Qwen system improves Final@1 from 0.0457 for KBRD-only to around 0.1424 for the reranked system.

Therefore, the next improvement should focus first on improving retrieval: better entity extraction, movie title normalization, alias matching, and candidate fusion.
