# Experiment Log

All evaluation runs for the KBRD + Qwen CRS pipeline on the ReDial test set.
Format numbers correspond to serialization variants defined in `my_crs/reranker.py`.

---

## Ablation Runs — Format Comparison (200 conversations, ReDial)

These four runs were executed on 2026-05-13 to select the best serialization format.
All runs: `--max_samples 200 --dataset redial --recommendation_only`

### Format 1 — 200 conversations

| Field | Value |
|---|---|
| Command | `python my_crs/evaluate.py --format 1 --dataset redial --max_samples 200 --recommendation_only` |
| Dataset | ReDial test set |
| Max samples | 200 |
| Evaluated conversations | 200 |
| Evaluation instances | ~600 (est.) |
| Recall@1 | 0.0422 |
| Recall@10 | 0.1734 |
| Recall@50 | 0.3375 |
| MRR | 0.0826 |
| Reranker@1 | **0.1328** |
| Reranker fallbacks | 0 |
| Result file | `experiments/eval_format1_redial_20260513_181059.json` |
| Notes | Ablation run; Format 1 selected as best on Reranker@1 |

### Format 2 — 200 conversations

| Field | Value |
|---|---|
| Command | `python my_crs/evaluate.py --format 2 --dataset redial --max_samples 200 --recommendation_only` |
| Dataset | ReDial test set |
| Max samples | 200 |
| Evaluated conversations | 200 |
| Recall@1 | 0.0422 |
| Recall@10 | 0.1750 |
| Recall@50 | 0.3359 |
| MRR | 0.0826 |
| Reranker@1 | 0.1000 |
| Reranker fallbacks | 0 |
| Result file | `experiments/eval_format2_redial_20260513_181441.json` |
| Notes | Ablation run |

### Format 3 — 200 conversations

| Field | Value |
|---|---|
| Command | `python my_crs/evaluate.py --format 3 --dataset redial --max_samples 200 --recommendation_only` |
| Dataset | ReDial test set |
| Max samples | 200 |
| Evaluated conversations | 200 |
| Recall@1 | 0.0422 |
| Recall@10 | 0.1797 |
| Recall@50 | 0.3359 |
| MRR | 0.0838 |
| Reranker@1 | 0.0938 |
| Reranker fallbacks | 1 |
| Result file | `experiments/eval_format3_redial_20260513_181904.json` |
| Notes | Ablation run; highest R@10 and MRR of the four formats, but lower Reranker@1 |

### Format 4 — 200 conversations

| Field | Value |
|---|---|
| Command | `python my_crs/evaluate.py --format 4 --dataset redial --max_samples 200 --recommendation_only` |
| Dataset | ReDial test set |
| Max samples | 200 |
| Evaluated conversations | 200 |
| Recall@1 | 0.0422 |
| Recall@10 | 0.1781 |
| Recall@50 | 0.3391 |
| MRR | 0.0833 |
| Reranker@1 | 0.1031 |
| Reranker fallbacks | 0 |
| Result file | `experiments/eval_format4_redial_20260514_034943.json` |
| Notes | Ablation run; best R@50 of the four formats |

---

## Full Evaluation — Format 1, ReDial (Full Test Set)

| Field | Value |
|---|---|
| Command | `python my_crs/evaluate.py --format 1 --dataset redial --max_samples 9999 --recommendation_only` |
| Dataset | ReDial test set |
| Test file conversations | 1342 |
| Evaluated conversations | **1301** |
| Evaluation instances | **3898** |
| Recall@1 | **0.0459** |
| Recall@10 | **0.1662** |
| Recall@50 | **0.3279** |
| MRR | **0.0832** |
| Reranker@1 | **0.1383** |
| Reranker fallbacks | 4 |
| Result file | `experiments/eval_format1_redial_20260517_161211.json` |
| MLflow run ID | `9bc2d981141c4e038ad1d953f30316d4` |
| Notes | Full test-set run with the selected Format 1. 41 conversations skipped (errors/empty). Reranker@1 improves slightly vs. 200-conv ablation, consistent with more statistical power. |

---

## Planned / Pending Runs

| Run | Status |
|---|---|
| Format 1 full — INSPIRED test set | Pending (preprocessing done 2026-05-17) |
| KBRD-only baseline (no Qwen reranking) | Pending |
| Format 1 with response generation (50–100 conv) | Pending |
