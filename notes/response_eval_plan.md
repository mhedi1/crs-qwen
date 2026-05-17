# Response Generation Evaluation Plan

## Sequencing

Response generation evaluation should be run **after** recommendation experiments
are finalized. Recommendation quality (Reranker@1, Recall@K) is the primary thesis
contribution; response quality is secondary and validates end-to-end usability.

---

## Evaluation Run

**Command:**
```bash
python my_crs/evaluate.py --format 1 --dataset redial --max_samples 100
```
(Omit `--recommendation_only` to enable response generation.)

**Suggested scale:** 50–100 conversations to balance compute cost and statistical
reliability. Start with 50; expand to 100 if results are noisy.

---

## Automatic Metrics

These are computed by `evaluate.py` automatically when not in `--recommendation_only` mode.

| Metric | What it measures |
|---|---|
| Distinct-2 | Bigram diversity across all generated responses — captures repetitiveness |
| Distinct-3 | Trigram diversity |
| Distinct-4 | 4-gram diversity |
| BLEU | Surface-level overlap with the reference (human) response |
| ROUGE-1 | Unigram recall vs. reference |
| ROUGE-2 | Bigram recall vs. reference |
| ROUGE-L | Longest common subsequence vs. reference |
| Avg Length | Mean token count of generated responses |

**Caveat:** BLEU and ROUGE compare against a single reference response from ReDial,
which is a noisy proxy. Distinct-N is more informative for open-ended generation.

---

## Manual Evaluation Protocol

Evaluate **20 randomly sampled responses** with three criteria, each scored 0–2.

| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| **Fluency** | Ungrammatical or incoherent | Understandable but awkward | Natural and fluent |
| **Informativeness** | No useful content about the movie | Some relevant detail | Specific, informative, engaging |
| **Recommendation consistency** | Contradicts or ignores the recommended movie | Mentions the movie but superficially | Clearly advocates for the recommended movie with supporting detail |

**Annotator:** Single annotator (thesis author) is sufficient for a pilot; note this
as a limitation. Use a fixed random seed when sampling the 20 examples.

**Score sheet path:** `experiments/manual_response_eval_{timestamp}.csv`

Columns: `conversation_id`, `turn_index`, `recommended_movie`, `generated_response`,
`fluency`, `informativeness`, `consistency`, `notes`

---

## Optional Comparison

If feasible, compare responses generated from two conditions:

| Condition | Movie source |
|---|---|
| A — KBRD top-1 | The highest-ranked KBRD candidate (no reranking) |
| B — Qwen-reranked | The movie selected by Qwen after reranking |

This isolates the downstream effect of better reranking on response quality:
a better-chosen movie should yield a more relevant and coherent response.
Run automatic metrics on both conditions and manual eval on 10 examples each.
