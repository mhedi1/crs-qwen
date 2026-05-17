# Next Steps

Ordered by priority. Items marked [DONE] are complete.

---

## Immediate

- [ ] **Run Format 1 full evaluation on INSPIRED test set**
  - Preprocessing complete as of 2026-05-17 (`data/inspired/test_data.jsonl`, 88 dialogues, 263 rec turns).
  - Command: `python my_crs/evaluate.py --format 1 --dataset inspired --max_samples 9999 --recommendation_only`
  - Expected challenge: ~53% of INSPIRED movie titles are outside KBRD's vocabulary (2019+ films).

- [ ] **Compare Format 1 results: ReDial vs. INSPIRED**
  - Key question: how much does the vocabulary gap degrade Recall@K and Reranker@1?
  - Document in `notes/experiment_log.md` and update `notes/results_tables_template.md`.

---

## Baseline

- [ ] **Implement KBRD-only baseline (no Qwen reranking)**
  - Evaluate using the raw KBRD top-1 candidate instead of Qwen's selection.
  - This isolates the contribution of Qwen reranking to overall performance.
  - Result to record: KBRD top-1 accuracy (equivalent to Recall@1 of the retriever alone).

- [ ] **Compare KBRD-only vs. KBRD + Qwen**
  - Primary comparison: KBRD Recall@1 vs. Reranker@1 (pipeline).
  - Secondary: MRR improvement.
  - This is a central thesis contribution table.

---

## Analysis

- [ ] **Add error analysis output to evaluate.py**
  - Save per-instance error records to a JSONL file alongside the eval JSON.
  - Fields: see `notes/error_analysis_plan.md`.
  - Error categories: retrieval failure, reranking failure, entity extraction failure, parsing/API failure.

- [ ] **Run error analysis on Format 1 full-run output**
  - Sample 50 failure cases; classify by error type.
  - Identify patterns (e.g., short dialogues, missing seeds, unknown movies).

---

## Response Generation

- [ ] **Run small response generation evaluation (50–100 conversations)**
  - Command: `python my_crs/evaluate.py --format 1 --dataset redial --max_samples 100`
  - (omit `--recommendation_only` to enable response generation)
  - Record: Distinct-2/3/4, BLEU, ROUGE-1/2/L, Avg Length.
  - See `notes/response_eval_plan.md` for full plan including manual evaluation protocol.

---

## Optional / Later

- [ ] **LoRA / QLoRA fine-tuning of a smaller Qwen model**
  - Fine-tune on ReDial (dialogue → recommendation) pairs to improve in-domain reranking.
  - Requires GPU access and ~1–2 days of compute time.
  - Evaluate the fine-tuned model under the same ablation protocol.
  - Only pursue if base-model results are insufficient for thesis contribution claims.
