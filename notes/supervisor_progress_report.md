# Progress Report: Conversational Movie Recommender System (CRS)

## 1. Project Objective
The primary objective of this thesis is to develop a hybrid conversational movie recommender system. The proposed architecture employs a two-stage approach: a graph-based neural retrieval model (KBRD) is first utilized to retrieve a broad set of candidate movies, followed by a Large Language Model (Qwen) that reranks these candidates based on the nuanced conversational context.

## 2. System Architecture
The system architecture has been modularized into several sequential stages:
a. **Dialogue Input**: The user's conversational history is processed as the primary input.
b. **KBRD Candidate Retrieval**: The baseline KBRD model retrieves top movie candidates based on the dialogue context.
c. **Seed/Entity Extraction**: Movie entities (seeds) explicitly or implicitly mentioned in the dialogue are extracted.
d. **Seed Quality Filtering**: Extracted seeds are filtered to mitigate noisy or false-positive entity matches.
e. **Candidate Fusion**: Supplementary candidate sources are dynamically fused with the KBRD candidate pool.
f. **Qwen Reranking**: A Qwen LLM reranks the fused candidate list to select the optimal recommendation.
g. **Qwen Response Generation**: Finally, Qwen generates a natural language response seamlessly incorporating the selected recommendation.

### Pipeline Diagram
```text
Dialogue -> KBRD Candidate Retrieval -> Seed Filtering -> Candidate Fusion -> Qwen Reranking -> Qwen Response Generation
```

## 3. Implementation Progress
The core pipeline has been fully implemented. Key modules developed or modified include:
- `my_crs/kbrd_adapter.py`
- `my_crs/reranker.py`
- `my_crs/evaluate.py`
- `my_crs/prompts.py`
- `my_crs/response_generator.py`
- `experiments/improved_ekg/preprocess_inspired.py`

Furthermore, robust experimental tracking has been established using MLflow, enabling comprehensive logging of metrics and serialized JSON evaluation results.

## 4. Evaluation Protocol
The system is evaluated utilizing the ReDial turn-by-turn evaluation protocol. 

Crucially, we have implemented a strict **"no-leak" correction** in our evaluation methodology. Under this protocol, the model is strictly prevented from observing the current target recommendation turn in its dialogue history. This ensures that the model cannot trivially extract the ground-truth answer from the turn it is meant to predict. While this rigorous constraint naturally yields quantitatively lower performance metrics compared to standard baselines, it provides a significantly more reliable, honest, and realistic measure of the model's true recommendation capabilities.

Primary evaluation metrics include Recall@1, Recall@10, Recall@50, Precision@K, NDCG@K, MRR, and Reranker@1.

## 5. Main Results
The final no-leak ablation results on ReDial are summarized below. Detailed experiment outputs were generated on JupyterHub and logged under the experiments/ directory. The latest local extended-metrics result file is `experiments/eval_format1_redial_20260519_175558.json`.

| System | Recall@1 / Final@1 | Recall@10 | Recall@50 | Precision@1 | NDCG@10 | NDCG@50 | MRR | Reranker@1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pure KBRD, no fusion, no Qwen | 0.0231 | 0.0980 | 0.2188 | 0.0231 | 0.0487 | 0.0744 | 0.0457 | 0.0231 |
| KBRD + Candidate Fusion, no Qwen | 0.0221 | 0.0967 | 0.2453 | 0.0221 | 0.0481 | 0.0789 | 0.0457 | 0.0221 |
| KBRD + Candidate Fusion + Qwen reranker | 0.0228 | 0.0983 | 0.2447 | 0.0228 | 0.0493 | 0.0799 | 0.0470 | 0.0274 |

**Analysis and Interpretation:**
- **Improved Coverage**: Candidate Fusion successfully increases Recall@50 (from 0.2188 to ~0.245), indicating that the fusion process effectively expands the pool of relevant candidates.
- **Top-1 Bottleneck**: Despite the improved coverage, Candidate Fusion alone does not improve top-1 accuracy. This is structurally expected, as fused candidates are appended after the preserved KBRD top-30 list. 
- **Reranking Impact**: Qwen reranking provides a modest but distinct improvement in final selected recommendation accuracy (Reranker@1 improves from 0.0221 to 0.0274). However, the improvement is not yet large enough to drastically shift top-1 outcomes.
- **Conclusion**: The primary bottleneck remains the initial retrieval coverage and the inherent difficulty of ranking under the strict, realistic no-leak evaluation protocol.

## 6. INSPIRED Dataset Work
To evaluate cross-dataset generalization, the INSPIRED dataset has been preprocessed into a ReDial-compatible JSONL format (`data/inspired/test_data.jsonl`).
- The processed dataset comprises 88 dialogues and 263 recommendation turns.
- Analysis reveals that KBRD vocabulary coverage for INSPIRED is limited to approximately 47.3%.
- Consequently, we anticipate lower absolute performance on the INSPIRED dataset, as a significant portion of its movies fall outside the KBRD/ReDial training vocabulary.

## 7. Error Analysis
Initial qualitative evaluations identified a critical "seed noise" issue, wherein common conversational words (e.g., "it", "saw", "time", "loved") were erroneously extracted as movie titles.
- To address this, a robust seed quality filter was implemented, successfully mitigating false positives.
- To facilitate deeper failure interpretation, system diagnostics have been expanded to capture:
  - `extracted_seeds`
  - `filtered_noisy_seeds`
  - `qwen_fallback_seeds`
  - `candidate_sources`
  - `selected_candidate_source`
  - `gold_candidate_source`
- This granular diagnostic logging is now actively utilized to analyze edge cases and system failures.

## 8. Contributions to Date
The primary academic and technical contributions of the project thus far include:
- Development of a fully functional hybrid CRS pipeline integrating KBRD with Qwen.
- Implementation of a strict, honest no-leak turn-by-turn evaluation methodology.
- Engineering of a seed quality filter to resolve noisy entity extraction.
- Introduction of Candidate Fusion paired with comprehensive source diagnostics.
- Integration of LLM-based (Qwen) reranking and response generation.
- Establishment of extended evaluation metrics and MLflow experiment tracking.
- Preprocessing of the INSPIRED dataset for subsequent cross-dataset evaluation.
- Deployment of dedicated error analysis artifacts for rigorous failure interpretation.

## 9. Current Limitations
- **Retrieval Ceiling**: The baseline candidate retrieval coverage limits overall system potential.
- **Reranker Efficacy**: Qwen reranking currently yields only modest quantitative improvements.
- **Vocabulary Constraints**: KBRD's static vocabulary severely limits generalization to newer movies or out-of-domain datasets.
- **Response Generation Evaluation**: While the natural language response generation stage is fully implemented and operational, it has not yet been subjected to rigorous quantitative evaluation metrics (e.g., BLEU/ROUGE).
- **Fine-tuning**: No LoRA or prompt-tuning techniques have been applied to optimize the LLM yet.

## 10. Next Planned Work
- Finalize the documentation and formal report writing.
- Draft the methodology and experiments sections for the thesis.
- Conduct a preliminary quantitative evaluation of the response generation module, if time permits.
- Discuss with the supervisor whether implementing LoRA/prompt-tuning is strictly necessary or optional for the final submission.
- Polish final tables and figures for the thesis manuscript.
