import json
import math
import re
import sys
import os
import argparse
import tempfile
from datetime import datetime
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from collections import defaultdict
import mlflow
import yaml
import statistics

_MY_CRS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_MY_CRS_DIR)
_KBRD_PATH = os.path.join(_PROJECT_ROOT, "baseline_repo", "KBRD_project", "KBRD")

sys.path.insert(0, _MY_CRS_DIR)
sys.path.insert(0, _KBRD_PATH)

with open(os.path.join(_MY_CRS_DIR, "config.yaml")) as _f:
    _cfg = yaml.safe_load(_f)

from kbrd_adapter import get_kbrd_candidates
from reranker import rerank
from response_generator import generate_response

def normalize_title(title: str) -> str:
    """Normalize title: strip year, punctuation, collapse whitespace, lowercase."""
    title = re.sub(r'\(\d{4}\)', '', title)
    title = re.sub(r'[^\w\s]', '', title)
    title = re.sub(r'\s+', ' ', title)
    return title.lower().strip()

def strict_title_match(title_a: str, title_b: str) -> bool:
    """Check whether two movie titles are identical after normalization.

    Args:
        title_a: First title string.
        title_b: Second title string.

    Returns:
        True if normalized titles are equal, False otherwise.
    """
    return normalize_title(title_a) == normalize_title(title_b)

def is_hit(candidates: list, ground_truth: list, k: int) -> bool:
    """Check whether any ground-truth title appears in the top-k candidates.

    Args:
        candidates: Ranked list of candidate dicts with a "title" key.
        ground_truth: Ground-truth movie title strings to match against.
        k: Number of top candidates to consider.

    Returns:
        True if at least one ground-truth title exactly matches a top-k candidate.
    """
    top_k = candidates[:k]
    for c in top_k:
        for gt_movie in ground_truth:
            if strict_title_match(c.get("title", ""), gt_movie):
                return True
    return False

def get_rank(candidates: list, ground_truth: list) -> int:
    """Return the 1-indexed rank of the first ground-truth hit in candidates.

    Args:
        candidates: Ranked list of candidate dicts with a "title" key.
        ground_truth: Ground-truth movie title strings to match against.

    Returns:
        Rank of the first matching candidate (1-indexed), or 0 if no hit.
    """
    for rank, c in enumerate(candidates, 1):
        for gt_movie in ground_truth:
            if strict_title_match(c.get("title", ""), gt_movie):
                return rank
    return 0

def get_precision_at_k(candidates: list, ground_truth: list, k: int) -> float:
    """Precision@K: fraction of top-K candidates that match any gold movie."""
    if k == 0:
        return 0.0
    top_k = candidates[:k]
    hits = sum(
        1 for c in top_k
        if any(strict_title_match(c.get("title", ""), gt) for gt in ground_truth)
    )
    return hits / k


def get_ndcg_at_k(candidates: list, ground_truth: list, k: int) -> float:
    """NDCG@K with binary relevance; IDCG uses min(|gold|, K) ideal top positions."""
    top_k = candidates[:k]
    dcg = 0.0
    for i, c in enumerate(top_k, start=1):
        rel = 1.0 if any(strict_title_match(c.get("title", ""), gt) for gt in ground_truth) else 0.0
        dcg += rel / math.log2(i + 1)

    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def build_dialogue_up_to(sample: dict, turn_index: int) -> str:
    """
    Builds dialogue text using only messages up to and INCLUDING the given turn index.
    Pass turn_index - 1 to exclude the current (recommendation) turn from the context.
    Replaces @movie_id with movie name.
    Cleans HTML entities.
    """
    movie_mentions = sample.get("movieMentions", {})
    messages = sample.get("messages", [])
    initiator = sample.get("initiatorWorkerId", -1)

    turns = []
    for i, msg in enumerate(messages):
        if i > turn_index:
            break
            
        role = "User" if msg.get("senderWorkerId") == initiator else "System"
        text = msg.get("text", "").strip()

        for movie_id, movie_name in movie_mentions.items():
            text = text.replace(f"@{movie_id}", movie_name.strip())

        text = text.replace("&quot;", '"').replace("&amp;", "&")

        if text:
            turns.append(f"{role}: {text}")

    return "\n".join(turns)

def get_recommended_movies_at_turn(sample: dict, turn_index: int) -> list:
    """
    Looks at the message at turn_index.
    Finds all @movie_id references in the text.
    Returns only those movies where "suggested" equals 1 in respondentQuestions.
    """
    messages = sample.get("messages", [])
    if turn_index >= len(messages):
        return []
        
    msg = messages[turn_index]
    text = msg.get("text", "")
    movie_mentions = sample.get("movieMentions", {})
    respondent_questions = sample.get("respondentQuestions", {})

    if isinstance(respondent_questions, list):
        print(f"[WARN] respondentQuestions is a list (index {turn_index}), treating as empty dict")
        respondent_questions = {}

    movie_ids_in_text = re.findall(r'@(\d+)', text)

    recommended_movies = []
    for movie_id in movie_ids_in_text:
        info = respondent_questions.get(movie_id, {})
        if info.get("suggested", 0) == 1:
            movie_name = movie_mentions.get(movie_id, "")
            if movie_name:
                recommended_movies.append(movie_name.strip().lower())
                
    return recommended_movies

def calculate_distinct_n(responses: list, n: int) -> float:
    """Calculate Distinct-N for a list of responses."""
    if not responses:
        return 0.0
    
    unique_ngrams = set()
    total_ngrams = 0
    
    for response in responses:
        tokens = nltk.word_tokenize(response.lower())
        if len(tokens) < n:
            continue
        ngrams = list(nltk.ngrams(tokens, n))
        unique_ngrams.update(ngrams)
        total_ngrams += len(ngrams)
        
    if total_ngrams == 0:
        return 0.0
    return len(unique_ngrams) / total_ngrams

def evaluate(args):
    """Run the turn-by-turn CRS evaluation loop over the test set.

    Computes recommendation and conversation metrics, logs parameters and
    metrics to MLflow, and writes a JSON report to the experiments/ directory.

    Args:
        args: Parsed CLI arguments (format, dataset, max_samples, recommendation_only).
    """
    mlflow_db_path = os.path.join(_PROJECT_ROOT, "experiments", f"mlflow_format{args.format}.db")
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
    
    experiment_name = _cfg["mlflow"]["experiment_name"]
    if _cfg["extraction"].get("use_improved_extraction", False):
        experiment_name = "crs-thesis-entity-experiment"
        
    mlflow.set_experiment(experiment_name)
    mlflow_run = mlflow.start_run()
    mlflow.log_params({
        "format": args.format,
        "dataset": args.dataset,
        "max_samples": args.max_samples,
        "recommendation_only": args.recommendation_only,
        "skip_reranker": args.skip_reranker,
        "disable_fusion": args.disable_fusion,
    })

    k_values = _cfg["evaluation"]["k_values"]
    hits = {k: [] for k in k_values}
    precisions = {k: [] for k in k_values}
    ndcg_values = {k: [] for k in k_values}
    mrrs = []
    reranker_hits = []
    reranker_fallbacks = 0

    # per-conversation success trackers (keyed by conversation_id)
    _conv_success1: dict[str, bool] = {}
    _conv_recall10: dict[str, bool] = {}
    _conv_recall50: dict[str, bool] = {}

    gold_ranks_found: list[int] = []

    generated_responses = []
    reference_responses = []
    response_sample_records = []
    response_lengths = []

    total_conversations_processed = 0
    total_evaluation_instances = 0
    skipped_instances = 0
    skipped_conversations = 0

    if args.dataset == 'redial':
        data_path = os.path.join(_PROJECT_ROOT, "baseline_repo", "KBRD_project", "KBRD",
                                 "data", "redial", "test_data.jsonl")
    else:
        data_path = os.path.join(_PROJECT_ROOT, "data", "inspired", "test_data.jsonl")
        if not os.path.exists(data_path):
            print(f"INSPIRED dataset not found at {data_path}.\nPlease download it first.")
            return

    if args.skip_reranker:
        mode_label = "KBRD-only"
    elif args.recommendation_only:
        mode_label = "recommendation only"
    else:
        mode_label = "full recommendation + response evaluation"
    print(f"\n{'='*60}")
    print(f"EVALUATION — {args.dataset.upper()} Test Set (Turn-by-Turn)")
    print(f"Format: {args.format}")
    print(f"Max conversations: {args.max_samples}")
    print(f"Mode: {mode_label}")
    print(f"{'='*60}\n")

    error_analysis_records = [] if args.save_error_analysis else None

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = defaultdict(list)
    bleu_scores = []
    smooth_fn = SmoothingFunction().method1

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if total_conversations_processed >= args.max_samples:
                break
                
            try:
                sample = json.loads(line)
                messages = sample.get("messages", [])
                respondent = sample.get("respondentWorkerId", -1)
                
                conversation_has_instances = False
                
                for turn_index, msg in enumerate(messages):
                    sender = msg.get("senderWorkerId", -1)
                    
                    if sender == respondent:
                        # System turn
                        recommended_movies = get_recommended_movies_at_turn(sample, turn_index)
                        
                        if recommended_movies:
                            try:
                                # Evaluation instance
                                dialogue_up_to = build_dialogue_up_to(sample, turn_index - 1)

                                _diag = {} if error_analysis_records is not None else None
                                candidates, detected_decades = get_kbrd_candidates(
                                    dialogue_up_to,
                                    top_k=_cfg["pipeline"]["top_k_candidates"],
                                    diagnostics=_diag,
                                    use_fusion=not args.disable_fusion,
                                )

                                # Compute all values before appending to prevent partial appends if an exception fires
                                hit_values = {k: is_hit(candidates, recommended_movies, k) for k in k_values}
                                prec_values = {k: get_precision_at_k(candidates, recommended_movies, k) for k in k_values}
                                ndcg_vals = {k: get_ndcg_at_k(candidates, recommended_movies, k) for k in k_values}
                                rank = get_rank(candidates, recommended_movies)
                                mrr = 1.0 / rank if rank > 0 else 0.0
                                if args.skip_reranker:
                                    selected_movie = candidates[0]
                                    is_fallback = False
                                else:
                                    selected_movie, is_fallback = rerank(
                                        dialogue_up_to, candidates,
                                        era_hints=detected_decades,
                                        serialization_format=args.format
                                    )
                                reranker_hit = any(
                                    strict_title_match(selected_movie.get("title", ""), gt)
                                    for gt in recommended_movies
                                )

                                conv_id = str(sample.get("conversationId", total_conversations_processed))

                                # Append all at once — only reached if all computations above succeeded
                                for k in k_values:
                                    hits[k].append(hit_values[k])
                                    precisions[k].append(prec_values[k])
                                    ndcg_values[k].append(ndcg_vals[k])
                                mrrs.append(mrr)
                                reranker_hits.append(reranker_hit)
                                if is_fallback:
                                    reranker_fallbacks += 1

                                if rank > 0:
                                    gold_ranks_found.append(rank)

                                # Conversation-level success flags (OR across turns)
                                if reranker_hit:
                                    _conv_success1[conv_id] = True
                                elif conv_id not in _conv_success1:
                                    _conv_success1[conv_id] = False

                                if hit_values.get(10, False):
                                    _conv_recall10[conv_id] = True
                                elif conv_id not in _conv_recall10:
                                    _conv_recall10[conv_id] = False

                                if hit_values.get(50, False):
                                    _conv_recall50[conv_id] = True
                                elif conv_id not in _conv_recall50:
                                    _conv_recall50[conv_id] = False

                                if error_analysis_records is not None:
                                    error_analysis_records.append({
                                        "conversation_id": sample.get("conversationId", total_conversations_processed),
                                        "turn_index": turn_index,
                                        "dataset": args.dataset,
                                        "format": args.format,
                                        "skip_reranker": args.skip_reranker,
                                        "dialogue_history": dialogue_up_to,
                                        "ground_truth_movies": recommended_movies,
                                        "detected_decades": detected_decades,
                                        "kbrd_top1": candidates[0].get("title", "") if candidates else "",
                                        "selected_movie": selected_movie.get("title", ""),
                                        "correct": reranker_hit,
                                        "gold_rank": rank if rank > 0 else -1,
                                        "gold_in_top_1": hit_values.get(1, False),
                                        "gold_in_top_10": hit_values.get(10, False),
                                        "gold_in_top_50": hit_values.get(50, False),
                                        "candidate_count": len(candidates),
                                        "reranker_fallback": is_fallback,
                                        "extracted_seeds": _diag.get("extracted_seeds", []),
                                        "qwen_fallback_seeds": _diag.get("qwen_fallback_seeds", []),
                                        "seed_entity_ids": _diag.get("seed_entity_ids", []),
                                        "weak_seed_fallback": _diag.get("weak_seed_fallback", False),
                                        "num_extracted_seeds": _diag.get("num_extracted_seeds", 0),
                                        "num_matched_seeds": _diag.get("num_matched_seeds", 0),
                                        "filtered_noisy_seeds": _diag.get("filtered_noisy_seeds", []),
                                        "num_filtered_noisy_seeds": _diag.get("num_filtered_noisy_seeds", 0),
                                        "num_fused_seed_candidates": _diag.get("num_fused_seed_candidates", 0),
                                        "num_fused_qwen_candidates": _diag.get("num_fused_qwen_candidates", 0),
                                        "fused_candidate_titles": _diag.get("fused_candidate_titles", []),
                                        "candidate_sources": _diag.get("candidate_sources", {}),
                                        "selected_candidate_source": selected_movie.get("source", "UNKNOWN"),
                                        "gold_candidate_source": next(
                                            (
                                                c.get("source", "UNKNOWN")
                                                for c in candidates
                                                if any(
                                                    strict_title_match(c.get("title", ""), gt)
                                                    for gt in recommended_movies
                                                )
                                            ),
                                            "NOT_IN_LIST",
                                        ),
                                    })

                                if not args.recommendation_only:
                                    response = generate_response(dialogue_up_to, selected_movie)
                                    generated_responses.append(response)

                                    movie_mentions = sample.get("movieMentions", {})
                                    ref_text = msg.get("text", "").strip()
                                    for movie_id, movie_name in movie_mentions.items():
                                        ref_text = ref_text.replace(f"@{movie_id}", movie_name.strip())
                                    ref_text = ref_text.replace("&quot;", '"').replace("&amp;", "&")
                                    reference_responses.append(ref_text)

                                    response_sample_records.append({
                                        "conversation_id": sample.get("conversationId"),
                                        "turn_index": turn_index,
                                        "selected_movie": selected_movie.get("title", "Unknown"),
                                        "generated_response": response,
                                        "reference_response": ref_text,
                                    })

                                    response_lengths.append(len(nltk.word_tokenize(response)))

                                    ref_tokens = [nltk.word_tokenize(ref_text.lower())]
                                    gen_tokens = nltk.word_tokenize(response.lower())
                                    bleu = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smooth_fn)
                                    bleu_scores.append(bleu)

                                    r_scores = scorer.score(ref_text, response)
                                    rouge_scores['rouge1'].append(r_scores['rouge1'].fmeasure)
                                    rouge_scores['rouge2'].append(r_scores['rouge2'].fmeasure)
                                    rouge_scores['rougeL'].append(r_scores['rougeL'].fmeasure)

                                total_evaluation_instances += 1
                                conversation_has_instances = True

                                if total_evaluation_instances % 10 == 0:
                                    r1 = sum(hits[1]) / len(hits[1]) if hits[1] else 0
                                    r10 = sum(hits[10]) / len(hits[10]) if hits[10] else 0
                                    r50 = sum(hits[50]) / len(hits[50]) if hits[50] else 0
                                    p1 = sum(precisions[1]) / len(precisions[1]) if precisions[1] else 0
                                    nd10 = sum(ndcg_values[10]) / len(ndcg_values[10]) if ndcg_values[10] else 0
                                    print(f"[{total_evaluation_instances} instances] "
                                          f"R@1={r1:.4f} R@10={r10:.4f} R@50={r50:.4f} "
                                          f"P@1={p1:.4f} NDCG@10={nd10:.4f}")

                            except Exception as e:
                                print(f"[SKIP] Instance error (conv turn {turn_index}): {e}")
                                skipped_instances += 1
                                      
                if conversation_has_instances:
                    total_conversations_processed += 1
                    
            except Exception as e:
                print(f"[SKIP] Conversation error: {e}")
                skipped_conversations += 1
                continue

    # Calculate final metrics
    final_metrics = {
        "dataset": args.dataset,
        "format": args.format,
        "recommendation_only": args.recommendation_only,
        "skip_reranker": args.skip_reranker,
        "disable_fusion": args.disable_fusion,
        "conversations": total_conversations_processed,
        "instances": total_evaluation_instances,
        "skipped_conversations": skipped_conversations,
        "skipped_instances": skipped_instances,
        "reranker_fallbacks": reranker_fallbacks,
        "recommendation": {},
        "conversation": {}
    }
    
    if total_evaluation_instances > 0:
        for k in k_values:
            final_metrics["recommendation"][f"Recall@{k}"] = sum(hits[k]) / len(hits[k])
            final_metrics["recommendation"][f"Precision@{k}"] = sum(precisions[k]) / len(precisions[k])
            final_metrics["recommendation"][f"NDCG@{k}"] = sum(ndcg_values[k]) / len(ndcg_values[k])
        final_metrics["recommendation"]["MRR"] = sum(mrrs) / len(mrrs)
        final_metrics["recommendation"]["Reranker@1"] = sum(reranker_hits) / len(reranker_hits) if reranker_hits else 0.0

        final_metrics["recommendation"]["AvgGoldRank"] = (
            sum(gold_ranks_found) / len(gold_ranks_found) if gold_ranks_found else None
        )
        final_metrics["recommendation"]["MedianGoldRank"] = (
            statistics.median(gold_ranks_found) if gold_ranks_found else None
        )

        n_conv = len(_conv_success1)
        final_metrics["conversation_level"] = {
            "conversations_evaluated": n_conv,
            "ConvSuccess@1": sum(_conv_success1.values()) / n_conv if n_conv else 0.0,
            "ConvRecall@10": sum(_conv_recall10.values()) / len(_conv_recall10) if _conv_recall10 else 0.0,
            "ConvRecall@50": sum(_conv_recall50.values()) / len(_conv_recall50) if _conv_recall50 else 0.0,
        }

        if not args.recommendation_only:
            final_metrics["conversation"]["Distinct-2"] = calculate_distinct_n(generated_responses, 2)
            final_metrics["conversation"]["Distinct-3"] = calculate_distinct_n(generated_responses, 3)
            final_metrics["conversation"]["Distinct-4"] = calculate_distinct_n(generated_responses, 4)
            final_metrics["conversation"]["BLEU"] = sum(bleu_scores) / len(bleu_scores)
            final_metrics["conversation"]["ROUGE-1"] = sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1'])
            final_metrics["conversation"]["ROUGE-2"] = sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2'])
            final_metrics["conversation"]["ROUGE-L"] = sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL'])
            final_metrics["conversation"]["Avg_Length"] = sum(response_lengths) / len(response_lengths)

    if total_evaluation_instances > 0:
        mlflow_rec = final_metrics["recommendation"]
        mlflow_metrics: dict = {
            "Recall_at_1": mlflow_rec["Recall@1"],
            "Recall_at_10": mlflow_rec["Recall@10"],
            "Recall_at_50": mlflow_rec["Recall@50"],
            "Precision_at_1": mlflow_rec["Precision@1"],
            "Precision_at_10": mlflow_rec["Precision@10"],
            "Precision_at_50": mlflow_rec["Precision@50"],
            "NDCG_at_10": mlflow_rec["NDCG@10"],
            "NDCG_at_50": mlflow_rec["NDCG@50"],
            "MRR": mlflow_rec["MRR"],
            "Reranker_at_1": mlflow_rec["Reranker@1"],
        }
        if mlflow_rec["AvgGoldRank"] is not None:
            mlflow_metrics["AvgGoldRank"] = mlflow_rec["AvgGoldRank"]
        if mlflow_rec["MedianGoldRank"] is not None:
            mlflow_metrics["MedianGoldRank"] = float(mlflow_rec["MedianGoldRank"])
        conv_lvl = final_metrics.get("conversation_level", {})
        mlflow_metrics["ConvSuccess_at_1"] = conv_lvl.get("ConvSuccess@1", 0.0)
        mlflow_metrics["ConvRecall_at_10"] = conv_lvl.get("ConvRecall@10", 0.0)
        mlflow_metrics["ConvRecall_at_50"] = conv_lvl.get("ConvRecall@50", 0.0)
        mlflow.log_metrics(mlflow_metrics)
    mlflow.log_metric("reranker_fallbacks", reranker_fallbacks)
    final_metrics["mlflow_run_id"] = mlflow_run.info.run_id

    # Print summary table
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total conversations processed: {total_conversations_processed}")
    print(f"Total evaluation instances: {total_evaluation_instances}\n")
    
    if total_evaluation_instances > 0:
        rec = final_metrics["recommendation"]
        print("Recommendation Metrics:")
        print(f"  Recall@1:      {rec['Recall@1']:.4f}")
        print(f"  Recall@10:     {rec['Recall@10']:.4f}")
        print(f"  Recall@50:     {rec['Recall@50']:.4f}")
        print(f"  Precision@1:   {rec['Precision@1']:.4f}")
        print(f"  Precision@10:  {rec['Precision@10']:.4f}")
        print(f"  Precision@50:  {rec['Precision@50']:.4f}")
        print(f"  NDCG@10:       {rec['NDCG@10']:.4f}")
        print(f"  NDCG@50:       {rec['NDCG@50']:.4f}")
        print(f"  MRR:           {rec['MRR']:.4f}")
        print(f"  Reranker@1:    {rec['Reranker@1']:.4f}")
        avg_r = rec["AvgGoldRank"]
        med_r = rec["MedianGoldRank"]
        print(f"  AvgGoldRank:   {avg_r:.2f}" if avg_r is not None else "  AvgGoldRank:   N/A")
        print(f"  MedianGoldRank:{med_r:.1f}" if med_r is not None else "  MedianGoldRank:N/A")

        conv_lvl = final_metrics.get("conversation_level", {})
        print(f"\nConversation-Level Metrics ({conv_lvl.get('conversations_evaluated', 0)} conversations):")
        print(f"  ConvSuccess@1:  {conv_lvl.get('ConvSuccess@1', 0.0):.4f}")
        print(f"  ConvRecall@10:  {conv_lvl.get('ConvRecall@10', 0.0):.4f}")
        print(f"  ConvRecall@50:  {conv_lvl.get('ConvRecall@50', 0.0):.4f}")
        print()
        
        if not args.recommendation_only:
            print("Conversation Metrics:")
            print(f"  Distinct-2: {final_metrics['conversation']['Distinct-2']:.4f}")
            print(f"  Distinct-3: {final_metrics['conversation']['Distinct-3']:.4f}")
            print(f"  Distinct-4: {final_metrics['conversation']['Distinct-4']:.4f}")
            print(f"  BLEU:       {final_metrics['conversation']['BLEU']:.4f}")
            print(f"  ROUGE-1:    {final_metrics['conversation']['ROUGE-1']:.4f}")
            print(f"  ROUGE-2:    {final_metrics['conversation']['ROUGE-2']:.4f}")
            print(f"  ROUGE-L:    {final_metrics['conversation']['ROUGE-L']:.4f}")
            print(f"  Avg Length: {final_metrics['conversation']['Avg_Length']:.2f} words\n")

    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_dir, f"eval_format{args.format}_{args.dataset}_{timestamp}.json")

    fd, tmp_path = tempfile.mkstemp(dir=results_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
            json.dump(final_metrics, tmp_f, indent=4)
        os.replace(tmp_path, results_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    # Save response samples for qualitative thesis analysis
    if not args.recommendation_only and response_sample_records:
        rs_dir = os.path.join(results_dir, "response_samples")
        os.makedirs(rs_dir, exist_ok=True)
        rs_path = os.path.join(rs_dir, f"responses_{args.dataset}_{timestamp}.jsonl")
        with open(rs_path, "w", encoding="utf-8") as f:
            for record in response_sample_records:
                f.write(json.dumps(record) + "\n")
        print(f"Response samples saved to {rs_path}")

    # Save error analysis JSONL if requested
    ea_path = None
    if error_analysis_records is not None:
        reranker_tag = "kbrd_only" if args.skip_reranker else "reranked"
        ea_dir = os.path.join(results_dir, "error_analysis")
        os.makedirs(ea_dir, exist_ok=True)
        ea_filename = f"error_analysis_{args.dataset}_format{args.format}_{reranker_tag}_{timestamp}.jsonl"
        ea_path = os.path.join(ea_dir, ea_filename)
        fd_ea, tmp_ea = tempfile.mkstemp(dir=ea_dir, suffix=".tmp")
        try:
            with os.fdopen(fd_ea, "w", encoding="utf-8") as ea_f:
                for record in error_analysis_records:
                    ea_f.write(json.dumps(record) + "\n")
            os.replace(tmp_ea, ea_path)
        except Exception:
            try:
                os.remove(tmp_ea)
            except OSError:
                pass
            raise

    print(f"Skip Report:")
    print(f"  Skipped conversations: {skipped_conversations}")
    print(f"  Skipped instances:     {skipped_instances}")
    print(f"  Reranker fallbacks:    {reranker_fallbacks}")
    print(f"{'='*60}")
    print(f"Results saved to experiments/eval_format{args.format}_{args.dataset}_{timestamp}.json")
    if ea_path:
        print(f"Error analysis saved to experiments/error_analysis/{ea_filename}")
    print(f"MLflow run ID: {mlflow_run.info.run_id}")
    print(f"{'='*60}")
    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CRS Pipeline")
    parser.add_argument("--format", type=int, choices=[1, 2, 3, 4], default=_cfg["pipeline"]["serialisation_format"],
                        help="Serialization format (1-4)")
    parser.add_argument("--dataset", type=str, choices=['redial', 'inspired'], default='redial',
                        help="Dataset choice (redial or inspired)")
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Max conversations to process")
    parser.add_argument("--recommendation_only", action="store_true", default=False,
                        help="Skip response generation; compute only recommendation metrics")
    parser.add_argument("--skip_reranker", action="store_true", default=False,
                        help="Skip Qwen reranking; use KBRD top-1 candidate directly")
    parser.add_argument("--disable_fusion", action="store_true", default=False,
                        help="Disable Candidate Fusion; use pure KBRD candidates only")
    parser.add_argument("--save_error_analysis", action="store_true", default=False,
                        help="Save per-instance error analysis records to experiments/error_analysis/")

    args = parser.parse_args()
    evaluate(args)