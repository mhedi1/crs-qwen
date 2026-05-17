import json
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

def build_dialogue_up_to(sample: dict, turn_index: int) -> str:
    """
    Builds dialogue text using only messages up to and including the given turn index.
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
    mlflow.set_experiment("crs-thesis")
    mlflow_run = mlflow.start_run()
    mlflow.log_params({
        "format": args.format,
        "dataset": args.dataset,
        "max_samples": args.max_samples,
        "recommendation_only": args.recommendation_only,
    })

    k_values = _cfg["evaluation"]["k_values"]
    hits = {k: [] for k in k_values}
    mrrs = []
    reranker_hits = []
    reranker_fallbacks = 0

    generated_responses = []
    reference_responses = []
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

    mode_label = "recommendation only" if args.recommendation_only else "full recommendation + response evaluation"
    print(f"\n{'='*60}")
    print(f"EVALUATION — {args.dataset.upper()} Test Set (Turn-by-Turn)")
    print(f"Format: {args.format}")
    print(f"Max conversations: {args.max_samples}")
    print(f"Mode: {mode_label}")
    print(f"{'='*60}\n")

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
                                dialogue_up_to = build_dialogue_up_to(sample, turn_index)

                                candidates, detected_decades = get_kbrd_candidates(dialogue_up_to, top_k=_cfg["pipeline"]["top_k_candidates"])

                                # Compute all values before appending to prevent partial appends if an exception fires
                                hit_values = {k: is_hit(candidates, recommended_movies, k) for k in k_values}
                                rank = get_rank(candidates, recommended_movies)
                                mrr = 1.0 / rank if rank > 0 else 0.0
                                selected_movie, is_fallback = rerank(
                                    dialogue_up_to, candidates,
                                    era_hints=detected_decades,
                                    serialization_format=args.format
                                )
                                reranker_hit = any(
                                    strict_title_match(selected_movie.get("title", ""), gt)
                                    for gt in recommended_movies
                                )

                                # Append all at once — only reached if all computations above succeeded
                                for k in k_values:
                                    hits[k].append(hit_values[k])
                                mrrs.append(mrr)
                                reranker_hits.append(reranker_hit)
                                if is_fallback:
                                    reranker_fallbacks += 1

                                if not args.recommendation_only:
                                    response = generate_response(dialogue_up_to, selected_movie)
                                    generated_responses.append(response)

                                    movie_mentions = sample.get("movieMentions", {})
                                    ref_text = msg.get("text", "").strip()
                                    for movie_id, movie_name in movie_mentions.items():
                                        ref_text = ref_text.replace(f"@{movie_id}", movie_name.strip())
                                    ref_text = ref_text.replace("&quot;", '"').replace("&amp;", "&")
                                    reference_responses.append(ref_text)

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
                                    print(f"[{total_evaluation_instances} instances] "
                                          f"R@1={r1:.4f} R@10={r10:.4f} R@50={r50:.4f}")

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
        final_metrics["recommendation"]["MRR"] = sum(mrrs) / len(mrrs)
        final_metrics["recommendation"]["Reranker@1"] = sum(reranker_hits) / len(reranker_hits) if reranker_hits else 0.0

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
        mlflow.log_metrics({
            "Recall_at_1": final_metrics["recommendation"]["Recall@1"],
            "Recall_at_10": final_metrics["recommendation"]["Recall@10"],
            "Recall_at_50": final_metrics["recommendation"]["Recall@50"],
            "MRR": final_metrics["recommendation"]["MRR"],
            "Reranker_at_1": final_metrics["recommendation"]["Reranker@1"],
        })
    mlflow.log_metric("reranker_fallbacks", reranker_fallbacks)
    final_metrics["mlflow_run_id"] = mlflow_run.info.run_id

    # Print summary table
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total conversations processed: {total_conversations_processed}")
    print(f"Total evaluation instances: {total_evaluation_instances}\n")
    
    if total_evaluation_instances > 0:
        print("Recommendation Metrics:")
        print(f"  Recall@1:    {final_metrics['recommendation']['Recall@1']:.4f}")
        print(f"  Recall@10:   {final_metrics['recommendation']['Recall@10']:.4f}")
        print(f"  Recall@50:   {final_metrics['recommendation']['Recall@50']:.4f}")
        print(f"  MRR:         {final_metrics['recommendation']['MRR']:.4f}")
        print(f"  Reranker@1:  {final_metrics['recommendation']['Reranker@1']:.4f}\n")
        
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

    print(f"Skip Report:")
    print(f"  Skipped conversations: {skipped_conversations}")
    print(f"  Skipped instances:     {skipped_instances}")
    print(f"  Reranker fallbacks:    {reranker_fallbacks}")
    print(f"{'='*60}")
    print(f"Results saved to experiments/eval_format{args.format}_{args.dataset}_{timestamp}.json")
    print(f"MLflow run ID: {mlflow_run.info.run_id}")
    print(f"{'='*60}")
    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CRS Pipeline")
    parser.add_argument("--format", type=int, choices=[1, 2, 3, 4], default=1,
                        help="Serialization format (1-4)")
    parser.add_argument("--dataset", type=str, choices=['redial', 'inspired'], default='redial',
                        help="Dataset choice (redial or inspired)")
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Max conversations to process")
    parser.add_argument("--recommendation_only", action="store_true", default=False,
                        help="Skip response generation; compute only recommendation metrics")

    args = parser.parse_args()
    evaluate(args)