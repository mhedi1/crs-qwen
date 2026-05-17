"""
Preprocess INSPIRED test.tsv into ReDial-compatible JSONL format.

Identifies recommendation turns as RECOMMENDER utterances that first mention
a movie not previously seen in the dialogue (new recommendation heuristic).
"""

import ast
import csv
import json
import os
import pickle
import re
import sys
from collections import defaultdict

# ── paths ────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))

TSV_PATH = os.path.join(
    _PROJECT_ROOT, "data", "inspired", "Inspired-master",
    "data", "dialog_data", "test.tsv"
)
OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "data", "inspired", "test_data.jsonl")
KBRD_DATA_DIR = os.path.join(
    _PROJECT_ROOT, "baseline_repo", "KBRD_project", "KBRD", "data", "redial"
)

RECOMMENDER_ID = 1
SEEKER_ID = 2


# ── helpers ───────────────────────────────────────────────────────────────────

def normalize_title(title: str) -> str:
    """Strip year, punctuation, collapse whitespace, lowercase."""
    title = re.sub(r'\(\d{4}\)', '', title)
    title = re.sub(r'[^\w\s]', '', title)
    title = re.sub(r'\s+', ' ', title)
    return title.lower().strip()


def load_kbrd_vocab() -> set:
    """Return a set of normalized movie titles from KBRD's entity vocabulary."""
    id2entity_path = os.path.join(KBRD_DATA_DIR, "id2entity.pkl")
    try:
        with open(id2entity_path, "rb") as f:
            id2entity = pickle.load(f)
        # Values are DBpedia URI strings like "Inception_(film)" or raw titles.
        titles = set()
        for v in id2entity.values():
            if not v:
                continue
            # Strip DBpedia URI wrapper: <http://dbpedia.org/resource/Movie_Name>
            name = re.sub(r'^<http://dbpedia\.org/resource/', '', v)
            name = re.sub(r'>$', '', name)
            # Underscores → spaces
            name = name.replace("_", " ").strip()
            # Remove trailing disambiguation like "(2009 film)" "(film)" "(1994)"
            name = re.sub(r'\(\s*\d*\s*film\s*\)', '', name, flags=re.I)
            name = re.sub(r'\(\s*\d{4}\s*\)', '', name)
            name = re.sub(r'\s+', ' ', name).strip()
            titles.add(normalize_title(name))
        return titles
    except FileNotFoundError:
        print(f"[WARN] KBRD vocab not found at {id2entity_path}; skipping match check")
        return set()


def parse_movie_dict(raw: str) -> dict:
    """Parse the movie_dict column (Python-literal dict) safely."""
    if not raw or raw.strip() in ('', '{}'):
        return {}
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}


def split_movies_field(movies_str: str) -> list:
    """Split the semicolon-separated movies field, stripping whitespace."""
    if not movies_str or not movies_str.strip():
        return []
    return [m.strip() for m in movies_str.split(';') if m.strip()]


def text_with_movie_ids(text_placeholder: str, movie_dict: dict) -> str:
    """
    Replace [MOVIE_TITLE_N] placeholders with @N so evaluate.py can
    locate @movie_id references in dialogue text.
    """
    # Replace each [MOVIE_TITLE_N] -> @N
    result = re.sub(r'\[MOVIE_TITLE_(\d+)\]', r'@\1', text_placeholder)
    return result


# ── main preprocessing ────────────────────────────────────────────────────────

def preprocess(tsv_path: str, output_path: str) -> list:
    """
    Read INSPIRED test.tsv, convert to ReDial JSONL, return list of records.
    """
    # Read all rows grouped by dialog_id
    dialogues: dict[str, list] = defaultdict(list)
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            dialogues[row['dialog_id']].append(row)

    # Sort utterances within each dialogue by utt_id
    for dialog_id in dialogues:
        dialogues[dialog_id].sort(key=lambda r: int(r['utt_id']))

    records = []
    total_rec_turns = 0

    for dialog_id, rows in dialogues.items():
        # Extract movie_dict from first row (same across all rows in a dialogue)
        movie_dict = parse_movie_dict(rows[0].get('movie_dict', ''))
        if not movie_dict:
            continue

        # Build id_to_title mapping (index is the movie ID we'll use)
        id_to_title = {str(v): k for k, v in movie_dict.items()}

        messages = []
        seen_movies: set = set()        # all movies mentioned so far
        recommended_ids: set = set()    # movie IDs first introduced by RECOMMENDER

        for row in rows:
            speaker = row['speaker']
            sender_id = RECOMMENDER_ID if speaker == 'RECOMMENDER' else SEEKER_ID

            # Build message text with @movie_id references
            text_ph = row.get('text_with_placeholder', '').strip()
            if not text_ph:
                # Fallback: use original text, removing QUOTATION_MARK tokens
                text_ph = row.get('text', '').replace('QUOTATION_MARK', '"')
            text = text_with_movie_ids(text_ph, movie_dict)

            # Determine movies mentioned in this utterance
            movies_in_turn = split_movies_field(row.get('movies', ''))

            # Track new recommendations (RECOMMENDER first-mentions)
            if speaker == 'RECOMMENDER':
                for movie in movies_in_turn:
                    if movie not in seen_movies and movie in movie_dict:
                        recommended_ids.add(str(movie_dict[movie]))

            seen_movies.update(movies_in_turn)

            messages.append({
                'timeOffset': 0,
                'text': text,
                'senderWorkerId': sender_id,
                'messageId': int(row['utt_id']),
            })

        if not recommended_ids:
            continue

        # Count how many messages actually reference a recommended movie
        # (used for stats; evaluate.py will count these at runtime)
        for msg in messages:
            if msg['senderWorkerId'] == RECOMMENDER_ID:
                mentioned = set(re.findall(r'@(\d+)', msg['text']))
                if mentioned & recommended_ids:
                    total_rec_turns += 1

        # Build respondentQuestions
        respondent_questions = {
            str(idx): {
                'suggested': 1 if str(idx) in recommended_ids else 0,
                'seen': 0,
                'liked': 0,
            }
            for title, idx in movie_dict.items()
        }

        records.append({
            'movieMentions': id_to_title,
            'respondentQuestions': respondent_questions,
            'messages': messages,
            'conversationId': dialog_id,
            'respondentWorkerId': RECOMMENDER_ID,
            'initiatorWorkerId': SEEKER_ID,
            'initiatorQuestions': {},
        })

    # Write JSONL output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')

    return records, total_rec_turns


def check_vocab_coverage(records: list, kbrd_vocab: set) -> tuple:
    """
    Check how many INSPIRED movie titles appear in the KBRD vocabulary.
    Returns (matched_titles, unmatched_titles).
    """
    all_titles: set = set()
    for rec in records:
        all_titles.update(rec['movieMentions'].values())

    matched = set()
    unmatched = set()
    for title in all_titles:
        if normalize_title(title) in kbrd_vocab:
            matched.add(title)
        else:
            unmatched.add(title)

    return matched, unmatched


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    print(f"Reading INSPIRED test data from:\n  {TSV_PATH}\n")

    if not os.path.exists(TSV_PATH):
        print(f"ERROR: File not found: {TSV_PATH}")
        sys.exit(1)

    records, total_rec_turns = preprocess(TSV_PATH, OUTPUT_PATH)

    print(f"Dialogues processed:       {len(records)}")
    print(f"Recommendation turns:      {total_rec_turns}")
    print(f"Output written to:\n  {OUTPUT_PATH}\n")

    # ── Show first 3 examples ─────────────────────────────────────────────────
    print("=" * 60)
    print("FIRST 3 OUTPUT RECORDS")
    print("=" * 60)
    for i, rec in enumerate(records[:3]):
        print(f"\n--- Record {i+1}: {rec['conversationId']} ---")
        print(f"Movies in dialogue:  {rec['movieMentions']}")
        suggested = [
            f"{rec['movieMentions'][mid]} (id={mid})"
            for mid, info in rec['respondentQuestions'].items()
            if info['suggested'] == 1
        ]
        print(f"Recommended movies:  {suggested}")
        print("Messages:")
        for msg in rec['messages']:
            role = "REC" if msg['senderWorkerId'] == RECOMMENDER_ID else "SEK"
            print(f"  [{role}] {msg['text'][:100]}")

    # ── KBRD vocabulary coverage ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("KBRD VOCABULARY COVERAGE CHECK")
    print("=" * 60)
    kbrd_vocab = load_kbrd_vocab()
    if kbrd_vocab:
        matched, unmatched = check_vocab_coverage(records, kbrd_vocab)
        all_count = len(matched) + len(unmatched)
        pct = 100 * len(matched) / all_count if all_count else 0
        print(f"INSPIRED unique movie titles: {all_count}")
        print(f"Matched in KBRD vocab:        {len(matched)} ({pct:.1f}%)")
        print(f"NOT in KBRD vocab:            {len(unmatched)}")
        if unmatched:
            print("\nUnmatched titles (KBRD will likely score 0 for these):")
            for t in sorted(unmatched)[:20]:
                print(f"  - {t}")
            if len(unmatched) > 20:
                print(f"  ... and {len(unmatched) - 20} more")


if __name__ == "__main__":
    main()
