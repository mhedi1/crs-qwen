def truncate_history(history: str, 
                     max_turns: int = 5) -> str:
    """
    Keep only the last max_turns lines of dialogue.
    Prevents prompt overflow for long conversations.
    """
    lines = [l for l in history.strip().split('\n') 
             if l.strip()]
    if len(lines) <= max_turns:
        return history
    truncated = lines[-max_turns:]
    return '\n'.join(truncated)


def build_rerank_prompt(history: str, candidates: list[dict], era_hints: list = None, serialization_format: int = 1) -> str:
    """
    Build the prompt for Qwen reranking.
    Each candidate should have:
    - id
    - title
    - genre
    - decade
    - year (optional)
    - director (optional)
    """
    history = truncate_history(history, max_turns=5)
    candidate_lines = []
    for i, c in enumerate(candidates):
        title = c['title']
        year = c.get('year', None)
        genre = c.get('genre', 'Unknown')
        decade = c.get('decade', 'Unknown')
        director = c.get('director', None)
        
        if serialization_format == 1:
            title_part = title
        else:
            title_part = f"{title} ({year})" if year else title
            
        parts = [title_part]
        
        if serialization_format >= 3:
            if genre and genre != 'Unknown':
                parts.append(genre)
            if decade and decade != 'Unknown':
                parts.append(decade)
                
        if serialization_format >= 4:
            if director and director != 'Unknown':
                parts.append(director)
        
        candidate_lines.append(
            f"{i+1}. {' | '.join(parts)}"
        )

    candidate_block = "\n".join(candidate_lines)

    era_section = ""
    if era_hints:
        eras = ", ".join(era_hints)
        era_section = (f"\nDetected user era preference: "
                       f"{eras}. Strongly prefer candidates"
                       f" from this era.\n")

    messages = [
        {"role": "system", "content": "You are a movie recommendation reranker."},
        {"role": "user", "content": f"""Select the single best movie from the candidate list based on the dialogue.
Follow explicit user constraints strictly.
Prefer candidates that match genre, year/era, and semantic clues from the conversation.
Pay special attention to exclusions — if the user has already seen a movie, mentioned disliking it, or asked for something different, do NOT select that movie even if it appears in the candidate list.

Return only:
ANSWER: <number>
where <number> is the position in the list (1 to 50)
{era_section}
[Dialogue History]
{history}

[Candidate List]
{candidate_block}"""}
    ]
    return messages


def build_response_prompt(history: str, selected_movie: dict, reason_hints: str = None) -> str:
    """
    Build the prompt for natural response generation.
    """
    history = truncate_history(history, max_turns=5)
    if reason_hints is None:
        lines = history.split('\n')
        last_user_msg = ""
        for line in reversed(lines):
            if line.startswith("User:"):
                last_user_msg = line
                break
        
        if last_user_msg.startswith("User:"):
            last_user_msg = last_user_msg[5:].strip()
            
        reason_hints = last_user_msg[:50]
        if not reason_hints:
            reason_hints = "matches user preference from conversation"

    prompt = f"""You are a helpful and natural movie recommender.

Write one conversational response using the selected recommendation.

The reply must:
- mention the movie naturally,
- justify it with the user's preferences,
- sound concise and friendly,
- avoid mentioning unselected candidates.

Dialogue:
{history}

Selected recommendation:
{selected_movie['title']} | genres: {selected_movie.get('genre', 'unknown')} | decade: {selected_movie.get('decade', 'unknown')} | reason hints: {reason_hints}
"""
    return prompt