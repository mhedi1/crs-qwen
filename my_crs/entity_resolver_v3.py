import re
from typing import List, Tuple, Dict, Any
from my_crs import movie_catalogue

_MOVIE_CONTEXT_KEYWORDS = {
    "movie", "film", "watch", "watched", "watching", "see", "saw", "seen",
    "love", "loved", "like", "liked", "hate", "hated", "favorite", "favourite",
    "recommend", "recommended", "suggest", "suggested", "good", "great", "bad",
    "awesome", "terrible", "boring", "funny", "scary", "sad", "comedy", "action",
    "horror", "drama", "thriller", "romance", "sci-fi", "fantasy", "documentary"
}

class ResolverV3:
    def __init__(self):
        items = movie_catalogue.get_all_movie_items()
        
        self._title_map: Dict[str, List[Dict[str, Any]]] = {}
        for entity_id, title, popularity, year in items:
            norm_title = self._normalize_title(title)
            if not norm_title:
                continue
                
            entry = {
                "entity_id": entity_id,
                "original_title": title,
                "popularity": popularity,
                "year": year
            }
            if norm_title not in self._title_map:
                self._title_map[norm_title] = []
            self._title_map[norm_title].append(entry)
            
        # Sort titles by length (descending) to ensure longest-first match
        self._sorted_titles = sorted(self._title_map.keys(), key=len, reverse=True)
        
    def _normalize_title(self, text: str) -> str:
        t = re.sub(r'[^\w\s]', '', text.lower())
        return re.sub(r'\s+', ' ', t).strip()

    def _normalize_dialogue_with_map(self, text: str) -> Tuple[str, List[int]]:
        t_lower = text.lower()
        norm_chars = []
        idx_map = []
        
        for i, c in enumerate(t_lower):
            if re.match(r'[\w\s]', c):
                norm_chars.append(c)
                idx_map.append(i)
                
        temp_str = ''.join(norm_chars)
        final_chars = []
        final_map = []
        in_space = False
        
        for i, c in enumerate(temp_str):
            if c.isspace():
                if not in_space:
                    final_chars.append(' ')
                    final_map.append(idx_map[i])
                    in_space = True
            else:
                final_chars.append(c)
                final_map.append(idx_map[i])
                in_space = False
                
        # Strip trailing space if any
        if final_chars and final_chars[-1] == ' ':
            final_chars.pop()
            final_map.pop()
        # Strip leading space if any
        if final_chars and final_chars[0] == ' ':
            final_chars.pop(0)
            final_map.pop(0)
            
        return ''.join(final_chars), final_map

    def resolve_mentions(self, dialogue: str, doc) -> Tuple[List[int], List[str], List[Dict[str, Any]]]:
        norm_dialogue, idx_map = self._normalize_dialogue_with_map(dialogue)
        
        found_entity_ids = []
        found_descriptions = []
        found_metadata = []
        matched_spans = []
        
        for norm_title in self._sorted_titles:
            # Exact match using word boundaries
            pattern = r'\b' + re.escape(norm_title) + r'\b'
            for match in re.finditer(pattern, norm_dialogue):
                start, end = match.span()
                
                # Check overlap with longer matched titles
                overlap = False
                for m_start, m_end in matched_spans:
                    if not (end <= m_start or start >= m_end):
                        overlap = True
                        break
                
                if overlap:
                    continue
                    
                candidates = self._title_map[norm_title]
                
                if not self._passes_structural_filter(norm_title, doc):
                    continue
                    
                matched_spans.append((start, end))
                best_candidate, resolution_reason = self._resolve_collision(candidates, dialogue)
                
                found_entity_ids.append(best_candidate["entity_id"])
                prov_tag = f"[V3: {resolution_reason}] {best_candidate['original_title']}"
                found_descriptions.append(prov_tag)
                
                # Retrieve actual exact character match from dialogue using the map
                orig_start = idx_map[start]
                orig_end = idx_map[end - 1] + 1
                surface_text = dialogue[orig_start:orig_end]
                
                found_metadata.append({
                    "entity_id": best_candidate["entity_id"],
                    "surface_text": surface_text,
                    "start_char": orig_start,
                    "end_char": orig_end,
                    "provenance": prov_tag
                })
                
        return found_entity_ids, found_descriptions, found_metadata

    def _passes_structural_filter(self, norm_title: str, doc) -> bool:
        words = norm_title.split()
        if len(words) > 1:
            return True
            
        if len(norm_title) < 2:
            return False
            
        # Apply structural checks for ALL single-word titles
        for token in doc:
            if token.text.lower() == norm_title:
                if self._is_valid_title_token(token, doc):
                    return True
        return False
        
    def _is_valid_title_token(self, token, doc) -> bool:
        start_idx = max(0, token.i - 4)
        end_idx = min(len(doc), token.i + 5)
        
        # Priority 1: Explicit year nearby
        for i in range(start_idx, end_idx):
            if re.match(r"^\(?(19|20)\d{2}\)?$", doc[i].text):
                return True
                
        # Priority 2: Explicit capitalization mid-sentence
        if token.is_title and not token.is_sent_start:
            return True
            
        # Priority 3: Tagged as Proper Noun
        if token.pos_ == "PROPN":
            return True
            
        # If it reaches here, it's lowercase (or start of sentence), no year, not a proper noun.
        # We reject it if it's acting as a grammatical word or verb.
        is_grammar_or_verb = token.pos_ in ("PRON", "DET", "ADP", "PART", "VERB", "AUX", "SCONJ", "CCONJ", "INTJ", "PRON")
        
        if is_grammar_or_verb:
            return False
            
        # If it's a normal noun/adjective (like "alien", "jumanji"), we still require movie context
        context_words = [doc[i].text.lower() for i in range(start_idx, end_idx) if i != token.i]
        has_movie_context = any(w in _MOVIE_CONTEXT_KEYWORDS for w in context_words)
        
        if has_movie_context:
            return True
            
        return False

    def _resolve_collision(self, candidates: List[Dict[str, Any]], dialogue: str) -> Tuple[Dict[str, Any], str]:
        if len(candidates) == 1:
            return candidates[0], "Exact"
            
        for c in candidates:
            if c["year"] and str(c["year"]) in dialogue:
                return c, "Year-Disambiguated"
                
        sorted_candidates = sorted(candidates, key=lambda x: x["popularity"], reverse=True)
        return sorted_candidates[0], "Deterministic Fallback"

_v3_resolver = None

def resolve_mentions(dialogue: str, doc) -> Tuple[List[int], List[str]]:
    global _v3_resolver
    if _v3_resolver is None:
        _v3_resolver = ResolverV3()
    return _v3_resolver.resolve_mentions(dialogue, doc)
