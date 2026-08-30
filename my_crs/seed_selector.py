from typing import List, Tuple, Dict, Any
import re
from my_crs import movie_catalogue

def apply_selection(dialogue: str, seed_list: List[int], seed_metadata: List[Dict[str, Any]], policy: str, movie_ids: set) -> Tuple[List[int], List[int], Dict[str, Any]]:
    """
    Applies a configurable seed selection policy BETWEEN extraction and KBRD candidate generation.
    """
    diagnostics = {
        "seed_selection_policy": policy,
        "num_seeds_before_selection": len(seed_list),
        "num_seeds_after_selection": len(seed_list),
        "removed_seed_ids": [],
        "selected_movie_seed_ids": [],
        "selected_movie_seed_positions": []
    }
    
    if policy == "all":
        # Strictly preserve exact list, no deduplication, no reordering
        return seed_list, [], diagnostics
        
    # Map entity_id to its most recent start_char
    metadata_map = {}
    for meta in seed_metadata:
        eid = meta["entity_id"]
        # In case of duplicates, keep the highest start_char (most recent)
        if eid not in metadata_map or meta["start_char"] > metadata_map[eid]["start_char"]:
            metadata_map[eid] = meta
            
    removed_seed_ids = []
    
    # Identify which IDs are movie entities and map them to their best position
    movie_candidates = []
    for eid in seed_list:
        if eid in movie_ids:
            if eid not in [c[0] for c in movie_candidates]:
                pos = metadata_map[eid]["start_char"] if eid in metadata_map else -1
                movie_candidates.append((eid, pos))
                
    surviving_movie_ids = set()
    
    if policy in ("recent_3", "recent_5"):
        k = int(policy.split("_")[1])
        
        # Sort by position descending (most recent first)
        movie_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Take top k
        surviving_movie_ids = set([c[0] for c in movie_candidates[:k]])
        
        # Removed ones are those that didn't make the cut
        removed_seed_ids = [c[0] for c in movie_candidates[k:]]
        
    elif policy == "no_contextual_year_titles":
        # Suppress numeric movie titles that are structurally just release years
        for eid, _ in movie_candidates:
            title = movie_catalogue.get_title(eid)
            if title and title.isdigit():
                # Check occurrences in dialogue
                occurrences = [m.span() for m in re.finditer(r'\b' + re.escape(title) + r'\b', dialogue.lower())]
                is_contextual_year = False
                
                if occurrences:
                    all_are_years = True
                    for start, end in occurrences:
                        padded = " " * 15 + dialogue.lower() + " " * 15
                        p_start = start + 15
                        p_end = end + 15
                        
                        before = padded[p_start-15:p_start]
                        after = padded[p_end:p_end+5]
                        
                        if '(' in before and ')' in after:
                            pass
                        elif re.search(r'\b(in|of|from|released in)\s*$', before):
                            pass
                        else:
                            all_are_years = False
                            break
                    if all_are_years:
                        is_contextual_year = True
                        
                if is_contextual_year:
                    removed_seed_ids.append(eid)
                else:
                    surviving_movie_ids.add(eid)
            else:
                surviving_movie_ids.add(eid)
    else:
        # Unknown policy, fallback to all
        surviving_movie_ids = set([c[0] for c in movie_candidates])
        
    # Reconstruct the list preserving EXACT relative order, including duplicates.
    new_seed_list = []
    seen_removed = set()
    for eid in seed_list:
        if eid in movie_ids:
            if eid in surviving_movie_ids:
                new_seed_list.append(eid)
            else:
                # Removed
                if eid not in seen_removed:
                    seen_removed.add(eid)
        else:
            # Non-movie entities are preserved perfectly
            new_seed_list.append(eid)
            
    diagnostics["num_seeds_after_selection"] = len(new_seed_list)
    diagnostics["removed_seed_ids"] = list(set(removed_seed_ids))
    diagnostics["selected_movie_seed_ids"] = list(surviving_movie_ids)
    
    # Store positions for surviving movie ids for diagnostics
    positions = []
    for eid in surviving_movie_ids:
        pos = metadata_map[eid]["start_char"] if eid in metadata_map else -1
        positions.append(pos)
    diagnostics["selected_movie_seed_positions"] = positions
    
    return new_seed_list, list(set(removed_seed_ids)), diagnostics
