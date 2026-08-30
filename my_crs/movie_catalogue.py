import os
import csv
import pickle
import re
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
KBRD_REPO_PATH = os.path.normpath(
    os.path.join(CURRENT_DIR, "..", "baseline_repo", "KBRD_project", "KBRD")
)

_entity_to_title = {}
_entity_to_uri = {}
_entity_to_mentions = {}
_entity_to_year = {}
_catalogue_loaded = False

def _clean_title(entity_uri: str) -> str:
    match = re.search(r"resource/(.+)>", str(entity_uri))
    title = match.group(1) if match else str(entity_uri)
    title = title.replace("_", " ")
    cleaned = re.sub(r"\s*\(.*?\)", "", title).strip()
    return cleaned if cleaned else title.strip()

def _clean_csv_title(title: str) -> str:
    title = title.replace("_", " ")
    cleaned = re.sub(r"\s*\(.*?\)", "", title).strip()
    return cleaned if cleaned else title.strip()

def load_catalogue():
    global _catalogue_loaded
    if _catalogue_loaded:
        return
    
    data_dir = os.path.join(KBRD_REPO_PATH, "data", "redial")
    
    with open(os.path.join(data_dir, "entity2entityId.pkl"), "rb") as f:
        e2id = pickle.load(f)
        
    id2e = {v: k for k, v in e2id.items()}
    
    with open(os.path.join(data_dir, "movie_ids.pkl"), "rb") as f:
        mids = pickle.load(f)
        
    with open(os.path.join(data_dir, "movies_with_mentions.csv"), encoding="utf-8") as f:
        csv_movies = {int(r["movieId"]): r["movieName"] for r in csv.DictReader(f)}
        
    # Load training popularity to avoid leakage
    train_mentions = {}
    with open(os.path.join(data_dir, "train_data.jsonl"), encoding="utf-8") as f:
        for line in f:
            conv = json.loads(line)
            for msg in conv.get("messages", []):
                text = msg.get("text", "")
                for mid_str in re.findall(r"@(\d+)", text):
                    train_mentions[int(mid_str)] = train_mentions.get(int(mid_str), 0) + 1
        
    for mid in mids:
        # Mentions calculation
        popularity = 0
        
        uri = id2e.get(mid)
        if uri and isinstance(uri, str) and 'dbpedia' in str(uri):
            _entity_to_uri[mid] = uri
            _entity_to_title[mid] = _clean_title(uri)
            
            # Extract year from URI
            year_match = re.search(r"\((\d{4})_film\)", uri)
            _entity_to_year[mid] = int(year_match.group(1)) if year_match else None
            
            # Popularity mapping uses the ReDial integer IDs mapped to this entity ID
            redial_ids = [k for k, v in e2id.items() if v == mid and isinstance(k, int)]
            for rid in redial_ids:
                popularity += train_mentions.get(rid, 0)
        else:
            redial_ids = [k for k, v in e2id.items() if v == mid and isinstance(k, int)]
            if redial_ids:
                redial_id = redial_ids[0]
                if redial_id in csv_movies:
                    csv_name = csv_movies[redial_id]
                    _entity_to_title[mid] = _clean_csv_title(csv_name)
                    _entity_to_uri[mid] = ""
                    
                    # Extract year from CSV name
                    year_match = re.search(r"\((\d{4})\)", csv_name)
                    _entity_to_year[mid] = int(year_match.group(1)) if year_match else None
                else:
                    _entity_to_title[mid] = "Unknown Title"
                    _entity_to_uri[mid] = ""
                    _entity_to_year[mid] = None
                
                for rid in redial_ids:
                    popularity += train_mentions.get(rid, 0)
            else:
                _entity_to_title[mid] = "Unknown Title"
                _entity_to_uri[mid] = ""
                _entity_to_year[mid] = None
                
        _entity_to_mentions[mid] = popularity
                
    _catalogue_loaded = True

def get_title(entity_id: int) -> str:
    if not _catalogue_loaded:
        load_catalogue()
    return _entity_to_title.get(entity_id)

def get_uri(entity_id: int) -> str:
    if not _catalogue_loaded:
        load_catalogue()
    return _entity_to_uri.get(entity_id)

def get_all_movies():
    """Returns a dict mapping entity_id to title for all valid KBRD movies."""
    if not _catalogue_loaded:
        load_catalogue()
    return _entity_to_title.copy()

def get_popularity(entity_id: int) -> int:
    if not _catalogue_loaded:
        load_catalogue()
    return _entity_to_mentions.get(entity_id, 0)

def get_year(entity_id: int):
    if not _catalogue_loaded:
        load_catalogue()
    return _entity_to_year.get(entity_id)

def get_all_movie_items():
    if not _catalogue_loaded:
        load_catalogue()
    items = []
    for mid, title in _entity_to_title.items():
        if title and title != "Unknown Title":
            items.append((mid, title, _entity_to_mentions.get(mid, 0), _entity_to_year.get(mid)))
    return items
