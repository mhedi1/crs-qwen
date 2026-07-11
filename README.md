# CRS-Qwen: Conversational Movie Recommender System

Bachelor's Thesis — Universidad de Jaén (Erasmus Programme)  
**Author:** Mohamed Hedi Foughali  
**Supervisors:** Prof. Luis Martínez López, Dr. Bapi Dutta  
**Academic Supervisor:** Dr. Ameni Youssfi (Pristini School of AI)

---

## Overview

CRS-Qwen is a two-stage conversational 
recommender system combining 
knowledge-graph-based candidate retrieval 
with zero-shot large language model 
reranking. The system integrates the KBRD 
retrieval model with Qwen3.5-35B-A3B for 
contextual reranking and natural language 
response generation.

### Results

| System | Reranker@1 |
|--------|-----------|
| Pure KBRD baseline (no reranker) | 0.0231 |
| KBRD + Fusion | 0.0228 |
| CRS-Qwen (full pipeline) | **0.0303** |

**+31.2% improvement** over the pure 
KBRD baseline on the ReDial test set 
(1,301 conversations, full test set).

---

## Architecture
User Input
↓
Stage 1: Entity Extraction +
KBRD Candidate Retrieval
↓
Stage 2: Qwen3.5-35B-A3B Reranking
↓
Stage 3: Natural Language Response
Generation
↓
Web Interface (Flask)


---

## Repository Structure
crs-qwen/
├── my_crs/
│ ├── config.yaml # Pipeline configuration
│ ├── evaluate.py # Offline evaluation
│ ├── kbrd_adapter.py # Stage 1: Entity extraction + KBRD
│ ├── prompts.py # Prompt templates
│ ├── recommender.py # Pipeline orchestrator
│ ├── reranker.py # Stage 2: Qwen reranking
│ ├── response_generator.py # Stage 3: Response generation
│ └── utils.py # Output parsing utilities
├── web_app/
│ ├── app.py # Flask web application
│ ├── templates/ # HTML templates
│ └── static/ # CSS and JS assets
├── experiments/ # Evaluation results
├── data/ # ReDial and INSPIRED datasets
├── baseline_repo/ # Original KBRD implementation
├── requirements.txt
└── .env.example


---

## Installation

### Prerequisites

- Python 3.10 or later
- Ollama with `qwen3.5:35b` model:

```bash
ollama pull qwen3.5:35b
```

- A valid TMDB API key from 
  https://www.themoviedb.org/settings/api

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/mhedi1/crs-qwen
cd crs-qwen

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download spaCy model
python -m spacy download en_core_web_sm

# 4. Download NLTK data
python -c "import nltk; nltk.download('punkt_tab')"

# 5. Configure environment
cp .env.example .env
# Edit .env with your TMDB_API_KEY 
# and FLASK_SECRET_KEY
```

---

## Usage

### Run Evaluation

```bash
python my_crs/evaluate.py \
    --format 1 \
    --dataset redial
```

### Run Web Application

```bash
python web_app/app.py
```

Open http://localhost:5000 in your browser.

---

## Evaluation

The system was evaluated on the ReDial 
benchmark and the INSPIRED out-of-domain 
dataset. Full results are reported in the 
thesis document.

**Human Evaluation (0-2 scale):**
- Fluency: 2.00
- Consistency: 2.00
- Informativeness: 1.55

---

## Thesis

The full thesis document is available 
in the repository as
TFG_Foughali_Mohamed_Hedi.pdf

---

## License

This project was developed as a 
Bachelor's thesis at the Universidad 
de Jaén. All rights reserved.