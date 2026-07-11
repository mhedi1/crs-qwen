# CRS-Qwen: Conversational Movie Recommender System

Bachelor's Thesis — Universidad de Jaén (Erasmus Programme)  
**Author:** Mohamed Hedi Foughali  
**Supervisors:** Prof. Luis Martínez López, Dr. Bapi Dutta  
**Academic Supervisor:** Dr. Ameni Youssfi (Pristini School of AI)

---

## 📖 Overview

CRS-Qwen is a two-stage conversational recommender system combining knowledge-graph-based candidate retrieval with zero-shot large language model reranking. The system integrates the KBRD retrieval model with `Qwen3.5-35B-A3B` for contextual reranking and natural language response generation.

### 🏆 Results

| System | Reranker@1 |
|--------|-----------|
| Pure KBRD baseline (no reranker) | 0.0231 |
| KBRD + Fusion | 0.0228 |
| **CRS-Qwen (full pipeline)** | **0.0303** |

**+31.2% improvement** over the pure KBRD baseline on the ReDial test set (1,301 conversations, full test set).

---

## ⚙️ Architecture

The system operates in three main stages:

1. **Entity Extraction & Candidate Retrieval:** Extracts entities from user dialogue and queries the KBRD knowledge graph.
2. **LLM Semantic Reranking:** Uses Qwen3.5-35B to evaluate and rank the KBRD candidates contextually.
3. **Response Generation:** Generates a natural, conversational response recommending the top-ranked movie.

---

## 📁 Repository Structure

```text
crs-qwen/
├── my_crs/
│   ├── config.yaml              # Pipeline configuration
│   ├── evaluate.py              # Offline evaluation script
│   ├── kbrd_adapter.py          # Stage 1: Entity extraction + KBRD
│   ├── prompts.py               # Prompt templates
│   ├── recommender.py           # Pipeline orchestrator
│   ├── reranker.py              # Stage 2: Qwen reranking
│   ├── response_generator.py    # Stage 3: Response generation
│   └── utils.py                 # Output parsing utilities
├── web_app/
│   ├── app.py                   # Flask web application
│   ├── templates/               # HTML templates (Premium UI)
│   └── static/                  # CSS and JS assets
├── experiments/                 # Evaluation results & JSON logs
├── data/                        # ReDial and INSPIRED datasets
├── baseline_repo/               # Original KBRD implementation
├── requirements.txt             # Python dependencies
└── .env.example                 # Environment variables template
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed with the `qwen3.5:35b` model:
  ```bash
  ollama pull qwen3.5:35b
  ```
- A valid TMDB API key from [The Movie Database](https://www.themoviedb.org/settings/api).

### Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/mhedi1/crs-qwen.git
cd crs-qwen

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download required NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt_tab')"

# 5. Configure environment variables
cp .env.example .env
# Edit .env and insert your TMDB_API_KEY and a FLASK_SECRET_KEY
```

---

## 💻 Usage

### Run the Web Application (Demo)

Launch the premium Flask web interface:
```bash
python web_app/app.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser to interact with the system.

### Run Offline Evaluation

Evaluate the pipeline on the ReDial dataset:
```bash
python my_crs/evaluate.py --format 1 --dataset redial
```

---

## 📊 Evaluation Metrics

The system was evaluated on the ReDial benchmark and the INSPIRED out-of-domain dataset. Full detailed results, including ablation studies, are reported in the thesis document.

**Human Evaluation (0-2 scale):**
- **Fluency:** 2.00
- **Consistency:** 2.00
- **Informativeness:** 1.55

---

## 🎓 Thesis Document

The full thesis document detailing the research, methodology, and comprehensive evaluation is available in this repository:
📄 [**TFG_Foughali_Mohamed_Hedi.pdf**](./TFG_Foughali_Mohamed_Hedi.pdf)

---

## 📄 License

This project was developed as a Bachelor's thesis at the Universidad de Jaén. All rights reserved.