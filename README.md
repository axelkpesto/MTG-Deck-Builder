# MTG Deck Builder
An ML-powered Magic: The Gathering Commander deck generation system with:
- Card vector search and similarity
- Tag prediction for cards and custom vectors
- Commander deck generation with a graph neural network (GNN)
- Deck analysis utilities for color, curve, land ratio, and tag composition
- A production-ready Flask API (Cloud Run + Docker compatible)

## Why this project is interesting
This project combines multiple ML systems into one pipeline:
1. A vector database of MTG cards
2. A tagging model for card role prediction
3. A graph-based deck generator that samples full Commander decks
4. An API layer for real-time usage in apps/services

It is designed as both a research sandbox and an API backend you can deploy.

## Core components
- `vector_database.py`: vector storage, lookup, similarity search, persistence
- `tagging_model.py`: trains/loads a multi-label tag predictor
- `deckgen/`: GNN model + generation logic (`DeckGenBundle` entrypoint)
- `vector_db_server.py`: Flask API exposing vectors, tags, generation, and deck analysis
- `card_data/`: card/deck data models and analyzers
- `documentation/vector_db_server_api.json`: API spec

## Project structure
```text
MTG-Deck-Builder/
  card_data/              # Card/deck domain models + analyzers
  deckgen/                # Commander generation model + runtime
  datasets/               # Vector DB + graph + training datasets
  documentation/          # API schema docs
  firestore/              # API key auth integration
  models/                 # Saved ML checkpoints
  vector_db_server.py     # Flask API app
  vector_database.py      # Vector DB implementation
  tagging_model.py        # Tagging model training/inference
```

## Quickstart (local)
### 1. Requirements
- Python `3.12` (see `.python-version`)
- Pip
- Optional CUDA GPU for faster inference/training

### 2. Install
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure paths
Dataset/model paths are configured in `config/config.json`.
Default paths expect local files in `datasets/` and `models/`.

### 4. Run a local generation test
```bash
python test.py
```

## Using deck generation in Python
```python
import torch
from vector_database import VectorDatabase
from config import CONFIG
from deckgen import DeckGenBundle, DeckGenPaths

vd = VectorDatabase.load_static(CONFIG.datasets["VECTOR_DATABASE_PATH"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bundle = DeckGenBundle.load(paths=DeckGenPaths(), device=device, vector_db=vd)
deck_counts, stats = bundle.generate(commander_name="Atraxa, Praetors' Voice")
```

## API server
Run locally:
```bash
python vector_db_server.py
```
Server defaults:
- Host: `0.0.0.0`
- Port: `8080` (or `DEFAULT_PORT` env var)

Production entrypoints:
- `server.yaml`: `gunicorn -b :$PORT vector_db_server:app`
- `Dockerfile`: containerized Gunicorn runtime

### Public API routes
Public (no API key required):
- `GET /`
- `GET /help`
- `GET /examples`
- `GET /status`

Authenticated routes (API key required):
- `GET /get_vector/<v_id>`
- `GET /get_vector_description/<v_id>`
- `GET /get_random_vector`
- `GET /get_random_vector_description`
- `GET /get_similar_vectors/<v_id>?num_vectors=...`
- `GET /get_tags/<v_id>?threshold=...&top_k=...`
- `POST /get_tags_from_vector`
- `GET /generate_deck/<v_id>`
- `POST /analyze_deck`

Detailed schema: `documentation/vector_db_server_api.json`

## Environment variables
Common runtime variables:
- `FLASK_DEBUG`
- `DEFAULT_PORT`
- `REDIS_URL` (rate-limiter backend)
- `FIREBASE_API_KEY` / related auth config used by your Firestore key flow

## Testing
```bash
pytest -q
```

## Deployment
- Cloud Run config: `server.yaml`
- Container build: `Dockerfile`

## Data attribution
Card data originates from MTGJSON:
- https://mtgjson.com/downloads/all-files/
- https://mtgjson.com/data-models/set/
- https://mtgjson.com/data-models/card/card-set/

All source data rights remain with their respective owners/providers.