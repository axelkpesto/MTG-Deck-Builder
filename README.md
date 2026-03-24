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
- `backend/vector_database.py`: vector storage, lookup, similarity search, persistence
- `backend/ml/tagging_model.py`: trains/loads a multi-label tag predictor
- `backend/deckgen/`: GNN model + generation logic (`DeckGenBundle` entrypoint)
- `backend/api/vector_db_server.py`: canonical Flask API module
- `backend/workers/deckgen_worker.py`: Redis-backed background worker for deck jobs
- `backend/card_data/`: card/deck data models and analyzers
- `backend/documentation/vector_db_server_api.json`: API spec

## Project structure
```text
MTG-Deck-Builder/
  backend/
    api/                  # Flask API modules
    workers/              # Background job consumers
    card_data/            # Card/deck domain models + analyzers
    deckgen/              # Commander generation model + runtime
    data/
      raw/                # Source datasets
      processed/          # Generated runtime datasets
      models/             # Saved ML checkpoints
    documentation/        # API schema docs
    firestore/            # API key auth integration
    ml/                   # ML model training/inference modules
    infra/                # Docker and deploy config
    vector_database.py    # Vector DB implementation
```

## Quickstart (local)
### 1. Requirements
- Python `3.12`
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
pip install -r backend/requirements.txt
```

### 3. Configure paths
Dataset/model paths are configured in `backend/config/config.json`.
Default paths expect local files in `backend/data/`.

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
python backend/api/vector_db_server.py
```
Server defaults:
- Host: `0.0.0.0`
- Port: `8080` (or `DEFAULT_PORT` env var)

Production entrypoints:
- `backend/infra/server.yaml`: `gunicorn -b :$PORT backend.api.vector_db_server:app`
- `backend/infra/Dockerfile`: containerized Gunicorn runtime

### Public API routes
Public (no API key required):
- `GET /`
- `GET /help`
- `GET /examples`
- `POST /status`

Authenticated routes (API key required):
- `POST /get_vector` with body `{"id":"Card Name"}`
- `POST /get_vector_description` with body `{"id":"Card Name"}`
- `POST /get_vector_descriptions` with body `{"cards":["Card Name", "..."]}`
- `POST /get_random_vector`
- `POST /get_random_vector_description`
- `POST /get_similar_vectors` with body `{"id":"Card Name","num_vectors":5}`
- `POST /get_tags` with body `{"id":"Card Name","threshold":0.5,"top_k":8}`
- `POST /get_tag_list` with body `{"cards":["Card Name", "..."],"threshold":0.5,"top_k":8}`
- `POST /get_tags_from_vector`
- `POST /generate_deck` with body `{"id":"Commander Name"}`
- `POST /analyze_deck`

Detailed schema: `backend/documentation/vector_db_server_api.json`

## Environment variables
Common runtime variables:
- `FLASK_DEBUG`
- `DEFAULT_PORT`
- `API_KEY` / API Key for Firebase Auth Check

## Public API Endpoint
https://mtg-deckbuilder-api-891777334325.us-west2.run.app
Please request an API key if interested in using API

## Deployment
- Cloud Run config: `backend/infra/server.yaml`
- Container build: `backend/infra/Dockerfile`

## Data attribution
Card data originates from MTGJSON and CommanderSpellbook:
- https://mtgjson.com/downloads/all-files/
- https://mtgjson.com/data-models/set/
- https://mtgjson.com/data-models/card/card-set/
- https://json.commanderspellbook.com/variants.json

And a special thanks to the https://moxfield.com/ team for allowing data collection through their API. 

All source data rights remain with their respective owners/providers.
