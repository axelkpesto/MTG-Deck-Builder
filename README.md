# MTG Deck Builder
An ML-powered Magic: The Gathering Commander deck builder with a full-stack web frontend. Features:
- Card vector search and similarity
- ML tag prediction for card role classification
- Commander deck generation with a graph neural network (GNN)
- Deck analysis utilities for color, curve, land ratio, and tag composition
- A PHP/Apache frontend with OAuth login, deck save/load, and card image support
- A production-ready Flask API (Cloud Run + Docker compatible)

## Why this project is interesting
This project combines multiple ML systems into one pipeline:
1. A vector database of MTG cards (embeddings + metadata)
2. A multi-label tagging model for card role prediction (Ramp, Draw, Removal, Combo, etc.)
3. A graph-based deck generator that samples full 100-card Commander decks
4. A PHP frontend that drives the full deck-building experience
5. An API layer for real-time usage in apps/services

## Core components

### Backend
- `backend/vector_database.py`: vector storage, case-insensitive lookup, similarity search, persistence
- `backend/ml/tagging_model.py`: trains/loads a multi-label tag predictor
- `backend/deckgen/`: GNN model + generation logic (`DeckGenBundle` entrypoint)
- `backend/api/vector_db_server.py`: canonical Flask API module
- `backend/card_data/`: card/deck data models and analyzers
- `backend/documentation/vector_db_server_api.json`: API spec

### Frontend
- `frontend/public/app.js`: single-page deck builder UI (stack view, list view, import, analysis)
- `frontend/public/card_images.php`: serves card image URLs from Scryfall
- `frontend/public/api.php`: proxies authenticated requests to the Flask backend
- `frontend/public/decks.php`: saved deck CRUD (Firestore-backed)
- `frontend/config.php`: shared PHP config, session management, env helpers
- `frontend/Dockerfile`: PHP 8.3 + Apache container

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
  frontend/
    public/               # PHP + static assets served by Apache
    config.php            # Shared PHP config and session helpers
    Dockerfile            # PHP 8.3 + Apache container image
    start.sh              # Container startup script
```

## Quickstart (local)

### Backend

#### 1. Requirements
- Python `3.12`
- Pip
- Optional CUDA GPU for faster inference/training

#### 2. Install
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r backend/requirements.txt
```

#### 3. Configure paths
Dataset/model paths are configured in `backend/config/config.json`.
Default paths expect local files in `backend/data/`.

#### 4. Run a local generation test
```bash
python test.py
```

### Frontend

#### Requirements
- PHP 8.3 + Apache (or Docker)
- The env vars listed below set in your shell or `.env.local`

#### Run with Docker
```bash
docker build -t mtg-frontend ./frontend
docker run -p 8000:8080 --env-file .env.local mtg-frontend
```

#### Run without Docker
Point Apache at `frontend/public/` as the document root and ensure the env vars are available to PHP (`SetEnv` in your Apache config or exported in the shell).

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
python -m backend.api.vector_db_server
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
- `POST /get_vector` — `{"id":"Card Name"}`
- `POST /get_vector_description` — `{"id":"Card Name"}`
- `POST /get_vector_descriptions` — `{"cards":["Card Name", "..."]}`
- `POST /get_random_vector`
- `POST /get_random_vector_description`
- `POST /get_similar_vectors` — `{"id":"Card Name","num_vectors":5}`
- `POST /get_tags` — `{"id":"Card Name","threshold":0.5,"top_k":8}`
- `POST /get_tag_list` — `{"cards":["Card Name", "..."],"threshold":0.5,"top_k":8}`
- `POST /get_tags_from_vector`
- `POST /generate_deck` — `{"id":"Commander Name"}`
- `POST /analyze_deck`

Detailed schema: `backend/documentation/vector_db_server_api.json`

## Environment variables

### Backend
| Variable | Description |
|---|---|
| `FLASK_DEBUG` | Enable Flask debug mode |
| `DEFAULT_PORT` | API server port (default `8080`) |
| `AUTHENTICATE` | Set to `0` to disable API key auth locally |
| `REDIS_URL` | Redis connection string for background workers |
| `API_KEY_PEPPER` | HMAC pepper for API key verification |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account JSON (local only) |

### Frontend
| Variable | Description |
|---|---|
| `FLASK_API_PATH` | Backend API base URL |
| `FIREBASE_API_KEY` | Global API key sent with backend requests |
| `OAUTH_CLIENT_ID` | Google OAuth client ID |
| `OAUTH_CLIENT_SECRET` | Google OAuth client secret |
| `OAUTH_REDIRECT_URI` | OAuth callback URL |
| `OAUTH_AUTHORIZE_URL` | OAuth authorization endpoint |
| `OAUTH_TOKEN_URL` | OAuth token exchange endpoint |
| `OAUTH_USERINFO_URL` | OAuth user info endpoint |
| `GCS_DATA_BUCKET` | GCS bucket name; if set, `start.sh` downloads `card_images.json` on startup |
| `CARD_IMAGES_PATH` | Override local path to `card_images.json` (optional; defaults to `/var/www/data/card_images.json`) |

### Card image resolution
`card_images.php` resolves card images in priority order:
1. Local dataset at `CARD_IMAGES_PATH` (populated from GCS in production)
2. Scryfall collection API — used automatically when the local dataset is absent (local dev)

## Importing decks
The deck builder accepts card lists in common formats:
```
1xAtraxa, Praetors' Voice
1x Sol Ring
1 Arcane Signet
Doubling Season x1
```

## Deck board
Cards are grouped into strategy columns (Ramp, Draw, Removal, Combo, etc.) with quantity badges. Hovering over a column title shows a short description of that strategy. The view can be toggled between stacked card art and a flat list.

## Deployment
- Cloud Run config: `backend/infra/server.yaml`
- Backend container: `backend/infra/Dockerfile`
- Frontend container: `frontend/Dockerfile`

## Public API endpoint
`https://mtg-deck-builder-api-891777334325.us-west1.run.app`

Please request an API key if you are interested in using the API directly.

## Public Web Endpoint:
`https://mtg-deck-builder-frontend-891777334325.us-west1.run.app`

## Data attribution
Card data originates from MTGJSON and CommanderSpellbook:
- https://mtgjson.com/downloads/all-files/
- https://mtgjson.com/data-models/set/
- https://mtgjson.com/data-models/card/card-set/
- https://json.commanderspellbook.com/variants.json

Card images are provided by the [Scryfall API](https://scryfall.com/docs/api) under their [terms of use](https://scryfall.com/docs/terms).

Special thanks to the [Moxfield](https://moxfield.com/) team for allowing data collection through their API.

All source data rights remain with their respective owners and providers.
