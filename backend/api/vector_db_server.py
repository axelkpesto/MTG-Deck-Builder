"""Flask server exposing vector database and deck generation endpoints."""
import json
import logging
import os
import threading
import time
import types
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from google.api_core.exceptions import GoogleAPICallError, RetryError

from backend.card_data import CardDecoder, SimpleDeck, SimpleDeckAnalyzer
from backend.config import CONFIG
from backend.deckgen import DeckGenBundle, DeckGenPaths
from backend.firestore.firestore_connector import authenticate_api_key, touch_last_used
from backend.ml.tagging_model import load_model, predicted_scores_from_probabilities
from backend.vector_database import VectorDatabase

app = Flask(__name__)
load_dotenv()
CORS(app)
app.config["DEBUG"] = bool(int(os.environ.get("FLASK_DEBUG", 0)))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector-db-server")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model: Any | None = None
class_names: list[str] = []
try:
    model, class_names = load_model(CONFIG.models["TAGGING_MODEL_PATH"])
    model.to(device).eval()
except (FileNotFoundError, RuntimeError, ValueError, KeyError, OSError) as e:
    print(f"Error loading tagging model: {e}")

vd = VectorDatabase(None, CardDecoder())
vd.load(CONFIG.datasets["VECTOR_DATABASE_PATH"])
with open(CONFIG.datasets["TAGS_DATASET_PATH"], "r", encoding="utf-8") as f:
    tag_dataset = json.load(f)

_deckgen = types.SimpleNamespace(bundle=None, state="loading")


def _warmup_node_embeddings(bundle: DeckGenBundle) -> None:
    """Pre-compute node embeddings in the background to reduce first-request latency.

    Args:
        bundle: Loaded DeckGenBundle whose node embeddings will be computed.

    Returns:
        None
    """
    if bundle.node_embeddings is not None:
        return

    if device.type != "cuda":
        logger.info("Skipping deck generation warmup on CPU device.")
        return

    warmup_enabled = bool(int(os.environ.get("DECKGEN_WARMUP", "1")))
    if not warmup_enabled:
        logger.info("Deck generation warmup disabled.")
        return

    try:
        bundle.get_node_embeddings()
        logger.info("Deck generation embeddings warmed.")
    except torch.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        logger.warning("Deck generation warmup skipped after CUDA OOM: %s", e)
    except (RuntimeError, ValueError, OSError) as e:
        logger.warning("Deck generation warmup failed: %s", e)


def _load_deckgen() -> None:
    """Load the DeckGenBundle into the global `gen` variable in a background thread.

    Args:
        None

    Returns:
        None
    """
    logger.info("DeckGenBundle load started (device=%s)", device)
    try:
        bundle = DeckGenBundle.load(paths=DeckGenPaths(), device=str(device), vector_db=vd)
        _deckgen.bundle = bundle
        _deckgen.state = "ready"
        logger.info("DeckGenBundle ready.")
        _warmup_node_embeddings(bundle)
    except Exception as e:  # pylint: disable=broad-except
        _deckgen.state = "failed"
        logger.exception("DeckGenBundle failed to load (%s): %s", type(e).__name__, e)


threading.Thread(target=_load_deckgen, daemon=True).start()


DEFAULT_RATE_LIMIT = "120 per minute"

limiter = Limiter(
    app=app,
    key_func=lambda: str(getattr(g, "api_key_id", get_remote_address())),
    default_limits=[DEFAULT_RATE_LIMIT],
    storage_uri=os.environ["REDIS_URL"],
)

auth_enabled = bool(int(os.environ.get("AUTHENTICATE", 1)))

def error(message: str, status: int = 400):
    """Return a JSON error response with the given message and HTTP status.

    Args:
        message: Human-readable error description.
        status: HTTP status code to return.

    Returns:
        Flask JSON response tuple (response, status_code).
    """
    return jsonify({"error": message}), status

def clamp_int(x: int, lo: int, hi: int) -> int:
    """Clamp an integer to the inclusive range [lo, hi].

    Args:
        x: Value to clamp.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        Integer clamped to [lo, hi].
    """
    return max(lo, min(hi, x))

def clamp_float(x: float, lo: float, hi: float) -> float:
    """Clamp a float to the inclusive range [lo, hi].

    Args:
        x: Value to clamp.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        Float clamped to [lo, hi].
    """
    return max(lo, min(hi, x))


def resolve_card_id(v_id: str) -> str:
    """Resolve a raw card name string to a canonical vector database key.

    Args:
        v_id: Raw card name from the request, possibly mis-capitalized.

    Returns:
        Canonical card name present in the vector database.

    Raises:
        KeyError: If no matching card is found after all resolution attempts.
    """
    raw = v_id.strip()
    if not raw:
        raise KeyError(v_id)
    if raw in vd:
        return raw

    formatted = format_id(raw)
    if formatted in vd:
        return formatted

    return vd.find_id(raw)


def parse_card_list_payload(data: dict[str, Any]) -> list[str]:
    """Extract and validate the 'cards' list from a JSON request payload.

    Args:
        data: Parsed JSON body from the request.

    Returns:
        Non-empty list of stripped card name strings.

    Raises:
        ValueError: If 'cards' is missing, not a list of strings, or empty after stripping.
    """
    cards = data.get("cards")
    if not isinstance(cards, list) or not all(isinstance(c, str) for c in cards):
        raise ValueError("JSON body must include 'cards': ['Card Name', ...]")
    cleaned = [c.strip() for c in cards if c.strip()]
    if not cleaned:
        raise ValueError("cards list cannot be empty")
    return cleaned


def parse_required_card_id(data: dict[str, Any], field_name: str = "id") -> str:
    """Extract and validate a required string field from a JSON request payload.

    Args:
        data: Parsed JSON body from the request.
        field_name: Key to look up in the payload.

    Returns:
        Stripped non-empty string value for the field.

    Raises:
        ValueError: If the field is missing, not a string, or blank after stripping.
    """
    value = data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"JSON body must include '{field_name}': 'Card Name'")
    return value.strip()


def predict_tags_for_card(card_id: str, threshold: float, top_k: int) -> dict[str, Any]:
    """Look up a card's vector and run tag prediction on it.

    Args:
        card_id: Canonical card name in the vector database.
        threshold: Minimum probability to include a tag in predicted output.
        top_k: Maximum number of top-scoring tags to include in the scores list.

    Returns:
        Dict with 'predicted', 'predicted_scores', 'scores', and 'threshold' keys.
    """
    vector = vd.get(card_id)
    vec_np = VectorDatabase.vector_to_numpy(vector)
    result = predict_from_vector(vec_np, threshold, top_k)
    return result

def get_api_key_from_request(req) -> str | None:
    """Extract an API key from a request's Authorization or X-API-KEY header.

    Args:
        req: Flask request object.

    Returns:
        API key string, or None if no key is present.
    """
    auth = req.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return req.headers.get("X-API-KEY")

def set_limit() -> str:
    """Return the rate limit string for the current request's API key tier.

    Args:
        None

    Returns:
        Rate limit string in Flask-Limiter format (e.g. '120 per minute').
    """
    if hasattr(g, "rate_limit") and g.rate_limit:
        rl = g.rate_limit
        if rl == "unlimited":
            return DEFAULT_RATE_LIMIT
        if "/" in rl:
            split = rl.split("/")
            return f"{split[0]} per {split[1]}"
        return rl
    return DEFAULT_RATE_LIMIT

@app.before_request
def _start_timer():
    """Record the request start time for latency logging.

    Args:
        None

    Returns:
        None
    """
    g.start_time = time.time()

@app.before_request
def _authenticate_api_key():
    """Validate the API key on every request that requires authentication.

    Args:
        None

    Returns:
        None on success, or a JSON error response on auth failure.
    """
    if not auth_enabled:
        return None

    if request.path in ["/", "/help", "/examples", "/status"]:
        return None

    api_key = get_api_key_from_request(request)
    if not api_key:
        return error("missing API key", 401)

    if api_key == g.get("last_raw"):
        return None

    info = authenticate_api_key(api_key)
    if not info:
        return error("invalid or expired API key", 403)

    g.api_key_id = info["api_key_id"]
    g.user_id = info["user_id"]
    g.rate_limit = info["rate_limit"]
    g.last_raw = api_key

    try:
        touch_last_used(g.api_key_id)
    except (GoogleAPICallError, RetryError, ValueError, RuntimeError):
        pass
    return None

@app.after_request
def _log_request(resp):
    """Log method, path, status code, and elapsed time for each completed request.

    Args:
        resp: Flask response object.

    Returns:
        The unmodified response object.
    """
    try:
        dt_ms = (time.time() - getattr(g, "start_time", time.time())) * 1000.0
        logger.info("%s %s -> %s (%.1fms)", request.method, request.path, resp.status_code, dt_ms)
    except (RuntimeError, AttributeError, TypeError, ValueError):
        pass
    return resp

@app.errorhandler(404)
def _not_found(_):
    """Handle 404 Not Found errors.

    Args:
        _: Unused exception argument.

    Returns:
        JSON error response with status 404.
    """
    return error("not found", 404)

@app.errorhandler(500)
def _server_error(e):
    """Handle unhandled 500 Internal Server Errors.

    Args:
        e: The exception that triggered the error handler.

    Returns:
        JSON error response with status 500.
    """
    logger.exception("Unhandled server error: %s", e)
    return error("internal server error", 500)

@app.route('/', methods=['GET', 'POST'])
def home():
    """Render the server landing page.

    Args:
        None

    Returns:
        HTML string with links to /help and /examples.
    """
    return "<h1>Vector Database Server</h1>" \
    "<p>This server provides access to querying the MTG card vector database.</p>" \
    "<p>For a list of available endpoints, visit <a href='/help'>/help</a></p>" \
    "<p>For a list of examples, visit <a href='/examples'>/examples</a></p>"

@app.route('/help', methods=['GET'])
def help_route():
    """Return a JSON description of all available API endpoints.

    Args:
        None

    Returns:
        JSON object mapping endpoint paths to method, body, and description.
    """
    return jsonify({
        "endpoints": {
            "/status": {
                "method": "POST",
                "body": {},
                "description": "Return service health status.",
            },
            "/get_vector": {
                "method": "POST",
                "body": {"id": "Card Name"},
                "description": "Get the vector for the given id.",
            },
            "/get_vector_description": {
                "method": "POST",
                "body": {"id": "Card Name"},
                "description": "Get the description for the given id.",
            },
            "/get_vector_descriptions": {
                "method": "POST",
                "body": {"cards": ["Card Name", "..."]},
                "description": "Get descriptions for a list of ids.",
            },
            "/get_random_vector": {
                "method": "POST",
                "body": {},
                "description": "Get a random vector from the database.",
            },
            "/get_random_vector_description": {
                "method": "POST",
                "body": {},
                "description": "Get the description for a random vector from the database.",
            },
            "/get_similar_vectors": {
                "method": "POST",
                "body": {"id": "Card Name", "num_vectors": 10},
                "description": (
                    "Get the most similar vectors to the given id, "
                    "num_vectors is optional and defaults to 5."
                ),
            },
            "/get_tags": {
                "method": "POST",
                "body": {"id": "Card Name", "threshold": 0.5, "top_k": 8},
                "description": (
                    "Get the tags for the given id, threshold and top_k are "
                    "optional and default to 0.5 and 8 respectively."
                ),
            },
            "/get_tag_list": {
                "method": "POST",
                "body": {"cards": ["Card Name", "..."], "threshold": 0.5, "top_k": 8},
                "description": "Get tags for a list of ids.",
            },
            "/get_tags_from_vector": {
                "method": "POST",
                "body": {"vector": ["number", "..."], "threshold": 0.5, "top_k": 8},
                "description": (
                    "Get the tags for the given vector, threshold and top_k are "
                    "optional and default to 0.5 and 8 respectively"
                ),
            },
            "/analyze_deck": {
                "method": "POST",
                "body": {"commander": "Card Name", "cards": ["Card Name", "..."]},
                "description": "Analyze a deck and return SimpleDeckAnalyzer metrics.",
            },
            "/generate_deck": {
                "method": "POST",
                "body": {"id": "Commander Name"},
                "description": "Generate a deck for the given commander id.",
            },
        }
    })

@app.route('/examples', methods=['GET'])
def examples():
    """Return example request bodies for all API endpoints.

    Args:
        None

    Returns:
        JSON object mapping endpoint paths to example method and body.
    """
    return jsonify({
        "examples": {
            "/status": {
                "method": "POST",
                "body": {},
            },
            "/get_vector": {
                "method": "POST",
                "body": {"id": "Magnus the Red"},
            },
            "/get_vector_description": {
                "method": "POST",
                "body": {"id": "Magnus the Red"},
            },
            "/get_vector_descriptions": {
                "method": "POST",
                "body": {"cards": ["Magnus the Red", "Sol Ring"]},
            },
            "/get_random_vector": {
                "method": "POST",
                "body": {},
            },
            "/get_random_vector_description": {
                "method": "POST",
                "body": {},
            },
            "/get_similar_vectors": {
                "method": "POST",
                "body": {"id": "Magnus the Red", "num_vectors": 10},
            },
            "/get_tags": {
                "method": "POST",
                "body": {"id": "Magnus the Red", "threshold": 0.5, "top_k": 8},
            },
            "/get_tag_list": {
                "method": "POST",
                "body": {
                    "cards": ["Magnus the Red", "Sol Ring"],
                    "threshold": 0.5,
                    "top_k": 8,
                },
            },
            "/get_tags_from_vector": {
                "method": "POST",
                "body": {"vector": ["number", "..."], "threshold": 0.5, "top_k": 8},
                "description": (
                    "Get the tags for the given vector, threshold and top_k are "
                    "optional and default to 0.5 and 8 respectively"
                ),
            },
            "/analyze_deck": {
                "method": "POST",
                "body": {
                    "commander": "Magnus the Red",
                    "cards": ["Sol Ring", "Island", "Mountain"],
                },
                "description": "Analyze a deck list and return deck statistics.",
            },
            "/generate_deck": {
                "method": "POST",
                "body": {"id": "Magnus the Red"},
            },
        }
    })

@app.route('/status', methods=['POST'])
def health():
    """Return the health status of all loaded server components.

    Args:
        None

    Returns:
        JSON object with status flags for the tagging model, vector database, and deck generator.
    """
    healthy = (model is not None) and (vd is not None) and len(vd) > 0
    return jsonify({
        "status": "healthy" if healthy else "error",
        "model_loaded": model is not None,
        "vector_db_loaded": vd is not None and len(vd) > 0,
        "vd_size": len(vd) if vd is not None else 0,
        "deckgen_loaded": _deckgen.bundle is not None,
        "deckgen_state": _deckgen.state,
    })

@app.route('/get_vector', methods=['POST'])
@limiter.limit(set_limit)
def get_vector():
    """Return the raw embedding vector for a card by name.

    Args:
        None

    Returns:
        JSON object with 'id' and 'vector' (list of floats).
    """
    data = request.get_json(silent=True) or {}
    try:
        v_id = parse_required_card_id(data)
    except ValueError:
        return error("invalid request payload", 400)

    try:
        card_id = resolve_card_id(v_id)
        vector = vd.get(card_id)
    except KeyError:
        return error("id not found", 400)
    return jsonify(
        {
            "id": card_id,
            "vector": vector.tolist() if vector is not None else None,
        }
    )

@app.route('/get_vector_description', methods=['POST'])
@limiter.limit(set_limit)
def get_vector_description():
    """Return the human-readable description dict for a card by name.

    Args:
        None

    Returns:
        JSON object with card metadata from the vector database.
    """
    data = request.get_json(silent=True) or {}
    try:
        v_id = parse_required_card_id(data)
    except ValueError:
        return error("invalid request payload", 400)

    try:
        vector_id = resolve_card_id(v_id)
    except KeyError:
        return error("id not found", 400)

    return jsonify(vd.get_vector_description_dict(vector_id))


@app.route('/get_vector_descriptions', methods=['POST'])
@limiter.limit(set_limit)
def get_vector_descriptions():
    """Return description dicts for a batch of cards by name.

    Args:
        None

    Returns:
        JSON object with 'found' mapping names to descriptions and 'missing' mapping names to errors.
    """
    data = request.get_json(silent=True) or {}
    try:
        cards = parse_card_list_payload(data)
    except ValueError:
        return error("invalid request payload", 400)

    found: dict[str, dict[str, Any]] = {}
    missing: dict[str, dict[str, str]] = {}
    for name in cards:
        try:
            card_id = resolve_card_id(name)
            found[name] = vd.get_vector_description_dict(card_id)
        except KeyError:
            missing[name] = {"error": "id not found", "requested_id": name}

    return jsonify({"found": found, "missing": missing})

@app.route('/get_random_vector', methods=['POST'])
@limiter.limit(set_limit)
def get_random_vector():
    """Return the id and raw embedding vector for a randomly sampled card.

    Args:
        None

    Returns:
        JSON object with 'id' and 'vector' (list of floats).
    """
    random_vector = vd.get_random_vector()
    return jsonify(
        {
            "id": random_vector[0],
            "vector": random_vector[1].tolist() if random_vector[1] is not None else None,
        }
    )

@app.route('/get_random_vector_description', methods=['POST'])
@limiter.limit(set_limit)
def get_random_vector_description():
    """Return the description dict for a randomly sampled card.

    Args:
        None

    Returns:
        JSON object with card metadata from the vector database.
    """
    return jsonify(vd.get_vector_description_dict(vd.get_random_vector()[0]))

@app.route('/get_similar_vectors', methods=['POST'])
@limiter.limit(set_limit)
def get_similar_vectors():
    """Return the most similar cards to a given card, ranked by vector similarity.

    Args:
        None

    Returns:
        JSON object mapping rank indices to card description dicts.
    """
    data = request.get_json(silent=True) or {}
    try:
        v_id = parse_required_card_id(data)
    except ValueError:
        return error("invalid request payload", 400)

    try:
        vector = vd.get(resolve_card_id(v_id))
    except KeyError:
        return error("id not found", 400)

    try:
        num_vectors = int(data.get("num_vectors", 5))
    except (TypeError, ValueError):
        return error("num_vectors must be an integer", 400)

    num_vectors = clamp_int(num_vectors, 1, 1000)
    results_list: list[tuple] = vd.get_similar_vectors(vector, num_vectors)

    results = {}
    for i, result in enumerate(results_list):
        results[i] = vd.get_vector_description_dict(result[0])

    return jsonify(results)

@app.route('/get_tags', methods=['POST'])
@limiter.limit(set_limit)
def get_tags():
    """Return predicted gameplay tags for a card by name.

    Args:
        None

    Returns:
        JSON object with 'predicted', 'predicted_scores', 'scores', and 'threshold'.
    """
    if model is None:
        return error("tagging model not loaded", 503)

    data = request.get_json(silent=True) or {}
    try:
        v_id = parse_required_card_id(data)
    except ValueError:
        return error("invalid request payload", 400)

    try:
        card_id = resolve_card_id(v_id)
    except KeyError:
        return error("id not found", 400)

    try:
        threshold = float(data.get("threshold", 0.5))
        top_k = int(data.get("top_k", 8))
    except (TypeError, ValueError):
        return error("threshold and top_k must be numbers", 400)
    threshold = clamp_float(threshold, 0.0, 1.0)
    top_k = clamp_int(top_k, 1, 1000)
    return jsonify(predict_tags_for_card(card_id, threshold, top_k))


@app.route('/get_tag_list', methods=['POST'])
@limiter.limit(set_limit)
def get_tag_list():
    """Return predicted gameplay tags for a batch of cards by name.

    Args:
        None

    Returns:
        JSON object with 'found' mapping names to tag results and 'missing' mapping names to errors.
    """
    if model is None:
        return error("tagging model not loaded", 503)

    data = request.get_json(silent=True) or {}
    try:
        cards = parse_card_list_payload(data)
        threshold = float(data.get("threshold", 0.5))
        top_k = int(data.get("top_k", 8))
    except (TypeError, ValueError):
        return error("invalid cards/threshold/top_k types", 400)

    threshold = clamp_float(threshold, 0.0, 1.0)
    top_k = clamp_int(top_k, 1, 1000)

    found: dict[str, dict[str, Any]] = {}
    missing: dict[str, dict[str, str]] = {}
    for name in cards:
        try:
            card_id = resolve_card_id(name)
            found[name] = {
                "card_id": card_id,
                **predict_tags_for_card(card_id, threshold, top_k),
            }
        except KeyError:
            missing[name] = {"error": "id not found", "requested_id": name}

    return jsonify({"found": found, "missing": missing})

@app.route('/get_tags_from_vector', methods=['POST'])
@limiter.limit(set_limit)
def get_tags_from_vector():
    """Return predicted gameplay tags for a raw embedding vector.

    Args:
        None

    Returns:
        JSON object with 'predicted', 'predicted_scores', 'scores', and 'threshold'.
    """
    if model is None:
        return error("tagging model not loaded", 503)

    data = request.get_json(silent=True) or {}
    if "vector" not in data or not isinstance(data["vector"], list):
        return error("JSON body must include 'vector': [float, ...]", 400)

    try:
        vec_np = np.asarray(data["vector"], dtype=np.float32)
        threshold = float(data.get("threshold", 0.5))
        top_k = int(data.get("top_k", 8))
    except (TypeError, ValueError):
        return error("invalid vector/threshold/top_k types", 400)

    threshold = clamp_float(threshold, 0.0, 1.0)
    top_k = clamp_int(top_k, 1, 1000)
    return jsonify(predict_from_vector(vec_np, threshold, top_k))

@app.route('/generate_deck', methods=['POST'])
@limiter.limit(set_limit)
def generate_deck():
    """Generate a Commander deck list for the given commander card.

    Args:
        None

    Returns:
        JSON object mapping card names to quantities, plus generation stats.
    """
    if _deckgen.bundle is None:
        if _deckgen.state == "failed":
            return error("deck generation model failed to load", 503)
        return error("deck generation model is still loading, please retry shortly", 503)

    data = request.get_json(silent=True) or {}
    try:
        v_id = parse_required_card_id(data)
    except ValueError:
        return error("invalid request payload", 400)

    try:
        card = resolve_card_id(v_id)
    except KeyError:
        return error("id not found", 400)

    return jsonify(_deckgen.bundle.generate(card))

@app.route('/analyze_deck', methods=['POST'])
@limiter.limit(set_limit)
def analyze_deck():
    """Analyze a submitted deck list and return SimpleDeckAnalyzer metrics.

    Args:
        None

    Returns:
        JSON object with mana curve, tag distribution, and other deck statistics.
    """
    data = request.get_json(silent=True) or {}
    commander = data.get("commander")
    cards = data.get("cards")

    if not isinstance(commander, str) or not commander.strip():
        return error("JSON body must include 'commander': 'Card Name'", 400)
    if not isinstance(cards, list) or not all(isinstance(c, str) for c in cards):
        return error("JSON body must include 'cards': ['Card Name', ...]", 400)

    deck = SimpleDeck.from_json(
        {
            "id": "",
            "commanders": [commander.strip()],
            "cards": [c.strip() for c in cards if c.strip()],
        }
    )

    analyzer = SimpleDeckAnalyzer(deck=deck, tag_dataset=tag_dataset, vd=vd)
    return jsonify(analyzer.analyze())

def format_id(v_id: str) -> str:
    """Apply title-case formatting to a card name, preserving lowercase transition words.

    Args:
        v_id: Raw card name string to format.

    Returns:
        Title-cased card name with transition words (of, the, in, etc.) in lowercase.
    """
    transition_words = {'of', 'the', 'in', 'on', 'at', 'to', 'for', 'and', 'but', 'or', 'nor'}

    words: list[str] = v_id.split(' ')

    capitalized_words = [
        word.capitalize() if word.lower() not in transition_words or i == 0 else word.lower()
        for i, word in enumerate(words)
    ]

    return ' '.join(capitalized_words)

@torch.inference_mode()
def predict_from_vector(vec_np: np.ndarray, threshold: float, top_k: int = 8):
    """Run tag prediction on a raw numpy embedding vector.

    Args:
        vec_np: Float32 numpy array representing a card embedding.
        threshold: Minimum sigmoid probability to include a tag in predicted output.
        top_k: Number of highest-scoring tags to include in the scores list.

    Returns:
        Dict with 'predicted' (list of tag strings), 'predicted_scores' (list of dicts),
        'scores' (top-k list of dicts), and 'threshold' (float).
    """
    x = torch.from_numpy(vec_np.astype(np.float32)).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.sigmoid(logits).float().cpu().numpy()[0]

    tk = max(1, top_k)
    top_idx = np.argsort(probs)[-tk:][::-1]
    scores = [{"tag": class_names[i], "score": float(probs[i])} for i in top_idx]

    predicted_scores, predicted = predicted_scores_from_probabilities(
        probs=probs,
        class_names=class_names,
        threshold=threshold,
    )

    return {
        "predicted": predicted,
        "predicted_scores": predicted_scores,
        "scores": scores,
        "threshold": float(threshold)
    }

def main() -> None:
    """Start the Flask development server on the configured port.

    Args:
        None

    Returns:
        None
    """
    app.run(host='0.0.0.0', port=int(os.environ.get('DEFAULT_PORT', 8080)))


if __name__ == '__main__':
    main()
