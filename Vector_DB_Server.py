import os, time

import logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from Vector_Database import VectorDatabase
from Tagging_Model import load_model
from firestore.Firestore_Connector import authenticate_api_key, touch_last_used

import torch
import numpy as np
from typing import Any

app = Flask(__name__)
load_dotenv()
CORS(app)
app.config["DEBUG"] = os.environ.get("FLASK_DEBUG", 0)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector-db-server")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, class_names = None, []
try:
    model, class_names = load_model("models/tagging_mlp.pt")
    model.to(device).eval()
except Exception as e:
    print(f"Error loading tagging model: {e}")

vd = VectorDatabase.load_static("datasets/vector_data.pt")

limiter = Limiter(
    app=app,
    key_func=lambda: str(getattr(g, "api_key_id", get_remote_address())),
    default_limits=["120 per minute"],
    storage_uri=os.environ.get("REDIS_URL", "")
)

def error(message: str, status: int = 400, **extra: Any):
    payload = {"error": message, **extra}
    return jsonify(payload), status

def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

def clamp_float(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def get_api_key_from_request() -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return request.headers.get("X-API-KEY")

def set_limit() -> str:
    if hasattr(g, "rate_limit") and g.rate_limit:
        if "/" in g.rate_limit:
            split = g.rate_limit.split("/")
            return f"{split[0]} per {split[1]}"
        else:
            return g.rate_limit
    return limiter.default_limits[0]

@app.before_request
def _start_timer():
    g._t0 = time.time()

@app.before_request
def _authenticate_api_key():
    if request.path in ["/", "/help", "/examples", "/status"]:
        return

    api_key = request.headers.get("X-API-KEY")
    if not api_key:
        return error("missing API key", 401)

    if api_key == g.get("last_raw"):
        return

    info = authenticate_api_key(api_key)
    if not info:
        return error("invalid or expired API key", 403)
    
    g.api_key_id = info["api_key_id"]
    g.user_id = info["user_id"]
    g.rate_limit = info.get("rate_limit")
    g.last_raw = api_key

    try:
        touch_last_used(g.api_key_id)
    except Exception:
        pass

@app.after_request
def _log_request(resp):
    try:
        dt_ms = (time.time() - getattr(g, "_t0", time.time())) * 1000.0
        logger.info("%s %s -> %s (%.1fms)", request.method, request.path, resp.status_code, dt_ms)
    except Exception:
        pass
    return resp

@app.errorhandler(404)
def _not_found(_):
    return error("not found", 404)

@app.errorhandler(500)
def _server_error(e):
    logger.exception("Unhandled server error: %s", e)
    return error("internal server error", 500)

@app.route('/', methods=['GET'])
def home():
    return "<h1>Vector Database Server</h1>" \
    "<p>This server provides access to querying the MTG card vector database.</p>" \
    "<p>For a list of available endpoints, visit <a href='/help'>/help</a></p>" \
    "<p>For a list of examples, visit <a href='/examples'>/examples</a></p>"

@app.route('/help', methods=['GET'])
def help():
    return jsonify({
        "endpoints": {
            "/get_vector/STR": "Get the vector for the given id",
            "/get_vector_description/STR": "Get the description for the given id",
            "/get_random_vector": "Get a random vector from the database",
            "/get_random_vector_description": "Get the description for a random vector from the database",
            "/get_similar_vectors/STR?num_vectors=INT": "Get the most similar vectors to the given id, num_vectors is optional and defaults to 5",
            "/get_tags/STR?threshold=FLOAT&top_k=INT": "Get the tags for the given id, threshold and top_k are optional and default to 0.5 and 8 respectively"
        }
    })

@app.route('/examples', methods=['GET'])
def examples():
    return jsonify({
        "examples": {
            "/get_vector/Magnus the Red": "Get the vector for the given id",
            "/get_vector_description/Magnus the Red": "Get the description for the given id",
            "/get_random_vector": "Get a random vector from the database",
            "/get_random_vector_description": "Get the description for a random vector from the database",
            "/get_similar_vectors/Magnus the Red?num_vectors=10": "Get the most similar vectors to the given id, num_vectors is optional and defaults to 5",
            "/get_tags/Magnus the Red?threshold=0.5&top_k=8": "Get the tags for the given id, threshold and top_k are optional and default to 0.5 and 8 respectively"
        }
    })

@app.route('/status', methods=['GET'])
def health():
    healthy = (model is not None) and (vd is not None) and len(vd) > 0
    return jsonify({
        "status": "healthy" if healthy else "error",
        "your_api_key": getattr(g, "api_key_id", None),
        "model_loaded": model is not None,
        "vector_db_loaded": vd is not None and len(vd) > 0,
        "vd_size": len(vd) if vd is not None else 0,
    })

#Example Request:
#(Invoke-RestMethod -Uri "http://127.0.0.1:5000/get_vector/Shunt")
@app.route('/get_vector/<string:v_id>', methods=['GET'])
@limiter.limit(set_limit)
def get_vector(v_id):
    try:
        vector = vd.find_vector(format_id(v_id))
    except KeyError:
        return error("id not found", 400, requested_id=v_id)
    return jsonify({"id": format_id(v_id), "vector": vector.tolist() if vector is not None else None})

#Example Request:
#(Invoke-RestMethod -Uri "http://127.0.0.1:5000/get_vector_description/Shunt")
@app.route('/get_vector_description/<string:v_id>', methods=['GET'])
@limiter.limit(set_limit)
def get_vector_description(v_id):
    try:
        id = vd.find_id(format_id(v_id))
    except KeyError:
        return error("id not found", 400, requested_id=v_id)
    
    return jsonify(vd.get_vector_description_dict(id))

#Example Request:
#curl http://127.0.0.1:5000/get_random_vector
@app.route('/get_random_vector', methods=['GET'])
@limiter.limit(set_limit)
def get_random_vector():
    random_vector = vd.get_random_vector()
    return jsonify({"id":(random_vector[0]), "vector": random_vector[1].tolist() if random_vector[1] is not None else None})

#Example Request:
#(Invoke-RestMethod -Uri "http://127.0.0.1:5000/get_random_vector_description")
@app.route('/get_random_vector_description', methods=['GET'])
@limiter.limit(set_limit)
def get_random_vector_description():
    return jsonify(vd.get_vector_description_dict(vd.get_random_vector()[0]))

#Example Request:
#(Invoke-RestMethod -Uri "http://127.0.0.1:5000/get_similar_vectors/Shunt?num_vectors=10")
@app.route('/get_similar_vectors/<string:v_id>', methods=['GET'])
@limiter.limit(set_limit)
def get_similar_vectors(v_id):
    try:
        vector = vd.find_vector(format_id(v_id))
    except KeyError:
        return error("id not found", 400, requested_id=v_id)

    try:
        num_vectors = request.args.get('num_vectors', default=5, type=int)
    except ValueError:
        return error("num_vectors must be an integer", 400, quantity=request.args.get('num_vectors'))
    
    num_vectors = clamp_int(num_vectors, 1, 1000)
    results_list: list[tuple] = vd.get_similar_vectors(vector, num_vectors)
    
    results = {}
    for i in range(len(results_list)):
        results[i] = vd.get_vector_description_dict(results_list[i][0])

    return jsonify(results)

# curl "http://127.0.0.1:5000/get_tags/Magnus the Red?threshold=0.55&top_k=8"
@app.route('/get_tags/<string:v_id>', methods=['GET'])
@limiter.limit(set_limit)
def get_tags(v_id):
    try:
        vector = vd.find_vector(format_id(v_id))
    except KeyError:
        return error("id not found", 400, requested_id=v_id)

    try:
        threshold = request.args.get('threshold', default=0.5, type=float)
        top_k = request.args.get('top_k', default=8, type=int)
    except ValueError:
        return error("threshold and top_k must be numbers", 400, threshold=request.args.get('threshold'), top_k=request.args.get('top_k'))

    if hasattr(vector, "detach"):
        vec_np = vector.detach().cpu().numpy()
    elif hasattr(vector, "cpu"):
        vec_np = vector.cpu().numpy()
    else:
        vec_np = np.asarray(vector, dtype=np.float32)

    threshold = clamp_float(threshold, 0.0, 1.0)
    top_k = clamp_int(top_k, 1, 1000)
    return jsonify(predict_from_vector(vec_np, threshold, top_k))

@app.route('/get_tags_from_vector', methods=['POST'])
@limiter.limit(set_limit)
def get_tags_from_vector():
    data = request.get_json(silent=True) or {}
    if "vector" not in data or not isinstance(data["vector"], list):
        return error("JSON body must include 'vector': [float, ...]", 400, received=data)

    try:
        vec_np = np.asarray(data["vector"], dtype=np.float32)
        threshold = float(data.get("threshold", 0.5))
        top_k = int(data.get("top_k", 8))
    except (TypeError, ValueError):
        return error("Invalid vector/threshold/top_k types", 400, received=data)

    threshold = clamp_float(threshold, 0.0, 1.0)
    top_k = clamp_int(top_k, 1, 1000)
    return jsonify(predict_from_vector(vec_np, threshold, top_k))

def format_id(v_id: str) -> str:
    transition_words = {'of', 'the', 'in', 'on', 'at', 'to', 'for', 'and', 'but', 'or', 'nor'}
    
    words: list[str] = v_id.split(' ')
    
    capitalized_words = [
        word.capitalize() if word.lower() not in transition_words or i == 0 else word.lower()
        for i, word in enumerate(words)
    ]
    
    return ' '.join(capitalized_words)

@torch.inference_mode()
def predict_from_vector(vec_np: np.ndarray, threshold: float, top_k: int = 8):
    if model is None:
        return error("Tagging model not loaded", 503)
    
    x = torch.from_numpy(vec_np.astype(np.float32)).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.sigmoid(logits).float().cpu().numpy()[0]

    pred_idxs = np.where(probs >= threshold)[0]
    predicted = [class_names[i] for i in pred_idxs.tolist()]

    tk = max(1, top_k)
    top_idx = np.argsort(probs)[-tk:][::-1]
    scores = [{"tag": class_names[i], "score": float(probs[i])} for i in top_idx]

    if not predicted:
        predicted = [s["tag"] for s in scores]

    return {
        "predicted": predicted,
        "scores": scores,
        "threshold": float(threshold)
    }

if __name__ == '__main__':
    app.run()
    import requests
    load_dotenv()
    API_KEY = os.environ.get("FIREBASE_API_KEY", "")
    url = "http://127.0.0.1:5000/get_random_vector"

    response = requests.get(
        url,
        headers={"x-api-key": API_KEY}
    )

    print(response.status_code)
    print(response.json())