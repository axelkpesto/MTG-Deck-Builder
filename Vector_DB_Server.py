import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from Vector_Database import VectorDatabase
from Card_Lib import CardEncoder, CardDecoder

import torch
import numpy as np

from Tagging_Model import load_model

app = Flask(__name__)
load_dotenv()
CORS(app)
app.config["DEBUG"] = os.environ.get("FLASK_DEBUG", 0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, class_names = load_model("models/tagging_mlp.pt")
model.to(device).eval()

vd = VectorDatabase(CardEncoder(), CardDecoder())
vd.load("datasets/vector_data.pt")

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

#Example Request:
#(Invoke-RestMethod -Uri "http://127.0.0.1:5000/get_vector/Shunt")
@app.route('/get_vector/<string:v_id>', methods=['GET'])
def get_vector(v_id):
    try:
        vector = vd.find_vector(format_id(v_id))
    except KeyError:
        return jsonify({"error": "id not found"}), 400
    return jsonify({"vector": vector.tolist() if vector is not None else None})

#Example Request:
#(Invoke-RestMethod -Uri "http://127.0.0.1:5000/get_vector_description/Shunt")
@app.route('/get_vector_description/<string:v_id>', methods=['GET'])
def get_vector_description(v_id):
    try:
        id = vd.find_id(format_id(v_id))
    except KeyError:
        return jsonify({"error": "id not found"}), 400
    
    return jsonify(vd.get_vector_description_dict(id))

#Example Request:
#curl http://127.0.0.1:5000/get_random_vector
@app.route('/get_random_vector', methods=['GET'])
def get_random_vector():
    random_vector = vd.get_random_vector()
    return jsonify({"id":(random_vector[0]), "vector": random_vector[1].tolist() if random_vector[1] is not None else None})

#Example Request:
#(Invoke-RestMethod -Uri "http://127.0.0.1:5000/get_random_vector_description")
@app.route('/get_random_vector_description', methods=['GET'])
def get_random_vector_description():
    return jsonify(vd.get_vector_description_dict(vd.get_random_vector()[0]))

#Example Request:
#(Invoke-RestMethod -Uri "http://127.0.0.1:5000/get_similar_vectors/Shunt?num_vectors=10")
@app.route('/get_similar_vectors/<string:v_id>', methods=['GET'])
def get_similar_vectors(v_id):
    try:
        vector = vd.find_vector(format_id(v_id))
    except KeyError:
        return jsonify({"error": "id not found"}), 400

    try:
        num_vectors = request.args.get('num_vectors', default=5, type=int)
    except ValueError:
        return jsonify({"error": "num_vectors must be an integer"}), 400
    
    results_list: list[tuple] = vd.get_similar_vectors(vector, num_vectors)
    
    results = {}
    for i in range(len(results_list)):
        results[i] = vd.get_vector_description_dict(results_list[i][0])

    return jsonify(results)

# curl "http://127.0.0.1:5000/get_tags/Magnus the Red?threshold=0.55&top_k=8"
@app.route('/get_tags/<string:v_id>', methods=['GET'])
def get_tags(v_id):
    try:
        vector = vd.find_vector(format_id(v_id))
    except KeyError:
        return jsonify({"error": "id not found"}), 400

    try:
        threshold = request.args.get('threshold', default=0.5, type=float)
        top_k = request.args.get('top_k', default=8, type=int)
    except ValueError:
        return jsonify({"error": "threshold and top_k must be numbers"}), 400

    if hasattr(vector, "detach"):
        vec_np = vector.detach().cpu().numpy()
    elif hasattr(vector, "cpu"):
        vec_np = vector.cpu().numpy()
    else:
        vec_np = np.asarray(vector, dtype=np.float32)

    return jsonify(_predict_from_vector(vec_np, threshold, top_k))

@app.route('/get_tags_from_vector', methods=['POST'])
def get_tags_from_vector():
    data = request.get_json(silent=True) or {}
    if "vector" not in data or not isinstance(data["vector"], list):
        return jsonify({"error": "JSON body must include 'vector': [float, ...]"}), 400

    try:
        vec_np = np.asarray(data["vector"], dtype=np.float32)
        threshold = float(data.get("threshold", 0.5))
        top_k = int(data.get("top_k", 8))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid vector/threshold/top_k types"}), 400

    return jsonify(_predict_from_vector(vec_np, threshold, top_k))

def format_id(v_id: str) -> str:
    transition_words = {'of', 'the', 'in', 'on', 'at', 'to', 'for', 'and', 'but', 'or', 'nor'}
    
    words: list[str] = v_id.split(' ')
    
    capitalized_words = [
        word.capitalize() if word.lower() not in transition_words or i == 0 else word.lower()
        for i, word in enumerate(words)
    ]
    
    return ' '.join(capitalized_words)

@torch.inference_mode()
def _predict_from_vector(vec_np: np.ndarray, threshold: float, top_k: int = 8):
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
