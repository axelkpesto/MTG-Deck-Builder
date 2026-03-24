"""Generate predicted tags for each vector-database card entry."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.tagging_model import load_model
from vector_database import VectorDatabase
from config import CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, class_names = load_model(CONFIG.models["TAGGING_MODEL_PATH"])
model.to(device).eval()

vd = VectorDatabase.load_static(CONFIG.datasets["VECTOR_DATABASE_PATH"])

output_path = CONFIG.datasets["TAGS_DATASET_PATH"]
tagged_data = {}
for card, vector in vd.items():
    vec_np = VectorDatabase.vector_to_numpy(vector)

    x = torch.from_numpy(vec_np.astype(np.float32)).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.sigmoid(logits).float().cpu().detach().numpy()[0]

    pred_idxs = np.where(probs >= 0.5)[0]
    predicted_scores = sorted(
        (
            {"tag": class_names[i], "score": float(probs[i])}
            for i in pred_idxs.tolist()
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    predicted_tags = [item["tag"] for item in predicted_scores]

    tagged_data[card] = {
        "tags": predicted_tags,
    }

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(tagged_data, f, ensure_ascii=False, indent=4)
