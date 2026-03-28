"""Generate predicted tags for each vector-database card entry."""
import json

import numpy as np
import torch

from backend.config import CONFIG
from backend.ml.tagging_model import load_model, predicted_scores_from_probabilities
from backend.vector_database import VectorDatabase

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

    _, predicted_tags = predicted_scores_from_probabilities(
        probs=probs,
        class_names=class_names,
        threshold=0.5,
    )

    tagged_data[card] = {
        "tags": predicted_tags,
    }

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(tagged_data, f, ensure_ascii=False, indent=4)
