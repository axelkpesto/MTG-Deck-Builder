#!/bin/bash
set -euo pipefail

if [ -n "${GCS_DATA_BUCKET:-}" ]; then
    echo "Downloading assets from gs://${GCS_DATA_BUCKET} ..."
    python - <<'EOF'
import os
from google.cloud import storage

REQUIRED_BLOBS = [
    "data/processed/vector_data.pt",
    "data/processed/tagging_labels.json",
    "data/processed/graph_nodes.pt",
    "data/processed/graph_edges.pt",
    "data/processed/decks_simple.json",
    "data/processed/node_embeddings.pt",
    "data/processed/node_features.pt",
    "data/models/tagging_mlp.pt",
    "data/models/commander_deck_gnn.pt",
]

bucket_name = os.environ["GCS_DATA_BUCKET"]
client = storage.Client()
bucket = client.bucket(bucket_name)

for blob_name in REQUIRED_BLOBS:
    local_path = os.path.join("/app/backend", blob_name)
    if os.path.isfile(local_path):
        print(f"  skip  {blob_name}")
        continue
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"  fetch {blob_name}")
    bucket.blob(blob_name).download_to_filename(local_path)

print("Asset download complete.")
EOF
fi

exec gunicorn --bind :"${PORT}" --workers 1 --threads 8 --timeout 0 backend.api.vector_db_server:app
