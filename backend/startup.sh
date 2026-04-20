#!/bin/bash
set -euo pipefail

if [ -n "${GCS_DATA_BUCKET:-}" ]; then
    echo "Downloading assets from gs://${GCS_DATA_BUCKET}/data/ ..."
    python - <<'EOF'
import os
from google.cloud import storage

bucket_name = os.environ["GCS_DATA_BUCKET"]
client = storage.Client()
bucket = client.bucket(bucket_name)

for blob in bucket.list_blobs(prefix="data/"):
    if blob.name.endswith("/"):
        continue
    local_path = os.path.join("/app/backend", blob.name)
    if os.path.isfile(local_path):
        print(f"  skip  {blob.name}")
        continue
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"  fetch {blob.name}")
    blob.download_to_filename(local_path)

print("Asset download complete.")
EOF
fi

exec gunicorn --bind :"${PORT}" --workers 1 --threads 8 --timeout 0 backend.api.vector_db_server:app
