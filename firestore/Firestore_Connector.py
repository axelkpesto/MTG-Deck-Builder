import os
from datetime import datetime, timedelta, timezone
from google.cloud import firestore

from firestore.Firebase_Auth import validate_api_key as validate_firebase_api_key
from firestore.Firebase_Auth import generate_api_key as generate_firebase_api_key

from dotenv import load_dotenv

load_dotenv()

def get_firestore_client(project_id: str | None = None) -> firestore.Client:
    return firestore.Client(project=project_id)

db = get_firestore_client(project_id=os.environ.get("PROJECT_ID"))

def create_api_key(user_id: str, rate_limit: str = "60/minute", prefix_len: int = 8) -> str:
    raw, prefix, key_hash = generate_firebase_api_key(prefix_len)

    now = datetime.now(timezone.utc)
    doc = {
        "user_id": user_id,
        "key_hash": key_hash,
        "is_active": True,
        "created_at": now,
        "last_used_at": None,
        "expires_at": now + timedelta(days=365),
        "rate_limit": rate_limit,
    }

    db.collection("api_keys").document(prefix).set(doc)
    return raw

def authenticate_api_key(raw_key: str) -> dict | None:
    if not raw_key or len(raw_key) < 8:
        return None

    prefix = raw_key[:8]
    doc_ref = db.collection("api_keys").document(prefix)
    doc = doc_ref.get()
    if not doc.exists:
        return None

    data = doc.to_dict()
    if not data.get("is_active", False):
        return None

    expires_at = data.get("expires_at")
    if expires_at is not None:
        now = datetime.now(timezone.utc)

        if isinstance(expires_at, datetime) and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        if expires_at < now:
            return None
        if not validate_firebase_api_key(raw_key, data["key_hash"]):
            return None

    doc_ref.update({"last_used_at": datetime.now(timezone.utc)})

    return {
        "api_key_id": doc.id,
        "user_id": data["user_id"],
        "rate_limit": data.get("rate_limit"),
    }

def touch_last_used(api_key_id: str) -> None:
    doc_ref = db.collection("api_keys").document(api_key_id)
    doc_ref.update({"last_used_at": datetime.now(timezone.utc)})

if __name__ == "__main__":
    user_id = "unlimited_user"
    raw_key = create_api_key(user_id, rate_limit="unlimited")
    print(f"Generated API Key: {raw_key}")