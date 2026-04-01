"""Firestore connector utilities for API-key lifecycle operations."""
import os
from datetime import datetime, timedelta
from typing import Any

from dotenv import load_dotenv
from google.cloud import firestore

from .firebase_auth import (
    ensure_utc,
    generate_api_key as generate_raw_api_key,
    get_or_create_pepper_config,
    get_pepper_config_ref,
    hash_api_key,
    now_utc,
    validate_api_key,
)

load_dotenv()

def get_firestore_client(project_id: str | None = None) -> firestore.Client:
    """Return a Firestore client for the configured or provided project."""
    return firestore.Client(project=project_id)

db = get_firestore_client(project_id=os.environ.get("PROJECT_ID"))


def key_prefix_length() -> int:
    """Return the configured API-key prefix length."""
    return int(os.environ.get("API_KEY_PREFIX_LENGTH"))


def key_expiration_duration() -> timedelta:
    """Return the configured API-key lifetime."""
    return timedelta(days=int(os.environ.get("API_KEY_EXPIRATION_DAYS")))


def _api_keys_collection():
    """Return the Firestore collection reference for API keys."""
    return db.collection("api_keys")

#TODO
def create_api_key(user_id: str, rate_limit: str = "60/minute", prefix_len: int | None = None) -> str:
    """Create and persist a new API key record; return the raw key."""
    prefix_len = prefix_len or key_prefix_length()
    pepper_config = get_or_create_pepper_config(db)
    raw, prefix = generate_raw_api_key(prefix_len)
    now = now_utc()
    expires_at = now + key_expiration_duration()

    doc = {
        "user_id": user_id,
        "key_hash": hash_api_key(raw, pepper_config["current_pepper"]),
        "pepper_version": pepper_config["current_version"],
        "previous_key_hash": None,
        "previous_key_prefix": None,
        "is_active": True,
        "created_at": now,
        "last_used_at": None,
        "expires_at": expires_at,
        "rate_limit": rate_limit,
        "replaced_by_prefix": None,
    }

    _api_keys_collection().document(prefix).set(doc)
    return raw


def _is_expired(dt: datetime | None, now: datetime) -> bool:
    """Return whether a document expiry timestamp is in the past."""
    dt = ensure_utc(dt)
    return bool(dt and dt < now)


def _auth_response(doc_id: str, data: dict[str, Any]) -> dict[str, Any]:
    """Build the normalized auth payload for successful API-key auth."""
    return {
        "api_key_id": doc_id,
        "user_id": data["user_id"],
        "rate_limit": data.get("rate_limit"),
    }


def validate_current_key(raw_key: str, data: dict[str, Any], pepper_config: dict[str, Any], now: datetime) -> bool:
    """Validate a key record against the current pepper generation."""
    if not data.get("is_active", False):
        return False
    if _is_expired(data.get("expires_at"), now):
        return False
    key_hash = data.get("key_hash")
    pepper_version = data.get("pepper_version")
    if not key_hash or pepper_version != pepper_config["current_version"]:
        return False
    return validate_api_key(raw_key, key_hash, pepper_config["current_pepper"])


#TODO
def _validate_previous_key(raw_key: str, data: dict[str, Any], pepper_config: dict[str, Any], now: datetime) -> bool:
    """Validate a key record against the previous pepper generation."""
    if not data.get("is_active", False):
        return False
    if _is_expired(data.get("expires_at"), now):
        return False
    previous_key_hash = data.get("previous_key_hash")
    previous_version = pepper_config.get("previous_version")
    previous_pepper = pepper_config.get("previous_pepper")
    pepper_version = data.get("pepper_version")
    if (not previous_key_hash or previous_version is None or not previous_pepper or pepper_version != pepper_config["current_version"]):
        return False
    return validate_api_key(raw_key, previous_key_hash, previous_pepper)


#TODO
def authenticate_api_key(raw_key: str) -> dict[str, Any] | None:
    """Validate a raw key and return auth metadata when valid."""
    if not raw_key or len(raw_key) < key_prefix_length():
        return None

    now = now_utc()
    pepper_config = get_or_create_pepper_config(db)
    prefix = raw_key[:key_prefix_length()]
    doc_ref = _api_keys_collection().document(prefix)
    doc = doc_ref.get()
    if not doc.exists:
        return None

    data = doc.to_dict() or {}
    if validate_current_key(raw_key, data, pepper_config, now):
        return _auth_response(doc.id, data)

    replaced_by_prefix = data.get("replaced_by_prefix")
    if replaced_by_prefix:
        replacement_doc = _api_keys_collection().document(str(replaced_by_prefix)).get()
        if replacement_doc.exists:
            replacement_data = replacement_doc.to_dict() or {}
            if _validate_previous_key(raw_key, replacement_data, pepper_config, now):
                return _auth_response(replacement_doc.id, replacement_data)

    if _validate_previous_key(raw_key, data, pepper_config, now):
        return _auth_response(doc.id, data)

    return None


def touch_last_used(api_key_id: str) -> None:
    """Update the last-used timestamp for a key document."""
    _api_keys_collection().document(api_key_id).update({"last_used_at": now_utc()})


def rotate_pepper_and_keys() -> dict[str, Any]:
    """Rotate the shared API pepper and rotate all active keys to the new generation."""
    started_at = now_utc()
    pepper_config = get_or_create_pepper_config(db)
    new_pepper = os.urandom(32).hex()
    new_version = int(pepper_config["current_version"]) + 1

    new_config = {
        "current_pepper": new_pepper,
        "current_version": new_version,
        "previous_pepper": pepper_config["current_pepper"],
        "previous_version": pepper_config["current_version"],
        "updated_at": started_at,
    }
    get_pepper_config_ref(db).set(new_config)

    active_docs = list(
        _api_keys_collection()
        .where(filter=firestore.FieldFilter("is_active", "==", True))
        .where(filter=firestore.FieldFilter("pepper_version", "==", pepper_config["current_version"]))
        .stream()
    )

    rotated_keys: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    keys_rotated_count = 0
    prefix_len = key_prefix_length()
    expires_delta = key_expiration_duration()

    for snapshot in active_docs:
        old_data = snapshot.to_dict() or {}
        user_id = old_data.get("user_id")
        if not user_id:
            errors.append({"api_key_id": snapshot.id, "error": "missing user_id"})
            continue

        raw_key, new_prefix = generate_raw_api_key(prefix_len)
        new_doc = {
            "user_id": user_id,
            "key_hash": hash_api_key(raw_key, new_pepper),
            "pepper_version": new_version,
            "previous_key_hash": old_data.get("key_hash"),
            "previous_key_prefix": snapshot.id,
            "is_active": True,
            "created_at": started_at,
            "last_used_at": None,
            "expires_at": started_at + expires_delta,
            "rate_limit": old_data.get("rate_limit", "60/minute"),
            "replaced_by_prefix": None,
        }

        try:
            batch = db.batch()
            batch.set(_api_keys_collection().document(new_prefix), new_doc)
            batch.update(
                _api_keys_collection().document(snapshot.id),
                {
                    "is_active": False,
                    "replaced_by_prefix": new_prefix,
                },
            )
            batch.commit()
            keys_rotated_count += 1
            rotated_keys.append(
                {
                    "user_id": user_id,
                    "new_key": raw_key,
                    "new_prefix": new_prefix,
                    "old_prefix": snapshot.id,
                }
            )
        except Exception as exc:
            errors.append({"api_key_id": snapshot.id, "error": str(exc)})

    audit_log = {
        "rotation_started_at": started_at,
        "rotation_completed_at": now_utc(),
        "new_pepper_version": new_version,
        "keys_rotated_count": keys_rotated_count,
        "errors": errors,
    }
    db.collection("security_audit_logs").document(started_at.isoformat()).set(audit_log)

    return {
        **audit_log,
        "rotated_keys": rotated_keys,
    }


if __name__ == "__main__":
    DEMO_USER_ID = "unlimited_user"
    _raw_key = create_api_key(DEMO_USER_ID, rate_limit="unlimited")
