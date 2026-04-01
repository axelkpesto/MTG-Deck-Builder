"""API key hashing and pepper-management helpers for Firestore-backed auth."""
import hashlib
import hmac
import secrets
from datetime import datetime, timezone
from typing import Any


def now_utc() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime | None) -> datetime | None:
    """Normalize naive datetimes to UTC and leave aware datetimes untouched."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def generate_pepper() -> str:
    """Generate a new random pepper value."""
    return secrets.token_urlsafe(48)


def hash_api_key(raw_key: str, pepper: str) -> str:
    """Compute the HMAC-SHA256 digest for a raw API key using a pepper."""
    return hmac.new(
        pepper.encode("utf-8"),
        raw_key.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def validate_api_key(raw_key: str, key_hash: str, pepper: str) -> bool:
    """Validate a raw API key against a stored HMAC hash and pepper."""
    computed_hash = hash_api_key(raw_key, pepper)
    return hmac.compare_digest(computed_hash, key_hash)


def generate_api_key(prefix_length: int = 8) -> tuple[str, str]:
    """Create a raw API key and its stable document prefix."""
    raw = "mtg_" + secrets.token_urlsafe(32)
    prefix = raw[:prefix_length]
    return raw, prefix


def default_pepper_config(current_pepper: str | None = None, current_version: int = 1, updated_at: datetime | None = None) -> dict[str, Any]:
    """Create the default Firestore pepper configuration document."""
    return {
        "current_pepper": current_pepper or generate_pepper(),
        "current_version": current_version,
        "previous_pepper": None,
        "previous_version": None,
        "updated_at": updated_at or now_utc(),
    }


def get_pepper_config_ref(db) -> Any:
    """Return the Firestore document reference for API-key pepper config."""
    return db.collection("security").document("api_key_config")


def get_or_create_pepper_config(db) -> dict[str, Any]:
    """Return the active pepper config, creating a default one if absent."""
    doc_ref = get_pepper_config_ref(db)
    snapshot = doc_ref.get()
    if snapshot.exists:
        data = snapshot.to_dict() or {}
        data["current_version"] = int(data.get("current_version") or 1)
        if data.get("previous_version") is not None:
            data["previous_version"] = int(data["previous_version"])
        data["updated_at"] = ensure_utc(data.get("updated_at")) or now_utc()
        return data

    config = default_pepper_config()
    doc_ref.set(config)
    return config
