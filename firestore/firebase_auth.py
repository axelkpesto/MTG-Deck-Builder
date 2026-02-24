"""API key generation and validation helpers for Firestore-backed auth."""

import os
import hmac
import hashlib
import secrets
from dotenv import load_dotenv

load_dotenv()
API_KEY_PEPPER = os.environ.get("API_KEY_PEPPER")
if API_KEY_PEPPER is None:
    raise KeyError("API_KEY_PEPPER not set in environment")

PEPPER_STR = str(API_KEY_PEPPER)

def generate_api_key(prefix_length: int = 8):
    """Create a raw API key, its document prefix, and HMAC hash."""
    raw = "mtg_" + secrets.token_urlsafe(32)
    prefix = raw[:prefix_length]
    digest = hmac.new(
        PEPPER_STR.encode("utf-8"),
        raw.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return raw, prefix, digest

def validate_api_key(raw_key: str, key_hash: str) -> bool:
    """Validate a raw API key against a stored HMAC hash."""
    computed_hash = hmac.new(
        PEPPER_STR.encode("utf-8"),
        raw_key.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(computed_hash, key_hash)
