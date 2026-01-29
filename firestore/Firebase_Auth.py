import os
import hmac
import hashlib
import secrets
from dotenv import load_dotenv

load_dotenv()
API_KEY_PEPPER = os.environ.get("API_KEY_PEPPER", KeyError("API_KEY_PEPPER not set in environment"))

def generate_api_key(prefix_length: int = 8):
    raw = "mtg_" + secrets.token_urlsafe(32)
    prefix = raw[:prefix_length]
    digest = hmac.new(API_KEY_PEPPER.encode("utf-8"), raw.encode("utf-8"), hashlib.sha256).hexdigest()
    return raw, prefix, digest

def validate_api_key(raw_key: str, key_hash: str) -> bool:
    computed_hash = hmac.new(API_KEY_PEPPER.encode("utf-8"), raw_key.encode("utf-8"), hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed_hash, key_hash)