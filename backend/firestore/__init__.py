"""Public firestore auth/connector exports."""

from .firebase_auth import generate_api_key, validate_api_key
from .firestore_connector import (
    authenticate_api_key,
    create_api_key,
    get_db,
    get_firestore_client,
    touch_last_used,
)

__all__ = [
    "generate_api_key",
    "validate_api_key",
    "create_api_key",
    "authenticate_api_key",
    "touch_last_used",
    "get_db",
    "get_firestore_client",
]
