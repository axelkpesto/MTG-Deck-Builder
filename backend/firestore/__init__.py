"""Public firestore auth/connector exports."""

from .firebase_auth import (
    generate_api_key,
    generate_pepper,
    get_or_create_pepper_config,
    hash_api_key,
    validate_api_key,
)
from .firestore_connector import (
    authenticate_api_key,
    create_api_key,
    db,
    get_firestore_client,
    rotate_pepper_and_keys,
    touch_last_used,
)

__all__ = [
    "db",
    "generate_api_key",
    "generate_pepper",
    "hash_api_key",
    "validate_api_key",
    "get_or_create_pepper_config",
    "create_api_key",
    "authenticate_api_key",
    "touch_last_used",
    "rotate_pepper_and_keys",
    "get_firestore_client",
]
