"""Firestore connector utilities for API-key lifecycle operations and saved decks."""

import os
from datetime import datetime, timedelta, timezone
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from dotenv import load_dotenv

from .firebase_auth import validate_api_key as validate_firebase_api_key
from .firebase_auth import generate_api_key as generate_firebase_api_key

load_dotenv()

def get_firestore_client(project_id: str | None = None) -> firestore.Client:
    """Return a Firestore client for the configured or provided project."""
    return firestore.Client(project=project_id)

db = get_firestore_client(project_id=os.environ.get("PROJECT_ID"))


def _saved_decks_collection():
    """Return the Firestore collection used for persisted user decks."""
    return db.collection("saved_decks")


def normalize_saved_cards(cards: list[dict]) -> list[dict]:
    """Normalize saved deck cards into the persisted schema."""
    normalized: list[dict] = []
    for row in cards:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        quantity = int(row.get("quantity", 1) or 1)
        normalized.append(
            {
                "name": name,
                "quantity": max(1, quantity),
            }
        )
    return normalized


def normalize_saved_deck_document(deck_id: str, payload: dict) -> dict:
    """Normalize saved-deck document shape for API consumers."""
    cards_payload = payload.get("cards", {})
    commander = str(cards_payload.get("commander", "")).strip()
    raw_cards = cards_payload.get("cards", [])
    deck_cards = normalize_saved_cards(raw_cards if isinstance(raw_cards, list) else [])
    return {
        "id": deck_id,
        **payload,
        "cards": {
            "commander": commander,
            "cards": deck_cards,
        },
        "card_count": sum(int(card.get("quantity", 1) or 1) for card in deck_cards),
    }


def save_user_deck(owner_id: str, owner_email: str, title: str, commander: str, cards: list[dict], deck_id: str | None = None, fmt: str = "commander") -> tuple[str, dict]:
    """Create or update a saved deck for a user and return its persisted payload."""
    saved_cards = normalize_saved_cards(cards)
    now = datetime.now(timezone.utc)
    deck_ref = _saved_decks_collection().document(deck_id) if deck_id else _saved_decks_collection().document()
    existing = deck_ref.get()

    payload = {
        "title": title.strip() or f"{commander.strip()} Deck",
        "owner_id": owner_id.strip(),
        "owner_email": owner_email.strip(),
        "format": fmt.strip() or "commander",
        "cards": {
            "commander": commander.strip(),
            "cards": saved_cards,
        },
        "updated_at": now,
    }

    if existing.exists:
        current = existing.to_dict() or {}
        if current.get("owner_id") != owner_id:
            raise PermissionError("Cannot overwrite a deck owned by another user.")
        payload["created_at"] = current.get("created_at", now)
        deck_ref.set(payload, merge=False)
    else:
        payload["created_at"] = now
        deck_ref.set(payload, merge=False)

    return deck_ref.id, normalize_saved_deck_document(deck_ref.id, payload)


def get_user_deck(owner_id: str, deck_id: str) -> dict | None:
    """Return a saved deck if it exists and belongs to the requested user."""
    doc = _saved_decks_collection().document(deck_id).get()
    if not doc.exists:
        return None
    payload = doc.to_dict() or {}
    if payload.get("owner_id") != owner_id:
        return None
    return normalize_saved_deck_document(doc.id, payload)


def list_user_decks(owner_id: str, limit: int = 100) -> list[dict]:
    """Return saved decks owned by a user ordered by most recent update."""
    query = (
        _saved_decks_collection()
        .where(filter=FieldFilter("owner_id", "==", owner_id))
        .order_by("updated_at", direction=firestore.Query.DESCENDING)
        .limit(int(limit))
    )
    return [normalize_saved_deck_document(doc.id, doc.to_dict() or {}) for doc in query.stream()]

def create_api_key(user_id: str, rate_limit: str = "60/minute", prefix_len: int = 8) -> str:
    """Create and persist a new API key record; return the raw key."""
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
    """Validate a raw key and return auth metadata when valid."""
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
    """Update the last-used timestamp for a key document."""
    doc_ref = db.collection("api_keys").document(api_key_id)
    doc_ref.update({"last_used_at": datetime.now(timezone.utc)})

if __name__ == "__main__":
    DEMO_USER_ID = "unlimited_user"
    _raw_key = create_api_key(DEMO_USER_ID, rate_limit="unlimited")
