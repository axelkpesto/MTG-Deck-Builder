"""Shared pytest fixtures and helpers for backend tests."""
import os
from typing import List

import numpy as np
import pytest
import torch

from backend.card_data import Card, CardDecoder, CardFields
from backend.vector_database import VectorDatabase, VectorStore

# Set required env vars for modules that read them at import time. conftest is
# imported before any test module, so backend.api / backend.firestore see these
# by the time they are imported. (card_data / vector_database do not read them.)
os.environ.setdefault("API_KEY_PEPPER", "test-pepper-value")
os.environ.setdefault("REDIS_URL", "memory://")
os.environ.setdefault("AUTHENTICATE", "0")
os.environ.setdefault("FLASK_DEBUG", "0")


# ----------------------------- helpers ----------------------------------------


def make_card(
    *,
    card_name: str = "Test Card",
    commander_legal: bool = True,
    card_types: List[str] | None = None,
    card_supertypes: str = "",
    card_subtypes: List[str] | None = None,
    mana_cost: int = 3,
    mana_cost_exp: str = "{2}{U}",
    color_identity: List[str] | None = None,
    defense: str = "",
    rarity: str = "rare",
    text: str = "Whenever this creature attacks, draw a card.",
    rank: str = "1000",
    power: str = "2",
    toughness: str = "3",
    loyalty: str = "",
    card_id: str = "abc-123",
) -> Card:
    """Build a `Card` with safe defaults for use across multiple tests."""
    return Card(
        commander_legal=commander_legal,
        card_name=card_name,
        card_types=card_types if card_types is not None else ["creature"],
        card_supertypes=card_supertypes,
        card_subtypes=card_subtypes if card_subtypes is not None else ["human", "wizard"],
        mana_cost=mana_cost,
        mana_cost_exp=mana_cost_exp,
        color_identity=color_identity if color_identity is not None else ["U"],
        defense=defense,
        rarity=rarity,
        text=text,
        rank=rank,
        power=power,
        toughness=toughness,
        loyalty=loyalty,
        card_id=card_id,
    )


def make_encoded_vector(
    *,
    card_types: List[str] | None = None,
    supertypes: List[str] | None = None,
    subtypes: List[str] | None = None,
    mana: int = 3,
    color_identity: List[str] | None = None,
    rarity_index: int = 3,
) -> np.ndarray:
    """Build a structurally valid (no-embedding) encoded card vector for decoder tests.

    Layout matches `CardEncoder.encode` minus the text embedding suffix.
    """
    card_types = card_types or []
    supertypes = supertypes or []
    subtypes = subtypes or []
    color_identity = color_identity or []

    type_vocab = CardFields.card_types()
    super_vocab = CardFields.card_supertypes()
    sub_vocab = CardFields.card_subtypes()
    color_vocab = CardFields.color_identities()

    vec: list[float] = []
    vec += [1.0 if t in card_types else 0.0 for t in type_vocab]
    vec += [1.0 if s in supertypes else 0.0 for s in super_vocab]
    vec += [1.0 if s in subtypes else 0.0 for s in sub_vocab]
    vec += [float(mana)]
    vec += [1.0 if c in color_identity else 0.0 for c in color_vocab]
    vec += [float(rarity_index)]
    return np.array(vec, dtype=np.float32)


# ----------------------------- fixtures ---------------------------------------


@pytest.fixture
def sample_card() -> Card:
    """A generic well-formed `Card` object."""
    return make_card()


@pytest.fixture
def legendary_creature_card() -> Card:
    """A legendary creature `Card` (valid commander)."""
    return make_card(
        card_name="Atraxa, Praetors' Voice",
        card_types=["creature"],
        card_supertypes="legendary",
        card_subtypes=["angel", "horror"],
        mana_cost=4,
        color_identity=["W", "U", "B", "G"],
        text="Flying, vigilance, deathtouch, lifelink. At the beginning of your end step, proliferate.",
        rarity="mythic",
    )


@pytest.fixture
def basic_land_card() -> Card:
    """A basic land `Card`."""
    return make_card(
        card_name="Forest",
        card_types=["land"],
        card_supertypes="basic",
        card_subtypes=["forest"],
        mana_cost=0,
        mana_cost_exp="",
        color_identity=[],
        text="({T}: Add {G}.)",
        rarity="common",
        power="",
        toughness="",
    )


@pytest.fixture
def colorless_card() -> Card:
    """A non-zero-cost card with no colors in identity."""
    return make_card(
        card_name="Sol Ring",
        card_types=["artifact"],
        card_supertypes="",
        card_subtypes=[],
        mana_cost=1,
        mana_cost_exp="{1}",
        color_identity=[],
        text="{T}: Add {C}{C}.",
        rarity="uncommon",
        power="",
        toughness="",
    )


@pytest.fixture
def encoded_vector_factory():
    """Factory for creating encoded vectors without needing a real CardEncoder."""
    return make_encoded_vector


@pytest.fixture(name="fake_encoder")
def _fake_encoder():
    """Lightweight stand-in for `CardEncoder` to skip SentenceTransformer downloads."""

    class _FakeEncoder:
        def __init__(self, dim: int = 16):
            self.dim = dim

        def encode(self, card: Card):
            """Return a deterministic (name, vector) pair seeded by the card name."""
            rng = np.random.default_rng(abs(hash(card.card_name)) % (2**32))
            return card.card_name, rng.standard_normal(self.dim).astype(np.float32)

    return _FakeEncoder()


@pytest.fixture(name="fake_decoder")
def _fake_decoder():
    """Decoder fixture: a real CardDecoder is fine since it has no heavy deps."""
    return CardDecoder()


@pytest.fixture
def empty_vector_store(fake_encoder, fake_decoder):
    """A fresh, empty `VectorStore`."""
    return VectorStore(fake_encoder, fake_decoder)


@pytest.fixture
def populated_vector_store(fake_encoder, fake_decoder):
    """A `VectorStore` populated with deterministic encoded vectors.

    Uses the real encoder layout (types|supertypes|subtypes|mana|colors|rarity)
    so the decoder can describe these vectors without IndexError.
    """
    store = VectorStore(fake_encoder, fake_decoder)
    rng = np.random.default_rng(0)
    specs = {
        "Atraxa": {"card_types": ["creature"], "color_identity": ["W", "U", "B", "G"], "mana": 4, "rarity_index": 4},
        "Sol Ring": {"card_types": ["artifact"], "mana": 1, "rarity_index": 2},
        "Lightning Bolt": {"card_types": ["instant"], "color_identity": ["R"], "mana": 1, "rarity_index": 1},
        "Forest": {"card_types": ["land"], "subtypes": ["forest"], "mana": 0, "rarity_index": 1},
        "Counterspell": {"card_types": ["instant"], "color_identity": ["U"], "mana": 2, "rarity_index": 1},
    }
    for name, kw in specs.items():
        v = make_encoded_vector(**kw)
        # Add a tiny jitter so vectors are not exactly equal for similarity tests
        v = v + rng.normal(0.0, 0.001, size=v.shape).astype(np.float32)
        store.add_vector(name, v)
    return store


@pytest.fixture
def empty_vector_database(fake_encoder, fake_decoder):
    """An empty `VectorDatabase`."""
    return VectorDatabase(fake_encoder, fake_decoder)


@pytest.fixture
def populated_vector_database(fake_encoder, fake_decoder):
    """A `VectorDatabase` populated with deterministic encoded vectors."""
    database = VectorDatabase(fake_encoder, fake_decoder)
    rng = np.random.default_rng(1)
    specs = {
        "Atraxa": {"card_types": ["creature"], "color_identity": ["W", "U", "B", "G"], "mana": 4, "rarity_index": 4},
        "Sol Ring": {"card_types": ["artifact"], "mana": 1, "rarity_index": 2},
        "Lightning Bolt": {"card_types": ["instant"], "color_identity": ["R"], "mana": 1, "rarity_index": 1},
        "Forest": {"card_types": ["land"], "subtypes": ["forest"], "mana": 0, "rarity_index": 1},
        "Counterspell": {"card_types": ["instant"], "color_identity": ["U"], "mana": 2, "rarity_index": 1},
    }
    for name, kw in specs.items():
        v = make_encoded_vector(**kw)
        v = v + rng.normal(0.0, 0.001, size=v.shape).astype(np.float32)
        database.add_vector(name, v)
    return database


@pytest.fixture
def deterministic_torch_seed():
    """Set torch's seed for tests that rely on RNG-driven code paths."""
    torch.manual_seed(0)
    yield
    torch.manual_seed(torch.initial_seed())
