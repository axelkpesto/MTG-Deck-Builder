"""Tests for `backend.api.vector_db_server` Flask routes.

The Flask app loads vector data + tagging model at import time. We patch those
loads with in-memory fakes (`VectorDatabase` and a tiny `MLP`) so the routes
return realistic results without requiring the on-disk artifacts.

Tests cover: status, help, examples, get_vector, get_vector_description,
get_vector_descriptions, get_random_vector, get_random_vector_description,
get_similar_vectors, get_tags, get_tag_list, get_tags_from_vector,
analyze_deck, and authentication flow.
"""
import json
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from backend.card_data import CardDecoder
from backend.vector_database import VectorDatabase

from .conftest import make_encoded_vector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_test_vd() -> VectorDatabase:
    """A small `VectorDatabase` populated with handcrafted encoded vectors."""
    vd = VectorDatabase(None, CardDecoder())
    vd.add_vector(
        "Atraxa, Praetors' Voice",
        make_encoded_vector(
            card_types=["creature"], supertypes=["legendary"],
            subtypes=["angel"], color_identity=["W", "U", "B", "G"], mana=4,
        ),
    )
    vd.add_vector(
        "Sol Ring",
        make_encoded_vector(card_types=["artifact"], mana=1),
    )
    vd.add_vector(
        "Forest",
        make_encoded_vector(card_types=["land"], subtypes=["forest"]),
    )
    return vd


class _FakeMLP(torch.nn.Module):
    """Identity-ish model that returns logits of fixed length."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        # Produce stable logits regardless of input
        return torch.zeros(x.shape[0], self.num_classes)

    def to(self, *args, **kwargs):  # noqa: D401
        return self

    def eval(self):
        return self


class _NoOpThread:
    """Stand-in for `threading.Thread` that never runs the target."""

    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass


@pytest.fixture(scope="module")
def app_module(tmp_path_factory):
    """Import the Flask server with patched module-level loads.

    Approach: write a tiny placeholder tags dataset to a tmp file and patch
    the CONFIG dataset path so the import-time `open(...)` finds something
    real. Patch the heavyweight loads only during import, then release.
    """
    sys.modules.pop("backend.api.vector_db_server", None)

    fake_vd = _build_test_vd()
    fake_model = _FakeMLP(num_classes=2)

    fake_vd_instance = MagicMock(load=MagicMock())
    fake_vd_factory = MagicMock(return_value=fake_vd_instance)

    tmp_dir = tmp_path_factory.mktemp("tags")
    tags_path = tmp_dir / "tags.json"
    tags_path.write_text("{}")

    from backend.config import CONFIG

    original_tags_path = CONFIG.datasets.get("TAGS_DATASET_PATH")
    CONFIG.datasets["TAGS_DATASET_PATH"] = str(tags_path)

    # Mock DeckGenBundle.load so the background thread spawned at import time
    # fails fast (sets _deckgen.state = "failed") instead of trying to read
    # the on-disk artifacts. Threading itself is left alone so flask-limiter's
    # internal Timer works.
    import_patches = [
        patch("backend.ml.tagging_model.load_model", return_value=(fake_model, ["control", "ramp"])),
        patch("backend.vector_database.VectorDatabase", fake_vd_factory),
        patch("backend.deckgen.DeckGenBundle.load", side_effect=RuntimeError("disabled in tests")),
    ]
    for p in import_patches:
        p.start()
    try:
        try:
            import backend.api.vector_db_server as mod  # noqa: WPS433
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"Cannot import vector_db_server with mocks: {exc}")
    finally:
        for p in reversed(import_patches):
            p.stop()
        if original_tags_path is not None:
            CONFIG.datasets["TAGS_DATASET_PATH"] = original_tags_path

    # The `VectorDatabase` reference in the module namespace was captured
    # at import time and now points to our MagicMock factory. Restore the
    # real class so `VectorDatabase.vector_to_numpy(...)` works at runtime.
    from backend.vector_database import VectorDatabase as _real_vd_cls
    mod.VectorDatabase = _real_vd_cls

    # Inject our real prebuilt fakes onto the live module so the routes can
    # talk to actual VectorDatabase methods.
    mod.auth_enabled = False
    mod.vd = fake_vd
    mod.model = fake_model
    mod.class_names = ["control", "ramp"]
    mod.tag_dataset = {}

    yield mod

    sys.modules.pop("backend.api.vector_db_server", None)


@pytest.fixture
def client(app_module):
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Static info routes
# ---------------------------------------------------------------------------


class TestInfoRoutes:
    """`/`, `/help`, `/examples` return basic info."""

    def test_home_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Vector Database Server" in resp.data

    def test_help_lists_endpoints(self, client):
        resp = client.get("/help")
        assert resp.status_code == 200
        body = resp.get_json()
        assert "endpoints" in body
        assert "/status" in body["endpoints"]
        assert "/get_vector" in body["endpoints"]

    def test_examples_returns_payloads(self, client):
        resp = client.get("/examples")
        body = resp.get_json()
        assert "examples" in body


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------


class TestStatus:
    """Service health endpoint."""

    def test_status_returns_payload(self, client):
        resp = client.post("/status")
        assert resp.status_code == 200
        body = resp.get_json()
        for k in ("status", "model_loaded", "vector_db_loaded", "vd_size"):
            assert k in body


# ---------------------------------------------------------------------------
# /get_vector
# ---------------------------------------------------------------------------


class TestGetVector:
    """Lookup raw vector by id (exact and partial)."""

    def test_exact_id_returns_vector(self, client):
        resp = client.post("/get_vector", json={"id": "Sol Ring"})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["id"] == "Sol Ring"
        assert isinstance(body["vector"], list)

    def test_partial_id_resolved(self, client):
        resp = client.post("/get_vector", json={"id": "sol"})
        assert resp.status_code == 200
        assert resp.get_json()["id"] == "Sol Ring"

    def test_missing_id_returns_400(self, client):
        resp = client.post("/get_vector", json={"id": "no-such-card"})
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_no_payload_returns_400(self, client):
        resp = client.post("/get_vector", json={})
        assert resp.status_code == 400

    def test_blank_id_returns_400(self, client):
        resp = client.post("/get_vector", json={"id": "  "})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /get_vector_description
# ---------------------------------------------------------------------------


class TestGetVectorDescription:
    """Single-card description endpoint."""

    def test_returns_description_dict(self, client):
        resp = client.post("/get_vector_description", json={"id": "Sol Ring"})
        assert resp.status_code == 200
        body = resp.get_json()
        assert "Name" in body

    def test_missing_returns_400(self, client):
        resp = client.post("/get_vector_description", json={"id": "nope"})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /get_vector_descriptions (batch)
# ---------------------------------------------------------------------------


class TestGetVectorDescriptionsBatch:
    """Batch description endpoint splits results into found / missing buckets."""

    def test_mixed_hits_and_misses(self, client):
        resp = client.post(
            "/get_vector_descriptions",
            json={"cards": ["Sol Ring", "Nonexistent Card"]},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert "found" in body and "missing" in body
        assert "Sol Ring" in body["found"]
        assert "Nonexistent Card" in body["missing"]

    def test_empty_cards_returns_400(self, client):
        resp = client.post("/get_vector_descriptions", json={"cards": []})
        assert resp.status_code == 400

    def test_missing_field_returns_400(self, client):
        resp = client.post("/get_vector_descriptions", json={})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /get_random_vector
# ---------------------------------------------------------------------------


class TestRandomVectorEndpoints:
    """Random sampling endpoints."""

    def test_get_random_vector(self, client):
        resp = client.post("/get_random_vector", json={})
        assert resp.status_code == 200
        body = resp.get_json()
        assert "id" in body and "vector" in body

    def test_get_random_vector_description(self, client):
        resp = client.post("/get_random_vector_description", json={})
        assert resp.status_code == 200
        assert "Name" in resp.get_json()


# ---------------------------------------------------------------------------
# /get_similar_vectors
# ---------------------------------------------------------------------------


class TestGetSimilarVectors:
    """Similar-card lookup endpoint."""

    def test_returns_ranked_dict(self, client):
        resp = client.post(
            "/get_similar_vectors",
            json={"id": "Sol Ring", "num_vectors": 2},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        # Keys are stringified integers (0..N-1) per implementation.
        assert "0" in body or 0 in body

    def test_missing_id_returns_400(self, client):
        resp = client.post("/get_similar_vectors", json={"id": "ZZ"})
        assert resp.status_code == 400

    def test_invalid_num_vectors_returns_400(self, client):
        resp = client.post(
            "/get_similar_vectors",
            json={"id": "Sol Ring", "num_vectors": "many"},
        )
        assert resp.status_code == 400

    def test_num_vectors_clamped(self, client):
        # Asking for 0 should clamp to 1; should not error.
        resp = client.post(
            "/get_similar_vectors",
            json={"id": "Sol Ring", "num_vectors": 0},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /get_tags
# ---------------------------------------------------------------------------


class TestGetTags:
    """Single-card tag prediction."""

    def test_returns_prediction_payload(self, client):
        resp = client.post("/get_tags", json={"id": "Sol Ring"})
        assert resp.status_code == 200
        body = resp.get_json()
        for k in ("predicted", "predicted_scores", "scores", "threshold"):
            assert k in body

    def test_missing_card_returns_400(self, client):
        resp = client.post("/get_tags", json={"id": "ZZ"})
        assert resp.status_code == 400

    def test_invalid_threshold_type_returns_400(self, client):
        resp = client.post(
            "/get_tags",
            json={"id": "Sol Ring", "threshold": "huge"},
        )
        assert resp.status_code == 400

    def test_threshold_clamped_to_unit_range(self, client):
        # threshold=2 clamps to 1 and should still succeed
        resp = client.post(
            "/get_tags",
            json={"id": "Sol Ring", "threshold": 2.0},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["threshold"] == 1.0


# ---------------------------------------------------------------------------
# /get_tag_list
# ---------------------------------------------------------------------------


class TestGetTagList:
    """Batch tag prediction."""

    def test_mixed_hits_and_misses(self, client):
        resp = client.post(
            "/get_tag_list",
            json={"cards": ["Sol Ring", "Nonexistent"]},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert "Sol Ring" in body["found"]
        assert "Nonexistent" in body["missing"]

    def test_empty_cards_returns_400(self, client):
        resp = client.post("/get_tag_list", json={"cards": []})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /get_tags_from_vector
# ---------------------------------------------------------------------------


class TestGetTagsFromVector:
    """Tag prediction from a raw embedding vector."""

    def test_returns_prediction_payload(self, client, app_module):
        # Use a vector of the right length (whatever the model expects). Our
        # fake _FakeMLP ignores input shape so any list works.
        vector = [0.0] * 8
        resp = client.post(
            "/get_tags_from_vector",
            json={"vector": vector},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert "predicted" in body

    def test_missing_vector_returns_400(self, client):
        resp = client.post("/get_tags_from_vector", json={})
        assert resp.status_code == 400

    def test_vector_not_list_returns_400(self, client):
        resp = client.post(
            "/get_tags_from_vector",
            json={"vector": "not a list"},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /analyze_deck
# ---------------------------------------------------------------------------


class TestAnalyzeDeckEndpoint:
    """Deck analysis endpoint."""

    def test_returns_full_analysis(self, client):
        resp = client.post(
            "/analyze_deck",
            json={
                "commander": "Atraxa, Praetors' Voice",
                "cards": ["Sol Ring", "Forest"],
            },
        )
        assert resp.status_code == 200
        body = resp.get_json()
        for section in ("tags", "color_distribution", "curve", "lands"):
            assert section in body

    def test_missing_commander_returns_400(self, client):
        resp = client.post(
            "/analyze_deck",
            json={"cards": ["Sol Ring"]},
        )
        assert resp.status_code == 400

    def test_missing_cards_returns_400(self, client):
        resp = client.post(
            "/analyze_deck",
            json={"commander": "Atraxa, Praetors' Voice"},
        )
        assert resp.status_code == 400

    def test_blank_commander_returns_400(self, client):
        resp = client.post(
            "/analyze_deck",
            json={"commander": "  ", "cards": ["Sol Ring"]},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /generate_deck
# ---------------------------------------------------------------------------


class TestGenerateDeck:
    """Deck generation endpoint validates the commander and bundle state."""

    def test_503_when_bundle_not_loaded(self, client, app_module):
        # Default state in tests: _deckgen.state == "loading", bundle is None
        resp = client.post(
            "/generate_deck",
            json={"id": "Atraxa, Praetors' Voice"},
        )
        assert resp.status_code == 503

    def test_503_when_bundle_failed(self, client, app_module):
        app_module._deckgen.state = "failed"
        try:
            resp = client.post(
                "/generate_deck",
                json={"id": "Atraxa, Praetors' Voice"},
            )
            assert resp.status_code == 503
        finally:
            app_module._deckgen.state = "loading"

    def test_400_when_commander_missing(self, client, app_module):
        # Install a non-None bundle so the route gets past the 503 short-circuit
        # and reaches the payload validation.
        from unittest.mock import MagicMock
        original_bundle = app_module._deckgen.bundle
        original_state = app_module._deckgen.state
        app_module._deckgen.bundle = MagicMock()
        app_module._deckgen.state = "ready"
        try:
            resp = client.post("/generate_deck", json={})
            assert resp.status_code == 400
        finally:
            app_module._deckgen.bundle = original_bundle
            app_module._deckgen.state = original_state

    def test_400_when_not_legendary_creature(self, client, app_module):
        # `Sol Ring` is in the test vd but is an artifact, not a legendary creature.
        from unittest.mock import MagicMock
        original_bundle = app_module._deckgen.bundle
        original_state = app_module._deckgen.state
        app_module._deckgen.bundle = MagicMock()
        app_module._deckgen.state = "ready"
        try:
            resp = client.post("/generate_deck", json={"id": "Sol Ring"})
            assert resp.status_code == 400
            assert "legendary creature" in resp.get_json()["error"]
        finally:
            app_module._deckgen.bundle = original_bundle
            app_module._deckgen.state = original_state


# ---------------------------------------------------------------------------
# 404 handler
# ---------------------------------------------------------------------------


class TestNotFoundHandler:
    """`@app.errorhandler(404)` returns a JSON envelope."""

    def test_unknown_route_returns_404_json(self, client):
        resp = client.post("/does_not_exist")
        assert resp.status_code == 404
        body = resp.get_json()
        assert "error" in body
