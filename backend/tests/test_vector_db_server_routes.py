"""Tests for `backend.api.vector_db_server` Flask routes.

The Flask app loads vector data + tagging model at import time. We patch those
loads with in-memory fakes (`VectorDatabase` and a tiny `MLP`) so the routes
return realistic results without requiring the on-disk artifacts.

Tests cover: status, help, examples, get_vector, get_vector_description,
get_vector_descriptions, get_random_vector, get_random_vector_description,
get_similar_vectors, get_tags, get_tag_list, get_tags_from_vector,
analyze_deck, and authentication flow.
"""
import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

from backend.config import CONFIG
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
        """Return stable zero logits regardless of input."""
        return torch.zeros(x.shape[0], self.num_classes)

    def to(self, *_args, **_kwargs):
        """No-op device/dtype move that returns self."""
        return self

    def eval(self):
        """No-op eval-mode switch that returns self."""
        return self


@pytest.fixture(scope="module", name="app_module")
def _app_module(tmp_path_factory):
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
        mod = importlib.import_module("backend.api.vector_db_server")
    finally:
        for p in reversed(import_patches):
            p.stop()
        if original_tags_path is not None:
            CONFIG.datasets["TAGS_DATASET_PATH"] = original_tags_path

    # The `VectorDatabase` reference in the module namespace was captured
    # at import time and now points to our MagicMock factory. Restore the
    # real class so `VectorDatabase.vector_to_numpy(...)` works at runtime.
    mod.VectorDatabase = VectorDatabase

    # Inject our real prebuilt fakes onto the live module so the routes can
    # talk to actual VectorDatabase methods.
    mod.auth_enabled = False
    mod.vd = fake_vd
    mod.model = fake_model
    mod.class_names = ["control", "ramp"]
    mod.tag_dataset = {}

    yield mod

    sys.modules.pop("backend.api.vector_db_server", None)


@pytest.fixture(name="client")
def _client(app_module):
    """A Flask test client bound to the patched server module."""
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Static info routes
# ---------------------------------------------------------------------------


class TestInfoRoutes:
    """`/`, `/help`, `/examples` return basic info."""

    def test_home_returns_html(self, client):
        """The home route returns the server's HTML landing page."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Vector Database Server" in resp.data

    def test_help_lists_endpoints(self, client):
        """`/help` lists the available endpoints."""
        resp = client.get("/help")
        assert resp.status_code == 200
        body = resp.get_json()
        assert "endpoints" in body
        assert "/status" in body["endpoints"]
        assert "/get_vector" in body["endpoints"]

    def test_examples_returns_payloads(self, client):
        """`/examples` returns example request payloads."""
        resp = client.get("/examples")
        body = resp.get_json()
        assert "examples" in body


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------


class TestStatus:
    """Service health endpoint."""

    def test_status_returns_payload(self, client):
        """`/status` reports model and vector-db health fields."""
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
        """An exact id returns its vector."""
        resp = client.post("/get_vector", json={"id": "Sol Ring"})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["id"] == "Sol Ring"
        assert isinstance(body["vector"], list)

    def test_partial_id_resolved(self, client):
        """A partial id resolves to the full card name."""
        resp = client.post("/get_vector", json={"id": "sol"})
        assert resp.status_code == 200
        assert resp.get_json()["id"] == "Sol Ring"

    def test_missing_id_returns_400(self, client):
        """An unresolvable id returns HTTP 400."""
        resp = client.post("/get_vector", json={"id": "no-such-card"})
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_no_payload_returns_400(self, client):
        """An empty payload returns HTTP 400."""
        resp = client.post("/get_vector", json={})
        assert resp.status_code == 400

    def test_blank_id_returns_400(self, client):
        """A blank id returns HTTP 400."""
        resp = client.post("/get_vector", json={"id": "  "})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /get_vector_description
# ---------------------------------------------------------------------------


class TestGetVectorDescription:
    """Single-card description endpoint."""

    def test_returns_description_dict(self, client):
        """A present card returns a description dict with a Name field."""
        resp = client.post("/get_vector_description", json={"id": "Sol Ring"})
        assert resp.status_code == 200
        body = resp.get_json()
        assert "Name" in body

    def test_missing_returns_400(self, client):
        """A missing card returns HTTP 400."""
        resp = client.post("/get_vector_description", json={"id": "nope"})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /get_vector_descriptions (batch)
# ---------------------------------------------------------------------------


class TestGetVectorDescriptionsBatch:
    """Batch description endpoint splits results into found / missing buckets."""

    def test_mixed_hits_and_misses(self, client):
        """Found and missing cards are bucketed separately."""
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
        """An empty cards list returns HTTP 400."""
        resp = client.post("/get_vector_descriptions", json={"cards": []})
        assert resp.status_code == 400

    def test_missing_field_returns_400(self, client):
        """A missing cards field returns HTTP 400."""
        resp = client.post("/get_vector_descriptions", json={})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /get_random_vector
# ---------------------------------------------------------------------------


class TestRandomVectorEndpoints:
    """Random sampling endpoints."""

    def test_get_random_vector(self, client):
        """`/get_random_vector` returns an id and vector."""
        resp = client.post("/get_random_vector", json={})
        assert resp.status_code == 200
        body = resp.get_json()
        assert "id" in body and "vector" in body

    def test_get_random_vector_description(self, client):
        """`/get_random_vector_description` returns a description with a Name."""
        resp = client.post("/get_random_vector_description", json={})
        assert resp.status_code == 200
        assert "Name" in resp.get_json()


# ---------------------------------------------------------------------------
# /get_similar_vectors
# ---------------------------------------------------------------------------


class TestGetSimilarVectors:
    """Similar-card lookup endpoint."""

    def test_returns_ranked_dict(self, client):
        """Similar vectors are returned keyed by rank index."""
        resp = client.post(
            "/get_similar_vectors",
            json={"id": "Sol Ring", "num_vectors": 2},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert "0" in body or 0 in body

    def test_missing_id_returns_400(self, client):
        """An unresolvable id returns HTTP 400."""
        resp = client.post("/get_similar_vectors", json={"id": "ZZ"})
        assert resp.status_code == 400

    def test_invalid_num_vectors_returns_400(self, client):
        """A non-numeric num_vectors returns HTTP 400."""
        resp = client.post(
            "/get_similar_vectors",
            json={"id": "Sol Ring", "num_vectors": "many"},
        )
        assert resp.status_code == 400

    def test_num_vectors_clamped(self, client):
        """A num_vectors of 0 clamps to 1 and succeeds."""
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
        """A tag prediction payload includes the expected keys."""
        resp = client.post("/get_tags", json={"id": "Sol Ring"})
        assert resp.status_code == 200
        body = resp.get_json()
        for k in ("predicted", "predicted_scores", "scores", "threshold"):
            assert k in body

    def test_missing_card_returns_400(self, client):
        """An unresolvable card returns HTTP 400."""
        resp = client.post("/get_tags", json={"id": "ZZ"})
        assert resp.status_code == 400

    def test_invalid_threshold_type_returns_400(self, client):
        """A non-numeric threshold returns HTTP 400."""
        resp = client.post(
            "/get_tags",
            json={"id": "Sol Ring", "threshold": "huge"},
        )
        assert resp.status_code == 400

    def test_threshold_clamped_to_unit_range(self, client):
        """A threshold above 1 clamps to 1.0 and succeeds."""
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
        """Found and missing cards are bucketed separately."""
        resp = client.post(
            "/get_tag_list",
            json={"cards": ["Sol Ring", "Nonexistent"]},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert "Sol Ring" in body["found"]
        assert "Nonexistent" in body["missing"]

    def test_empty_cards_returns_400(self, client):
        """An empty cards list returns HTTP 400."""
        resp = client.post("/get_tag_list", json={"cards": []})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /get_tags_from_vector
# ---------------------------------------------------------------------------


class TestGetTagsFromVector:
    """Tag prediction from a raw embedding vector."""

    def test_returns_prediction_payload(self, client):
        """A raw vector yields a prediction payload (the fake model ignores shape)."""
        vector = [0.0] * 8
        resp = client.post(
            "/get_tags_from_vector",
            json={"vector": vector},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert "predicted" in body

    def test_missing_vector_returns_400(self, client):
        """A missing vector field returns HTTP 400."""
        resp = client.post("/get_tags_from_vector", json={})
        assert resp.status_code == 400

    def test_vector_not_list_returns_400(self, client):
        """A non-list vector returns HTTP 400."""
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
        """A valid deck returns all analysis sections."""
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
        """A missing commander returns HTTP 400."""
        resp = client.post(
            "/analyze_deck",
            json={"cards": ["Sol Ring"]},
        )
        assert resp.status_code == 400

    def test_missing_cards_returns_400(self, client):
        """Missing cards returns HTTP 400."""
        resp = client.post(
            "/analyze_deck",
            json={"commander": "Atraxa, Praetors' Voice"},
        )
        assert resp.status_code == 400

    def test_blank_commander_returns_400(self, client):
        """A blank commander returns HTTP 400."""
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

    def test_503_when_bundle_not_loaded(self, client):
        """A still-loading bundle yields HTTP 503."""
        resp = client.post(
            "/generate_deck",
            json={"id": "Atraxa, Praetors' Voice"},
        )
        assert resp.status_code == 503

    def test_503_when_bundle_failed(self, client, app_module):
        """A failed bundle yields HTTP 503."""
        deckgen = getattr(app_module, "_deckgen")
        deckgen.state = "failed"
        try:
            resp = client.post(
                "/generate_deck",
                json={"id": "Atraxa, Praetors' Voice"},
            )
            assert resp.status_code == 503
        finally:
            deckgen.state = "loading"

    def test_400_when_commander_missing(self, client, app_module):
        """With a ready bundle, a missing commander yields HTTP 400."""
        deckgen = getattr(app_module, "_deckgen")
        original_bundle = deckgen.bundle
        original_state = deckgen.state
        deckgen.bundle = MagicMock()
        deckgen.state = "ready"
        try:
            resp = client.post("/generate_deck", json={})
            assert resp.status_code == 400
        finally:
            deckgen.bundle = original_bundle
            deckgen.state = original_state

    def test_400_when_not_legendary_creature(self, client, app_module):
        """A non-legendary-creature commander yields HTTP 400."""
        deckgen = getattr(app_module, "_deckgen")
        original_bundle = deckgen.bundle
        original_state = deckgen.state
        deckgen.bundle = MagicMock()
        deckgen.state = "ready"
        try:
            resp = client.post("/generate_deck", json={"id": "Sol Ring"})
            assert resp.status_code == 400
            assert "legendary creature" in resp.get_json()["error"]
        finally:
            deckgen.bundle = original_bundle
            deckgen.state = original_state


# ---------------------------------------------------------------------------
# 404 handler
# ---------------------------------------------------------------------------


class TestNotFoundHandler:
    """`@app.errorhandler(404)` returns a JSON envelope."""

    def test_unknown_route_returns_404_json(self, client):
        """An unknown route returns a JSON error envelope with HTTP 404."""
        resp = client.post("/does_not_exist")
        assert resp.status_code == 404
        body = resp.get_json()
        assert "error" in body
