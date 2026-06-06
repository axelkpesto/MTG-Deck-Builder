"""Tests for pure-function helpers in `backend.api.vector_db_server`.

The Flask app module loads the vector database, tagging model, and deck-gen
bundle at import time. We patch those load points so the module imports cleanly
in test environments that don't ship the data artifacts. The Flask routes
themselves are exercised in `test_vector_db_server_routes.py`.
"""
import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture(scope="module")
def server_module():
    """Import `backend.api.vector_db_server` with all heavy module-level loads patched."""

    # Make sure the module re-imports under patches.
    sys.modules.pop("backend.api.vector_db_server", None)

    fake_vd = MagicMock()
    fake_vd.__len__.return_value = 0
    fake_vd.__contains__.return_value = False

    fake_model = MagicMock()

    with patch("backend.api.vector_db_server.load_model", return_value=(fake_model, ["tag_a", "tag_b"])), \
         patch("backend.api.vector_db_server.VectorDatabase") as mock_vd_cls, \
         patch("backend.api.vector_db_server.threading.Thread"), \
         patch("backend.api.vector_db_server.open", create=True), \
         patch("backend.api.vector_db_server.json.load", return_value={}):
        mock_vd_cls.return_value = fake_vd
        try:
            import backend.api.vector_db_server as mod  # noqa: WPS433
        except Exception as exc:  # pragma: no cover - depends on env
            pytest.skip(f"Could not import vector_db_server even with mocks: {exc}")
        yield mod

    sys.modules.pop("backend.api.vector_db_server", None)


# ---------------------------------------------------------------------------
# clamp_int / clamp_float
# ---------------------------------------------------------------------------


class TestClampHelpers:
    """Bounded integer and float clamping."""

    @pytest.mark.parametrize("x,lo,hi,expected", [
        (5, 0, 10, 5),
        (-1, 0, 10, 0),
        (11, 0, 10, 10),
        (5, 5, 5, 5),
    ])
    def test_clamp_int(self, server_module, x, lo, hi, expected):
        assert server_module.clamp_int(x, lo, hi) == expected

    @pytest.mark.parametrize("x,lo,hi,expected", [
        (0.5, 0.0, 1.0, 0.5),
        (-1.0, 0.0, 1.0, 0.0),
        (2.0, 0.0, 1.0, 1.0),
        (0.0, 0.0, 1.0, 0.0),
    ])
    def test_clamp_float(self, server_module, x, lo, hi, expected):
        assert server_module.clamp_float(x, lo, hi) == expected


# ---------------------------------------------------------------------------
# format_id (title-case with transition-word handling)
# ---------------------------------------------------------------------------


class TestFormatId:
    """`format_id` mirrors `CardDecoder._title_case` for API-side id normalization."""

    def test_simple_title_case(self, server_module):
        assert server_module.format_id("sol ring") == "Sol Ring"

    def test_transition_words_lowercased(self, server_module):
        # The server's transition set is {of, the, in, on, at, to, for, and, but, or, nor}.
        assert server_module.format_id("city of brass") == "City of Brass"
        assert server_module.format_id("seat of the synod") == "Seat of the Synod"

    def test_leading_transition_word_capitalized(self, server_module):
        # First word always capitalized, even transition words.
        assert server_module.format_id("of one mind").split(" ")[0] == "Of"

    def test_already_titled_string_remains_titled(self, server_module):
        # Idempotency: applying twice should give the same result
        once = server_module.format_id("magnus the red")
        twice = server_module.format_id(once)
        assert once == twice

    def test_single_word(self, server_module):
        assert server_module.format_id("forest") == "Forest"


# ---------------------------------------------------------------------------
# parse_card_list_payload
# ---------------------------------------------------------------------------


class TestParseCardListPayload:
    """Validate request payload for batch-cards endpoints."""

    def test_returns_stripped_non_empty_strings(self, server_module):
        out = server_module.parse_card_list_payload(
            {"cards": ["  A  ", "B", "  ", "C"]}
        )
        assert out == ["A", "B", "C"]

    def test_missing_cards_key_raises(self, server_module):
        with pytest.raises(ValueError):
            server_module.parse_card_list_payload({})

    def test_not_a_list_raises(self, server_module):
        with pytest.raises(ValueError):
            server_module.parse_card_list_payload({"cards": "Atraxa"})

    def test_non_string_entry_raises(self, server_module):
        with pytest.raises(ValueError):
            server_module.parse_card_list_payload({"cards": ["A", 42]})

    def test_all_whitespace_entries_raises(self, server_module):
        with pytest.raises(ValueError):
            server_module.parse_card_list_payload({"cards": ["   ", "\t"]})

    def test_empty_list_raises(self, server_module):
        with pytest.raises(ValueError):
            server_module.parse_card_list_payload({"cards": []})


# ---------------------------------------------------------------------------
# parse_required_card_id
# ---------------------------------------------------------------------------


class TestParseRequiredCardId:
    """Validate request payload for single-card endpoints."""

    def test_returns_stripped_value(self, server_module):
        assert server_module.parse_required_card_id({"id": "  Atraxa  "}) == "Atraxa"

    def test_missing_raises(self, server_module):
        with pytest.raises(ValueError):
            server_module.parse_required_card_id({})

    def test_blank_string_raises(self, server_module):
        with pytest.raises(ValueError):
            server_module.parse_required_card_id({"id": "   "})

    def test_non_string_raises(self, server_module):
        with pytest.raises(ValueError):
            server_module.parse_required_card_id({"id": 123})

    def test_custom_field_name(self, server_module):
        assert server_module.parse_required_card_id(
            {"commander": "Atraxa"}, field_name="commander"
        ) == "Atraxa"


# ---------------------------------------------------------------------------
# error
# ---------------------------------------------------------------------------


class TestErrorHelper:
    """`error` returns a Flask-style JSON tuple."""

    def test_returns_tuple(self, server_module):
        with server_module.app.app_context():
            response, status = server_module.error("bad input", 400)
            assert status == 400
            data = response.get_json()
            assert data == {"error": "bad input"}

    def test_default_status_is_400(self, server_module):
        with server_module.app.app_context():
            _, status = server_module.error("oh no")
            assert status == 400


# ---------------------------------------------------------------------------
# resolve_card_id
# ---------------------------------------------------------------------------


class TestResolveCardId:
    """Exact, formatted, and partial-search resolution paths."""

    @pytest.fixture(autouse=True)
    def _mock_vd(self, server_module):
        """Replace the module-level vd with a MagicMock for these tests."""
        original = server_module.vd
        mock = MagicMock()
        server_module.vd = mock
        yield mock
        server_module.vd = original

    def test_exact_match_in_vd(self, server_module, _mock_vd):
        _mock_vd.__contains__.side_effect = lambda k: k == "Atraxa"
        assert server_module.resolve_card_id("Atraxa") == "Atraxa"

    def test_formatted_match(self, server_module, _mock_vd):
        # Raw "atraxa" not in vd; "Atraxa" (format_id output) is.
        _mock_vd.__contains__.side_effect = lambda k: k == "Atraxa"
        assert server_module.resolve_card_id("atraxa") == "Atraxa"

    def test_fallback_to_find_id(self, server_module, _mock_vd):
        _mock_vd.__contains__.return_value = False
        _mock_vd.find_id.return_value = "Sol Ring"
        assert server_module.resolve_card_id("sol") == "Sol Ring"

    def test_blank_raises(self, server_module, _mock_vd):
        with pytest.raises(KeyError):
            server_module.resolve_card_id("   ")


# ---------------------------------------------------------------------------
# get_api_key_from_request
# ---------------------------------------------------------------------------


class TestGetApiKeyFromRequest:
    """Auth header extraction handles bearer and X-API-KEY forms."""

    def _fake_request(self, headers):
        # Mirror Flask's `request.headers.get(key)` -> None for missing keys.
        req = MagicMock()
        sentinel = object()

        def _get(k, default=sentinel):
            v = headers.get(k)
            if v is not None:
                return v
            return default if default is not sentinel else None

        req.headers.get = _get
        return req

    def test_authorization_bearer(self, server_module):
        req = self._fake_request({"Authorization": "Bearer secret-key"})
        assert server_module.get_api_key_from_request(req) == "secret-key"

    def test_authorization_bearer_lowercase(self, server_module):
        req = self._fake_request({"Authorization": "bearer secret-key"})
        assert server_module.get_api_key_from_request(req) == "secret-key"

    def test_x_api_key_header(self, server_module):
        req = self._fake_request({"X-API-KEY": "abc"})
        assert server_module.get_api_key_from_request(req) == "abc"

    def test_none_when_missing(self, server_module):
        req = self._fake_request({})
        assert server_module.get_api_key_from_request(req) is None
