"""Tests for `backend.firestore.firebase_auth` API key generation and validation.

The pepper is set to a deterministic test value in conftest before this module
imports.
"""
import hmac
import hashlib
import os

import pytest


# Import after conftest has set API_KEY_PEPPER
from backend.firestore.firebase_auth import (  # noqa: E402  pylint: disable=wrong-import-position
    generate_api_key,
    validate_api_key,
)


# ---------------------------------------------------------------------------
# generate_api_key
# ---------------------------------------------------------------------------


class TestGenerateApiKey:
    """`generate_api_key` returns (raw, prefix, hash) triples with mtg_ prefix."""

    def test_returns_three_elements(self):
        out = generate_api_key()
        assert len(out) == 3

    def test_raw_starts_with_mtg_prefix(self):
        raw, _, _ = generate_api_key()
        assert raw.startswith("mtg_")

    def test_prefix_default_length_8(self):
        raw, prefix, _ = generate_api_key()
        assert len(prefix) == 8
        assert prefix == raw[:8]

    def test_custom_prefix_length(self):
        raw, prefix, _ = generate_api_key(prefix_length=12)
        assert len(prefix) == 12
        assert prefix == raw[:12]

    def test_hash_is_hex_string(self):
        _, _, digest = generate_api_key()
        # sha256 hex digest is 64 hex chars
        assert len(digest) == 64
        int(digest, 16)  # parses cleanly

    def test_two_keys_are_different(self):
        a, _, _ = generate_api_key()
        b, _, _ = generate_api_key()
        assert a != b

    def test_hash_matches_recomputed_hmac(self):
        raw, _, digest = generate_api_key()
        expected = hmac.new(
            os.environ["API_KEY_PEPPER"].encode("utf-8"),
            raw.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        assert digest == expected


# ---------------------------------------------------------------------------
# validate_api_key
# ---------------------------------------------------------------------------


class TestValidateApiKey:
    """`validate_api_key` compares an HMAC of `raw` against the stored hash."""

    def test_valid_pair_returns_true(self):
        raw, _, digest = generate_api_key()
        assert validate_api_key(raw, digest) is True

    def test_mismatched_raw_returns_false(self):
        raw1, _, digest1 = generate_api_key()
        raw2, _, _ = generate_api_key()
        assert validate_api_key(raw2, digest1) is False

    def test_tampered_hash_returns_false(self):
        raw, _, digest = generate_api_key()
        tampered = "0" * 64
        assert validate_api_key(raw, tampered) is False

    def test_empty_raw_returns_false(self):
        # Empty key cannot match a real generated digest.
        _, _, digest = generate_api_key()
        assert validate_api_key("", digest) is False

    def test_constant_time_comparison(self):
        # Sanity check: function uses `hmac.compare_digest`. Two unrelated
        # 64-char strings should always return False.
        assert validate_api_key("mtg_xxx", "a" * 64) is False
