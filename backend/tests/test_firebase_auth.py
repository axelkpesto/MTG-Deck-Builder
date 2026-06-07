"""Tests for `backend.firestore.firebase_auth` API key generation and validation.

The pepper is set to a deterministic test value in conftest (which pytest imports
before this module), so importing `firebase_auth` at module load is safe.
"""
import hmac
import hashlib
import os

from backend.firestore.firebase_auth import generate_api_key, validate_api_key


# ---------------------------------------------------------------------------
# generate_api_key
# ---------------------------------------------------------------------------


class TestGenerateApiKey:
    """`generate_api_key` returns (raw, prefix, hash) triples with mtg_ prefix."""

    def test_returns_three_elements(self):
        """A generated key unpacks into exactly three elements."""
        out = generate_api_key()
        assert len(out) == 3

    def test_raw_starts_with_mtg_prefix(self):
        """The raw key carries the public `mtg_` prefix."""
        raw, _, _ = generate_api_key()
        assert raw.startswith("mtg_")

    def test_prefix_default_length_8(self):
        """The default prefix is 8 chars and matches the raw key's head."""
        raw, prefix, _ = generate_api_key()
        assert len(prefix) == 8
        assert prefix == raw[:8]

    def test_custom_prefix_length(self):
        """A custom `prefix_length` controls the returned prefix length."""
        raw, prefix, _ = generate_api_key(prefix_length=12)
        assert len(prefix) == 12
        assert prefix == raw[:12]

    def test_hash_is_hex_string(self):
        """The digest is a 64-char sha256 hex string that parses as hex."""
        _, _, digest = generate_api_key()
        assert len(digest) == 64
        int(digest, 16)  # parses cleanly

    def test_two_keys_are_different(self):
        """Two generated raw keys are distinct."""
        a, _, _ = generate_api_key()
        b, _, _ = generate_api_key()
        assert a != b

    def test_hash_matches_recomputed_hmac(self):
        """The digest equals HMAC(pepper, raw) recomputed independently."""
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
        """A matching raw/digest pair validates."""
        raw, _, digest = generate_api_key()
        assert validate_api_key(raw, digest) is True

    def test_mismatched_raw_returns_false(self):
        """A raw key paired with another key's digest fails validation."""
        _, _, digest1 = generate_api_key()
        raw2, _, _ = generate_api_key()
        assert validate_api_key(raw2, digest1) is False

    def test_tampered_hash_returns_false(self):
        """A digest replaced with a dummy value fails validation."""
        raw, _, _ = generate_api_key()
        tampered = "0" * 64
        assert validate_api_key(raw, tampered) is False

    def test_empty_raw_returns_false(self):
        """An empty raw key cannot match a real generated digest."""
        _, _, digest = generate_api_key()
        assert validate_api_key("", digest) is False

    def test_constant_time_comparison(self):
        """Unrelated 64-char strings always compare False (uses compare_digest)."""
        assert validate_api_key("mtg_xxx", "a" * 64) is False
