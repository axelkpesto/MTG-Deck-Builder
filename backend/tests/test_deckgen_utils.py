"""Tests for `backend.deckgen.utils` helper functions."""
import json
import random

import numpy as np
import pytest
import torch

from backend.deckgen.utils import (
    allowed_basic_land_types,
    basic_land_type,
    clamp_int,
    duplicate_penalty,
    extract_basic_ratio,
    extract_curve_counts,
    extract_land_count,
    extract_tag_count,
    is_basic_land_name,
    mana_value_bucket,
    safe_read_json,
    set_seed,
)


# ---------------------------------------------------------------------------
# mana_value_bucket
# ---------------------------------------------------------------------------


class TestManaValueBucket:
    """`mana_value_bucket` rounds, clamps to [0, 6], and treats 6+ as 6."""

    @pytest.mark.parametrize("mv,expected", [
        (0, 0), (0.4, 0), (0.5, 0), (1, 1), (3.4, 3), (3.5, 4),
        (5, 5), (6, 6), (6.9, 6), (10, 6), (100, 6),
    ])
    def test_buckets(self, mv, expected):
        assert mana_value_bucket(mv) == expected

    def test_negative_clamped_to_zero(self):
        assert mana_value_bucket(-3) == 0

    def test_accepts_string_numbers(self):
        # `float()` accepts strings; document this behavior so a refactor stays safe.
        assert mana_value_bucket("4") == 4


# ---------------------------------------------------------------------------
# safe_read_json
# ---------------------------------------------------------------------------


class TestSafeReadJson:
    """Reads JSON or raises FileNotFoundError; never silently returns None."""

    def test_round_trip(self, tmp_path):
        p = tmp_path / "x.json"
        p.write_text(json.dumps({"a": 1, "b": [2, 3]}))
        out = safe_read_json(str(p))
        assert out == {"a": 1, "b": [2, 3]}

    def test_raises_when_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            safe_read_json(str(tmp_path / "nope.json"))

    def test_raises_on_invalid_json(self, tmp_path):
        p = tmp_path / "x.json"
        p.write_text("{not valid json")
        with pytest.raises(json.JSONDecodeError):
            safe_read_json(str(p))


# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------


class TestSetSeed:
    """`set_seed` produces reproducible Python, NumPy, and Torch RNG state."""

    def test_python_random_reproducible(self):
        set_seed(42)
        a = [random.random() for _ in range(5)]
        set_seed(42)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_numpy_reproducible(self):
        set_seed(7)
        a = np.random.rand(4)
        set_seed(7)
        b = np.random.rand(4)
        np.testing.assert_array_equal(a, b)

    def test_torch_reproducible(self):
        set_seed(123)
        a = torch.randn(3)
        set_seed(123)
        b = torch.randn(3)
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# clamp_int
# ---------------------------------------------------------------------------


class TestClampInt:
    """`clamp_int` enforces [lo, hi]."""

    @pytest.mark.parametrize("v,lo,hi,expected", [
        (5, 0, 10, 5),
        (-1, 0, 10, 0),
        (11, 0, 10, 10),
        (0, 0, 10, 0),
        (10, 0, 10, 10),
    ])
    def test_clamping(self, v, lo, hi, expected):
        assert clamp_int(v, lo, hi) == expected

    def test_lo_equals_hi(self):
        assert clamp_int(99, 5, 5) == 5


# ---------------------------------------------------------------------------
# is_basic_land_name / basic_land_type
# ---------------------------------------------------------------------------


class TestBasicLandNameHelpers:
    """Thin wrappers over `CardFields.is_basic_land` and `basic_type_name`."""

    @pytest.mark.parametrize("name", ["Forest", "forest", "  Plains  "])
    def test_recognizes_basics_case_and_whitespace(self, name):
        assert is_basic_land_name(name) is True

    @pytest.mark.parametrize("name", ["Sol Ring", "", "Strip Mine"])
    def test_rejects_non_basics(self, name):
        assert is_basic_land_name(name) is False

    def test_basic_land_type_title_cased(self):
        assert basic_land_type("forest") == "Forest"
        assert basic_land_type("ISLAND") == "Island"


# ---------------------------------------------------------------------------
# duplicate_penalty
# ---------------------------------------------------------------------------


class TestDuplicatePenalty:
    """Power-law duplicate penalty."""

    def test_zero_extra_returns_zero(self):
        assert duplicate_penalty(extra=0, lam=2.0, power=1.0) == 0.0

    def test_negative_extra_returns_zero(self):
        assert duplicate_penalty(extra=-5, lam=2.0, power=1.0) == 0.0

    def test_zero_lambda_returns_zero(self):
        assert duplicate_penalty(extra=5, lam=0.0, power=1.0) == 0.0

    def test_typical_value(self):
        # 2 * (4 ^ 1.5) = 2 * 8 = 16
        result = duplicate_penalty(extra=4, lam=2.0, power=1.5)
        assert pytest.approx(result, rel=1e-6) == 16.0

    def test_returns_float(self):
        assert isinstance(duplicate_penalty(2, 1.0, 1.0), float)


# ---------------------------------------------------------------------------
# allowed_basic_land_types
# ---------------------------------------------------------------------------


class TestAllowedBasicLandTypes:
    """Map a color-identity tensor to allowed basic land names."""

    def _one_hot(self, colors):
        from backend.card_data import CardFields
        idents = CardFields.color_identities()
        t = torch.zeros(len(idents))
        for c in colors:
            t[idents.index(c)] = 1
        return t

    def test_all_five_colors_returns_all_basics(self):
        t = self._one_hot(["W", "U", "B", "R", "G"])
        out = allowed_basic_land_types(t)
        assert set(out) == {"Plains", "Island", "Swamp", "Mountain", "Forest"}

    def test_colorless_falls_back_to_wastes(self):
        t = self._one_hot([])  # no bits set
        assert allowed_basic_land_types(t) == ["Wastes"]

    def test_skips_colorless_bit(self):
        t = self._one_hot(["C"])  # only 'C' set
        assert allowed_basic_land_types(t) == ["Wastes"]

    def test_partial_color_identity(self):
        t = self._one_hot(["U", "B"])
        out = allowed_basic_land_types(t)
        assert set(out) == {"Island", "Swamp"}


# ---------------------------------------------------------------------------
# extract_* helpers
# ---------------------------------------------------------------------------


class TestExtractHelpers:
    """Each `extract_*` pulls a single value out of the stats payload."""

    def _stats(self):
        return {
            "lands": {
                "lands": {
                    "land_count": 35,
                    "basic_count": 20,
                    "basic_ratio": 0.57,
                    "basic_types": {"Forest": 10, "Mountain": 10},
                }
            },
            "curve": {"mana_curve": {"counts": [2, 3, 5, 7, 4, 3, 1]}},
            "tags": {"tag_counts": {"ramp": 9, "removal": 8}},
        }

    def test_extract_land_count(self):
        assert extract_land_count(self._stats()) == 35

    def test_extract_basic_ratio(self):
        assert pytest.approx(extract_basic_ratio(self._stats()), rel=1e-6) == 0.57

    def test_extract_curve_counts(self):
        assert extract_curve_counts(self._stats()) == [2, 3, 5, 7, 4, 3, 1]

    def test_extract_tag_count_present(self):
        assert extract_tag_count(self._stats(), "ramp") == 9

    def test_extract_tag_count_missing_returns_zero(self):
        assert extract_tag_count(self._stats(), "nonexistent") == 0
