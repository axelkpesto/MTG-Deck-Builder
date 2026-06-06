"""Tests for `backend.card_data.deck.SimpleDeck`."""
import json
import tempfile
from pathlib import Path

import pytest
import torch

from backend.card_data import SimpleDeck


# ---------------------------------------------------------------------------
# Constructor & defaults
# ---------------------------------------------------------------------------


class TestSimpleDeckConstructor:
    """Default-arg behavior of the constructor."""

    def test_defaults_empty(self):
        d = SimpleDeck()
        assert d.id is None
        assert d.commanders == []
        assert d.cards == []

    def test_constructor_preserves_explicit_args(self):
        d = SimpleDeck(deck_id="d1", commanders=["Cmd"], cards=["A", "B"])
        assert d.id == "d1"
        assert d.commanders == ["Cmd"]
        assert d.cards == ["A", "B"]


# ---------------------------------------------------------------------------
# Equality, len, str
# ---------------------------------------------------------------------------


class TestSimpleDeckDunders:
    """Equality compares on both `cards` and `commanders`."""

    def test_eq_same_content(self):
        a = SimpleDeck(commanders=["X"], cards=["A", "B"])
        b = SimpleDeck(deck_id="other", commanders=["X"], cards=["A", "B"])
        assert a == b

    def test_neq_different_commanders(self):
        a = SimpleDeck(commanders=["X"], cards=["A"])
        b = SimpleDeck(commanders=["Y"], cards=["A"])
        assert a != b

    def test_neq_against_non_simple_deck(self):
        assert (SimpleDeck() == "hello") is False

    def test_len_counts_cards_and_commanders(self):
        d = SimpleDeck(commanders=["X"], cards=["A", "B", "C"])
        assert len(d) == 4

    def test_str_contains_id(self):
        d = SimpleDeck(deck_id="abc", commanders=["X"], cards=["A"])
        s = str(d)
        assert "abc" in s


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestSimpleDeckSerialization:
    """Round-trip via `to_json`, `from_json`, and `load_json_file`."""

    def test_to_json_parses_back(self):
        d = SimpleDeck(deck_id="i1", commanders=["C"], cards=["A", "B"])
        payload = json.loads(d.to_json())
        assert payload["id"] == "i1"
        assert payload["commanders"] == ["C"]
        assert payload["cards"] == ["A", "B"]

    def test_from_json_builds_object(self):
        d = SimpleDeck.from_json({"id": "i2", "commanders": ["C"], "cards": ["A"]})
        assert d.id == "i2"
        assert d.commanders == ["C"]
        assert d.cards == ["A"]

    def test_from_json_cards_default_empty(self):
        d = SimpleDeck.from_json({"id": "i3", "commanders": ["C"]})
        assert d.cards == []

    def test_load_json_file_list_of_decks(self, tmp_path):
        p = tmp_path / "decks.json"
        p.write_text(json.dumps([
            {"id": "a", "commanders": ["X"], "cards": ["1"]},
            {"id": "b", "commanders": ["Y"], "cards": ["2"]},
        ]))
        out = SimpleDeck.load_json_file(str(p))
        assert len(out) == 2
        assert out[0].id == "a"
        assert out[1].id == "b"

    def test_load_json_file_dict_payload(self, tmp_path):
        p = tmp_path / "decks.json"
        p.write_text(json.dumps({
            "k1": {"id": "a", "commanders": ["X"], "cards": ["1"]},
            "k2": {"id": "b", "commanders": ["Y"], "cards": ["2"]},
        }))
        out = SimpleDeck.load_json_file(str(p))
        assert len(out) == 2
        ids = {d.id for d in out}
        assert ids == {"a", "b"}


# ---------------------------------------------------------------------------
# shape_deck (padding logic)
# ---------------------------------------------------------------------------


class TestSimpleDeckShape:
    """`shape_deck` trims to 99 or pads with basic lands."""

    def test_trims_to_99(self):
        d = SimpleDeck(cards=[f"C{i}" for i in range(200)])
        d.shape_deck(["W"])
        assert len(d.cards) == 99

    def test_pads_with_basics_from_colors(self):
        d = SimpleDeck(cards=["X"])
        d.shape_deck(["W", "G"])
        assert len(d.cards) == 99
        # Padding should be drawn from Plains + Forest.
        suffix = d.cards[1:]
        assert all(c in ("Plains", "Forest") for c in suffix)

    def test_padding_with_empty_colors_falls_back_to_wastes(self):
        # Empty commander_colors → no color-mapped basics; pad with Wastes.
        d = SimpleDeck(cards=["X"])
        d.shape_deck([])
        assert len(d.cards) == 99
        assert d.cards[0] == "X"
        assert all(c == "Wastes" for c in d.cards[1:])

    def test_pads_uses_all_basics_when_99_under(self):
        d = SimpleDeck(cards=[])
        d.shape_deck(["U"])
        assert d.cards == ["Island"] * 99


# ---------------------------------------------------------------------------
# basic_lands_from_colors
# ---------------------------------------------------------------------------


class TestSimpleDeckBasicLands:
    """Mirror of Deck.basic_lands_from_colors but on SimpleDeck."""

    def test_all_colors(self):
        out = SimpleDeck.basic_lands_from_colors(["W", "U", "B", "R", "G", "C"])
        assert out == ["Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"]

    def test_unknown_colors_skipped(self):
        assert SimpleDeck.basic_lands_from_colors(["?"]) == []


# ---------------------------------------------------------------------------
# to_tensor_stack
# ---------------------------------------------------------------------------


class TestSimpleDeckToTensorStack:
    """`to_tensor_stack` looks up vectors in a VectorDatabase via find_*."""

    def test_stacks_present_cards(self, populated_vector_database):
        # populated_vector_database fixture contains Atraxa, Sol Ring, etc.
        d = SimpleDeck(commanders=["Atraxa"], cards=["Sol Ring", "Lightning Bolt"])
        t = d.to_tensor_stack(populated_vector_database)
        assert t.dim() == 2
        assert t.size(0) == 3
        assert t.dtype == torch.float32

    def test_skips_missing_cards(self, populated_vector_database):
        d = SimpleDeck(commanders=["Atraxa"], cards=["Sol Ring", "Nonexistent Card"])
        t = d.to_tensor_stack(populated_vector_database)
        assert t.size(0) == 2  # missing card silently dropped

    def test_empty_when_all_missing(self, populated_vector_database):
        d = SimpleDeck(commanders=[], cards=["Not Here", "Also Missing"])
        # All missing ⇒ torch.stack on empty list raises; this documents the
        # current behavior (the function expects at least one match).
        with pytest.raises((RuntimeError, ValueError)):
            d.to_tensor_stack(populated_vector_database)
