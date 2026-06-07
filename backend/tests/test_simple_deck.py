"""Tests for `backend.card_data.deck.SimpleDeck`."""
import json

import pytest
import torch

from backend.card_data import SimpleDeck


# ---------------------------------------------------------------------------
# Constructor & defaults
# ---------------------------------------------------------------------------


class TestSimpleDeckConstructor:
    """Default-arg behavior of the constructor."""

    def test_defaults_empty(self):
        """A default `SimpleDeck` has no id, commanders, or cards."""
        d = SimpleDeck()
        assert d.id is None
        assert d.commanders == []
        assert d.cards == []

    def test_constructor_preserves_explicit_args(self):
        """Explicit constructor arguments are stored verbatim."""
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
        """Decks with the same cards/commanders are equal regardless of id."""
        a = SimpleDeck(commanders=["X"], cards=["A", "B"])
        b = SimpleDeck(deck_id="other", commanders=["X"], cards=["A", "B"])
        assert a == b

    def test_neq_different_commanders(self):
        """Decks differing only in commanders are unequal."""
        a = SimpleDeck(commanders=["X"], cards=["A"])
        b = SimpleDeck(commanders=["Y"], cards=["A"])
        assert a != b

    def test_neq_against_non_simple_deck(self):
        """A deck is not equal to a non-deck object."""
        assert (SimpleDeck() == "hello") is False

    def test_len_counts_cards_and_commanders(self):
        """`len` counts both cards and commanders."""
        d = SimpleDeck(commanders=["X"], cards=["A", "B", "C"])
        assert len(d) == 4

    def test_str_contains_id(self):
        """The string form includes the deck id."""
        d = SimpleDeck(deck_id="abc", commanders=["X"], cards=["A"])
        s = str(d)
        assert "abc" in s


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestSimpleDeckSerialization:
    """Round-trip via `to_json`, `from_json`, and `load_json_file`."""

    def test_to_json_parses_back(self):
        """`to_json` produces a payload with id, commanders, and cards."""
        d = SimpleDeck(deck_id="i1", commanders=["C"], cards=["A", "B"])
        payload = json.loads(d.to_json())
        assert payload["id"] == "i1"
        assert payload["commanders"] == ["C"]
        assert payload["cards"] == ["A", "B"]

    def test_from_json_builds_object(self):
        """`from_json` reconstructs a deck from a dict payload."""
        d = SimpleDeck.from_json({"id": "i2", "commanders": ["C"], "cards": ["A"]})
        assert d.id == "i2"
        assert d.commanders == ["C"]
        assert d.cards == ["A"]

    def test_from_json_cards_default_empty(self):
        """`from_json` defaults `cards` to an empty list when absent."""
        d = SimpleDeck.from_json({"id": "i3", "commanders": ["C"]})
        assert d.cards == []

    def test_load_json_file_list_of_decks(self, tmp_path):
        """`load_json_file` parses a top-level list of deck payloads."""
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
        """`load_json_file` parses a dict-of-decks payload."""
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
        """An oversized deck is trimmed to 99 cards."""
        d = SimpleDeck(cards=[f"C{i}" for i in range(200)])
        d.shape_deck(["W"])
        assert len(d.cards) == 99

    def test_pads_with_basics_from_colors(self):
        """Padding draws basic lands from the commander's colors."""
        d = SimpleDeck(cards=["X"])
        d.shape_deck(["W", "G"])
        assert len(d.cards) == 99
        suffix = d.cards[1:]
        assert all(c in ("Plains", "Forest") for c in suffix)

    def test_padding_with_empty_colors_falls_back_to_wastes(self):
        """With no colors, padding falls back to Wastes."""
        d = SimpleDeck(cards=["X"])
        d.shape_deck([])
        assert len(d.cards) == 99
        assert d.cards[0] == "X"
        assert all(c == "Wastes" for c in d.cards[1:])

    def test_pads_uses_all_basics_when_99_under(self):
        """An empty deck pads entirely with the single color's basic land."""
        d = SimpleDeck(cards=[])
        d.shape_deck(["U"])
        assert d.cards == ["Island"] * 99


# ---------------------------------------------------------------------------
# basic_lands_from_colors
# ---------------------------------------------------------------------------


class TestSimpleDeckBasicLands:
    """Mirror of Deck.basic_lands_from_colors but on SimpleDeck."""

    def test_all_colors(self):
        """Every color symbol maps to its basic land name."""
        out = SimpleDeck.basic_lands_from_colors(["W", "U", "B", "R", "G", "C"])
        assert out == ["Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"]

    def test_unknown_colors_skipped(self):
        """Unknown color symbols are skipped."""
        assert SimpleDeck.basic_lands_from_colors(["?"]) == []


# ---------------------------------------------------------------------------
# to_tensor_stack
# ---------------------------------------------------------------------------


class TestSimpleDeckToTensorStack:
    """`to_tensor_stack` looks up vectors in a VectorDatabase via find_*."""

    def test_stacks_present_cards(self, populated_vector_database):
        """Present cards (commander + mainboard) stack into a float32 tensor."""
        d = SimpleDeck(commanders=["Atraxa"], cards=["Sol Ring", "Lightning Bolt"])
        t = d.to_tensor_stack(populated_vector_database)
        assert t.dim() == 2
        assert t.size(0) == 3
        assert t.dtype == torch.float32

    def test_skips_missing_cards(self, populated_vector_database):
        """Cards absent from the database are silently dropped."""
        d = SimpleDeck(commanders=["Atraxa"], cards=["Sol Ring", "Nonexistent Card"])
        t = d.to_tensor_stack(populated_vector_database)
        assert t.size(0) == 2

    def test_empty_when_all_missing(self, populated_vector_database):
        """All-missing cards raise, documenting that at least one match is required."""
        d = SimpleDeck(commanders=[], cards=["Not Here", "Also Missing"])
        with pytest.raises((RuntimeError, ValueError)):
            d.to_tensor_stack(populated_vector_database)
