"""Tests for `backend.card_data.deck.Deck` (the full-card deck model).

`SimpleDeck` and `SimpleDeckAnalyzer` have their own test files.
"""
import json

import torch

from backend.card_data import CardEncoder, Deck

from .conftest import make_card


# ---------------------------------------------------------------------------
# Constructor & derived fields
# ---------------------------------------------------------------------------


class TestDeckConstructor:
    """The constructor expands `cards` into `cards_expanded` and `all_cards`."""

    def test_cards_expanded_reflects_quantities(self):
        """`cards_expanded` repeats each card by its quantity."""
        a, b = make_card(card_name="A", card_id="mv-1"), make_card(card_name="B", card_id="mv-2")
        deck = Deck(
            deck_id="id1",
            colors=["W"],
            color_percentages={"W": 1.0},
            bracket=2,
            deck_format="commander",
            commanders=[],
            companions=[],
            mainboard_count=5,
            cards=[(a, 2), (b, 3)],
        )
        assert len(deck.cards_expanded) == 5
        assert deck.cards_expanded.count(a) == 2
        assert deck.cards_expanded.count(b) == 3

    def test_all_cards_concatenates_commanders_companions_expanded(self):
        """`all_cards` is commanders + companions + expanded mainboard, in order."""
        cmd = make_card(card_name="Cmd")
        comp = make_card(card_name="Comp")
        a = make_card(card_name="A")
        deck = Deck(
            deck_id=None,
            colors=[],
            color_percentages={},
            bracket=1,
            deck_format="commander",
            commanders=[cmd],
            companions=[comp],
            mainboard_count=1,
            cards=[(a, 1)],
        )
        assert deck.all_cards == [cmd, comp, a]


# ---------------------------------------------------------------------------
# Equality, str, len
# ---------------------------------------------------------------------------


class TestDeckDunders:
    """Equality compares on `cards` list; len sums quantities + cmd + companion."""

    def _empty_deck(self, cards=None, commanders=None, companions=None):
        """Build a minimal `Deck` with optional cards/commanders/companions."""
        return Deck(
            deck_id=None,
            colors=[],
            color_percentages={},
            bracket=0,
            deck_format=None,
            commanders=commanders or [],
            companions=companions or [],
            mainboard_count=0,
            cards=cards or [],
        )

    def test_eq_on_same_cards(self):
        """Two decks with identical cards compare equal."""
        a = make_card(card_name="A")
        d1 = self._empty_deck(cards=[(a, 1)])
        d2 = self._empty_deck(cards=[(a, 1)])
        assert d1 == d2

    def test_eq_false_against_non_deck(self):
        """A deck is not equal to a non-deck object."""
        d = self._empty_deck()
        assert (d == "deck") is False

    def test_neq_when_cards_differ(self):
        """Decks with different cards are unequal."""
        a = make_card(card_name="A", card_id="mv-1")
        b = make_card(card_name="B", card_id="mv-2")
        d1 = self._empty_deck(cards=[(a, 1)])
        d2 = self._empty_deck(cards=[(b, 1)])
        assert d1 != d2

    def test_len_sums_qty_plus_commanders_plus_companions(self):
        """`len` counts mainboard quantities plus commanders and companions."""
        a = make_card(card_name="A")
        cmd = make_card(card_name="Cmd")
        comp = make_card(card_name="Comp")
        d = self._empty_deck(cards=[(a, 4)], commanders=[cmd], companions=[comp])
        assert len(d) == 4 + 1 + 1

    def test_str_contains_id_and_colors(self):
        """The string form includes the deck id and its colors."""
        d = self._empty_deck()
        d.id = "deck-xyz"
        d.colors = ["U", "B"]
        s = str(d)
        assert "deck-xyz" in s
        assert "U" in s and "B" in s


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestDeckSerialization:
    """`get_attributes` + `to_json` produce JSON-serializable, complete payloads."""

    def test_to_json_parses_back(self):
        """`to_json` round-trips through `json.loads` with expected fields."""
        a = make_card(card_name="A")
        cmd = make_card(card_name="Cmd")
        deck = Deck(
            deck_id="d1",
            colors=["W"],
            color_percentages={"W": 1.0},
            bracket=3,
            deck_format="commander",
            commanders=[cmd],
            companions=[],
            mainboard_count=1,
            cards=[(a, 1)],
        )
        payload = json.loads(deck.to_json())
        assert payload["id"] == "d1"
        assert payload["bracket"] == 3
        assert payload["cards"][0]["quantity"] == 1
        assert payload["commanders"][0]["card_name"] == "Cmd"

    def test_get_attributes_keys(self):
        """`get_attributes` exposes all documented top-level keys."""
        deck = Deck(
            deck_id=None, colors=[], color_percentages={}, bracket=0,
            deck_format=None, commanders=[], companions=[], mainboard_count=0, cards=[],
        )
        attrs = deck.get_attributes()
        for k in ("id", "colors", "color_percentages", "bracket", "format",
                  "commanders", "companions", "mainboard_count", "cards"):
            assert k in attrs


# ---------------------------------------------------------------------------
# to_tensor
# ---------------------------------------------------------------------------


class TestDeckToTensor:
    """`to_tensor` encodes every card in `all_cards` into a stacked tensor."""

    def test_stack_shape(self):
        """The stacked tensor has one row per card in `all_cards`."""
        enc = CardEncoder(embed_model_name=None)
        a = make_card(card_name="A")
        b = make_card(card_name="B")
        deck = Deck(
            deck_id=None, colors=[], color_percentages={}, bracket=0,
            deck_format=None, commanders=[], companions=[],
            mainboard_count=2, cards=[(a, 1), (b, 1)],
        )
        t = deck.to_tensor(enc)
        assert t.dim() == 2
        assert t.size(0) == 2

    def test_returns_float_tensor(self):
        """`to_tensor` returns a float32 tensor."""
        enc = CardEncoder(embed_model_name=None)
        a = make_card(card_name="A")
        deck = Deck(
            deck_id=None, colors=[], color_percentages={}, bracket=0,
            deck_format=None, commanders=[], companions=[],
            mainboard_count=1, cards=[(a, 1)],
        )
        t = deck.to_tensor(enc)
        assert t.dtype == torch.float32


# ---------------------------------------------------------------------------
# basic_lands_from_colors
# ---------------------------------------------------------------------------


class TestBasicLandsFromColors:
    """Mapping color symbols to recommended basic lands."""

    def test_all_five_colors(self):
        """All five WUBRG colors map to their basic land names in order."""
        out = Deck.basic_lands_from_colors(["W", "U", "B", "R", "G"])
        assert out == ["Plains", "Island", "Swamp", "Mountain", "Forest"]

    def test_ignores_unknown_colors(self):
        """Unknown color symbols yield no basic lands."""
        assert Deck.basic_lands_from_colors(["X", "Y"]) == []

    def test_colorless_maps_to_wastes(self):
        """The colorless symbol maps to Wastes."""
        assert Deck.basic_lands_from_colors(["C"]) == ["Wastes"]

    def test_empty_input(self):
        """Empty input yields an empty list."""
        assert Deck.basic_lands_from_colors([]) == []


# ---------------------------------------------------------------------------
# shape_deck
# ---------------------------------------------------------------------------


class TestDeckShape:
    """`shape_deck` trims to 99 entries; does not pad on full Deck (handled in SimpleDeck)."""

    def test_trims_to_99_entries(self):
        """An oversized deck is trimmed down to 99 card entries."""
        cards = [(make_card(card_name=f"C{i}"), 1) for i in range(150)]
        deck = Deck(
            deck_id=None, colors=[], color_percentages={}, bracket=0,
            deck_format=None, commanders=[], companions=[],
            mainboard_count=150, cards=cards,
        )
        deck.shape_deck(_commander_colors=[])
        assert len(deck.cards) == 99

    def test_no_op_when_under_99(self):
        """A deck under 99 cards is left unchanged (full Deck does not pad)."""
        cards = [(make_card(card_name=f"C{i}"), 1) for i in range(5)]
        deck = Deck(
            deck_id=None, colors=[], color_percentages={}, bracket=0,
            deck_format=None, commanders=[], companions=[],
            mainboard_count=5, cards=cards,
        )
        deck.shape_deck(_commander_colors=["W"])
        assert len(deck.cards) == 5
