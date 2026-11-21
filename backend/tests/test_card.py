"""Tests for `backend.card_data.card.Card`.

Covers: constructor field assignment, equality semantics, __hash__, __len__,
__str__, get_attributes, and to_json serialization.
"""
import json

import pytest

from backend.card_data import Card

from .conftest import make_card


class TestCardConstructor:
    """Verify the Card constructor stores every field exactly as passed."""

    def test_all_fields_persisted(self):
        """Every constructor argument is stored on the instance unchanged."""
        c = Card(
            commander_legal=True,
            card_name="Magnus the Red",
            card_types=["creature"],
            card_supertypes="legendary",
            card_subtypes=["human", "wizard"],
            mana_cost=5,
            mana_cost_exp="{3}{U}{R}",
            color_identity=["U", "R"],
            defense="",
            rarity="mythic",
            text="When Magnus enters, scry 3.",
            rank="42",
            power="4",
            toughness="4",
            loyalty="",
            card_id="scry-1",
        )
        assert c.commander_legal is True
        assert c.card_name == "Magnus the Red"
        assert c.card_types == ["creature"]
        assert c.card_supertypes == "legendary"
        assert c.card_subtypes == ["human", "wizard"]
        assert c.mana_cost == 5
        assert c.mana_cost_exp == "{3}{U}{R}"
        assert c.color_identity == ["U", "R"]
        assert c.defense == ""
        assert c.rarity == "mythic"
        assert c.text == "When Magnus enters, scry 3."
        assert c.rank == "42"
        assert c.power == "4"
        assert c.toughness == "4"
        assert c.loyalty == ""
        assert c.card_id == "scry-1"

    def test_commander_legal_coerced_to_bool(self):
        """`commander_legal` is coerced to a real bool from truthy/falsy inputs."""
        c = make_card(commander_legal=0)
        assert c.commander_legal is False
        c = make_card(commander_legal=1)
        assert c.commander_legal is True
        c = make_card(commander_legal="anything")  # truthy string
        assert c.commander_legal is True

    def test_empty_subtypes_and_color_identity(self):
        """Empty subtype/color-identity lists are preserved."""
        c = make_card(card_subtypes=[], color_identity=[])
        assert c.card_subtypes == []
        assert c.color_identity == []

    def test_negative_mana_cost_allowed(self):
        """A negative mana cost (the encoder's 'missing' sentinel) is accepted."""
        c = make_card(mana_cost=-1)
        assert c.mana_cost == -1


class TestCardEquality:
    """Equality compares by `card_id` (MultiverseID) — the canonical per-card identity."""

    def test_equal_when_card_id_matches(self):
        """Cards sharing a MultiverseID are equal even with different names."""
        a = make_card(card_name="Forest", card_id="mv-1")
        b = make_card(card_name="Forest // Forest", card_id="mv-1")
        assert a == b

    def test_unequal_when_card_id_differs(self):
        """Cards with different MultiverseIDs are unequal."""
        a = make_card(card_name="Sol Ring", card_id="mv-1")
        b = make_card(card_name="Mox Pearl", card_id="mv-2")
        assert a != b

    def test_unequal_to_non_card_object(self):
        """A Card is unequal to non-Card objects, including None."""
        c = make_card(card_name="X")
        for other in ("X", None, 42, {"card_name": "X"}):
            assert (c == other) is False


class TestCardHash:
    """`__hash__` is bound to `card_id` and stays consistent with card_id-based `__eq__`."""

    def test_hash_matches_card_id_hash(self):
        """A card's hash equals the hash of its `card_id`."""
        c = make_card(card_id="mv-500")
        assert hash(c) == hash("mv-500")

    def test_equal_cards_hash_equal(self):
        """Equal cards (same card_id) hash equally, honoring the eq/hash contract."""
        a = make_card(card_name="Forest", card_id="mv-1")
        b = make_card(card_name="Forest // Forest", card_id="mv-1")
        assert a == b
        assert hash(a) == hash(b)

    def test_hash_stable_across_calls(self):
        """Hashing the same card repeatedly returns the same value."""
        c = make_card(card_id="mv-abc")
        assert hash(c) == hash(c)


class TestCardLen:
    """`__len__` is a constant 1 (one card)."""

    def test_len_is_one(self):
        """A card has length 1."""
        assert len(make_card()) == 1

    def test_len_unchanged_by_attributes(self):
        """Length stays 1 regardless of attribute values."""
        assert len(make_card(card_subtypes=[])) == 1
        assert len(make_card(text="x" * 1000)) == 1


class TestGetAttributes:
    """`get_attributes` mirrors the constructor params; used by serialization."""

    def test_round_trip_dict_keys(self):
        """`get_attributes` returns exactly the constructor field names."""
        c = make_card()
        attrs = c.get_attributes()
        expected_keys = {
            "commander_legal", "card_name", "card_types", "card_supertypes",
            "card_subtypes", "mana_cost", "mana_cost_exp", "color_identity",
            "defense", "rarity", "text", "rank", "power", "toughness",
            "loyalty", "card_id",
        }
        assert set(attrs.keys()) == expected_keys

    def test_values_match_constructor(self):
        """`get_attributes` values reflect the constructor arguments."""
        c = make_card(card_name="X", mana_cost=7, rarity="common")
        attrs = c.get_attributes()
        assert attrs["card_name"] == "X"
        assert attrs["mana_cost"] == 7
        assert attrs["rarity"] == "common"


class TestToJson:
    """`to_json` returns valid, parseable JSON whose content matches attributes."""

    def test_returns_valid_json_string(self):
        """`to_json` parses back into a dict."""
        c = make_card()
        payload = json.loads(c.to_json())
        assert isinstance(payload, dict)

    def test_round_trips_all_fields(self):
        """Selected fields survive the `to_json` round-trip."""
        c = make_card(card_name="Round Trip", mana_cost=4, rarity="rare")
        payload = json.loads(c.to_json())
        assert payload["card_name"] == "Round Trip"
        assert payload["mana_cost"] == 4
        assert payload["rarity"] == "rare"

    def test_handles_empty_lists(self):
        """Empty list fields serialize to empty JSON arrays."""
        c = make_card(card_subtypes=[], color_identity=[])
        payload = json.loads(c.to_json())
        assert payload["card_subtypes"] == []
        assert payload["color_identity"] == []


class TestStr:
    """`__str__` emits `field: value` lines for every attribute."""

    def test_includes_card_name(self):
        """The string form includes the card name and its field label."""
        c = make_card(card_name="Stringify Me")
        s = str(c)
        assert "Stringify Me" in s
        assert "card_name" in s

    def test_one_line_per_field(self):
        """The string form has one line per attribute."""
        c = make_card()
        s = str(c)
        assert len(s.splitlines()) == len(c.get_attributes())


class TestCardInDictAndSet:
    """Hash/equality combine to determine behavior in dict and set membership."""

    def test_cards_with_same_id_collapse_in_set(self):
        """Cards with the same card_id collapse to one entry in a set."""
        a = make_card(card_name="A", card_id="mv-1")
        b = make_card(card_name="A-variant", card_id="mv-1")
        assert len({a, b}) == 1

    def test_cards_with_different_ids_kept_in_set(self):
        """Cards with different card_ids are kept separately in a set."""
        a = make_card(card_name="A", card_id="mv-1")
        b = make_card(card_name="A", card_id="mv-2")
        assert len({a, b}) == 2

    def test_card_in_dict_lookup(self):
        """A card works as a dict key and round-trips to its value."""
        c = make_card(card_name="DictKey", card_id="mv-dk")
        d = {c: "value"}
        assert d[c] == "value"


@pytest.mark.parametrize(
    "mana,expected_type",
    [(0, int), (1, int), (16, int)],
)
def test_mana_cost_type_preserved(mana, expected_type):
    """Integer mana costs keep their `int` type on the constructed card."""
    c = make_card(mana_cost=mana)
    assert isinstance(c.mana_cost, expected_type)
    assert c.mana_cost == mana
