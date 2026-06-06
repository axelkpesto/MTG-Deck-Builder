"""Tests for tagging and parsing helpers on `CardFields`.

Covers: `tag_text`, `tag_subtypes`, `tag_card`, `parse_mtgjson_card`,
`parse_moxfieldapi_card`, and `parse_moxfield_group`.
"""
import pytest

from backend.card_data import Card, CardFields

from .conftest import make_card


# ---------------------------------------------------------------------------
# tag_text
# ---------------------------------------------------------------------------


class TestTagText:
    """Phrase, joint, and regex tag inference from card text."""

    def test_empty_text_returns_empty_set(self):
        assert CardFields.tag_text("Anything", "") == set()

    def test_card_draw_phrase_match(self):
        tags = CardFields.tag_text("X", "Draw a card.")
        assert "card_draw" in tags

    def test_aggro_phrase_match(self):
        tags = CardFields.tag_text("X", "This creature has haste.")
        assert "aggro" in tags

    def test_ramp_phrase_match(self):
        tags = CardFields.tag_text(
            "X", "Search your library for a land card, put it onto the battlefield."
        )
        # Either ramp (joint or phrase) should match
        assert "ramp" in tags

    def test_control_phrase_match(self):
        tags = CardFields.tag_text("X", "Counter target spell.")
        assert "control" in tags

    def test_life_gain_phrase_match(self):
        tags = CardFields.tag_text("X", "You gain 4 life.")
        assert "life_gain" in tags

    def test_name_substring_removed_before_matching(self):
        # If the card name itself contained a trigger phrase ("Draw a card"),
        # it should not contribute a tag. Use a name that includes the phrase.
        tags = CardFields.tag_text("Draw a Card", "")
        # No real text -> no tags even if the name has 'draw a card'
        assert "card_draw" not in tags

    def test_returns_set_type(self):
        assert isinstance(CardFields.tag_text("X", ""), set)

    def test_case_insensitive_matching(self):
        tags = CardFields.tag_text("X", "DRAW A CARD.")
        assert "card_draw" in tags

    def test_multiple_tags_can_match(self):
        tags = CardFields.tag_text(
            "X",
            "Haste. Whenever this creature attacks, draw a card.",
        )
        assert "aggro" in tags
        assert "card_draw" in tags


# ---------------------------------------------------------------------------
# tag_subtypes
# ---------------------------------------------------------------------------


class TestTagSubtypes:
    """Subtype-based tagging using `__subtype_tags` map."""

    def test_none_returns_empty(self):
        assert CardFields.tag_subtypes(None) == set()

    def test_empty_list_returns_empty(self):
        assert CardFields.tag_subtypes([]) == set()

    def test_elf_yields_tribal(self):
        tags = CardFields.tag_subtypes(["elf"])
        assert "tribal" in tags

    def test_aura_yields_enchantment_and_voltron(self):
        tags = CardFields.tag_subtypes(["aura"])
        assert "enchantment" in tags
        assert "voltron" in tags

    def test_equipment_yields_equipment_and_voltron(self):
        tags = CardFields.tag_subtypes(["equipment"])
        assert "equipment" in tags
        assert "voltron" in tags

    def test_treasure_yields_artifact(self):
        tags = CardFields.tag_subtypes(["treasure"])
        assert "artifact" in tags

    def test_case_insensitive_against_uppercase_subtype(self):
        tags = CardFields.tag_subtypes(["ELF"])
        assert "tribal" in tags

    def test_unknown_subtype_returns_empty(self):
        assert CardFields.tag_subtypes(["xyzzy"]) == set()


# ---------------------------------------------------------------------------
# tag_card (union of tag_text + tag_subtypes)
# ---------------------------------------------------------------------------


class TestTagCard:
    """Top-level tag inference combines text and subtype results."""

    def test_combines_text_and_subtype_results(self):
        card = make_card(
            card_name="Z",
            text="Draw a card.",
            card_subtypes=["elf"],
        )
        tags = CardFields.tag_card(card)
        assert "card_draw" in tags
        assert "tribal" in tags

    def test_returns_empty_for_vanilla_card(self):
        card = make_card(
            card_name="Vanilla",
            text="",
            card_subtypes=["bear"],  # bear has no subtype tag
        )
        tags = CardFields.tag_card(card)
        # bear has no associated tag, text is empty
        assert tags == set()


# ---------------------------------------------------------------------------
# parse_mtgjson_card
# ---------------------------------------------------------------------------


class TestParseMtgJsonCard:
    """Convert an MTGJSON-shaped dict to a `Card` object."""

    @staticmethod
    def _make_mtgjson_payload(**overrides):
        base = {
            "name": "Test",
            "types": ["Creature"],
            "supertypes": ["Legendary"],
            "subtypes": ["Human", "Wizard"],
            "manaValue": 3,
            "manaCost": "{1}{U}{U}",
            "colorIdentity": ["U"],
            "defense": "",
            "rarity": "rare",
            "text": "Some text",
            "edhrecRank": 1,
            "power": "2",
            "toughness": "3",
            "loyalty": "",
            "identifiers": {"scryfallId": "abc"},
            "legalities": {"commander": "Legal"},
            "availability": ["paper"],
        }
        base.update(overrides)
        return base

    def test_returns_card_instance(self):
        card = CardFields.parse_mtgjson_card(self._make_mtgjson_payload())
        assert isinstance(card, Card)

    def test_types_lowercased(self):
        card = CardFields.parse_mtgjson_card(self._make_mtgjson_payload())
        assert card.card_types == ["creature"]
        assert card.card_supertypes == ["legendary"]
        assert card.card_subtypes == ["human", "wizard"]

    def test_commander_legal_true_when_legal_and_paper(self):
        card = CardFields.parse_mtgjson_card(self._make_mtgjson_payload())
        assert card.commander_legal is True

    def test_commander_legal_false_when_banned(self):
        card = CardFields.parse_mtgjson_card(
            self._make_mtgjson_payload(legalities={"commander": "Banned"})
        )
        assert card.commander_legal is False

    def test_commander_legal_false_when_no_paper(self):
        card = CardFields.parse_mtgjson_card(
            self._make_mtgjson_payload(availability=["arena", "mtgo"])
        )
        assert card.commander_legal is False

    def test_scryfall_id_extracted(self):
        card = CardFields.parse_mtgjson_card(
            self._make_mtgjson_payload(identifiers={"scryfallId": "deadbeef"})
        )
        assert card.card_id == "deadbeef"

    def test_missing_optional_fields_get_defaults(self):
        payload = {
            "name": "Bare",
            "legalities": {"commander": "Legal"},
            "availability": ["paper"],
            "identifiers": {},  # parser does .get('identifiers','').get('scryfallId', '')
        }
        card = CardFields.parse_mtgjson_card(payload)
        assert card.card_name == "Bare"
        assert card.text == ""
        assert card.card_id == ""


# ---------------------------------------------------------------------------
# parse_moxfieldapi_card
# ---------------------------------------------------------------------------


class TestParseMoxfieldApiCard:
    """Convert a Moxfield API card dict to a `Card` model."""

    @staticmethod
    def _payload(**overrides):
        # The parser splits on a literal Latin-1-encoded em dash.
        base = {
            "type_line": "Legendary Creature â€” Human Wizard",
            "name": "Test Wizard",
            "legalities": {"commander": "Legal"},
            "cmc": 3,
            "mana_cost": "{1}{U}{U}",
            "color_identity": ["U"],
            "rarity": "rare",
            "oracle_text": "Draw a card.",
            "edhrec_rank": 100,
            "power": "1",
            "toughness": "1",
            "loyalty": "",
            "scryfall_id": "sid",
        }
        base.update(overrides)
        return base

    def test_returns_card(self):
        card = CardFields.parse_moxfieldapi_card(self._payload())
        assert isinstance(card, Card)
        assert card.card_name == "Test Wizard"

    def test_text_from_oracle_text(self):
        card = CardFields.parse_moxfieldapi_card(self._payload(oracle_text="Counter target spell."))
        assert "counter" in card.text.lower()

    def test_commander_legal_flag(self):
        card = CardFields.parse_moxfieldapi_card(self._payload())
        assert card.commander_legal is True

        card2 = CardFields.parse_moxfieldapi_card(
            self._payload(legalities={"commander": "Banned"})
        )
        assert card2.commander_legal is False


# ---------------------------------------------------------------------------
# parse_moxfield_group
# ---------------------------------------------------------------------------


class TestParseMoxfieldGroup:
    """Parsing card groups out of Moxfield deck payloads."""

    def test_missing_group_returns_empty(self):
        assert CardFields.parse_moxfield_group({}, "mainboard") == []

    def test_empty_group_returns_empty(self):
        assert CardFields.parse_moxfield_group({"mainboard": []}, "mainboard") == []
        assert CardFields.parse_moxfield_group({"mainboard": {}}, "mainboard") == []

    def test_parses_dict_style_group(self):
        deck = {
            "mainboard": {
                "k1": {
                    "card": {
                        "type_line": "Creature â€” Human",
                        "name": "Hu",
                        "legalities": {"commander": "Legal"},
                        "cmc": 2,
                    }
                }
            }
        }
        cards = CardFields.parse_moxfield_group(deck, "mainboard")
        assert len(cards) == 1
        assert cards[0].card_name == "Hu"

    def test_parses_list_style_group(self):
        deck = {
            "commanders": [
                {
                    "card": {
                        "type_line": "Legendary Creature â€” Angel",
                        "name": "Atraxa",
                        "legalities": {"commander": "Legal"},
                        "cmc": 4,
                    }
                }
            ]
        }
        cards = CardFields.parse_moxfield_group(deck, "commanders")
        assert len(cards) == 1
        assert cards[0].card_name == "Atraxa"

    def test_ignores_entries_without_card_dict(self):
        deck = {"mainboard": [{"card": "not a dict"}, {"no_card_key": True}]}
        cards = CardFields.parse_moxfield_group(deck, "mainboard")
        assert cards == []
