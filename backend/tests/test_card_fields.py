"""Tests for `backend.card_data.card_fields.CardFields` static vocabulary helpers.

This file covers the static lookups, set accessors, basic-land helpers, rarity maps,
predicate factories, and the synergy map. Tagging logic (`tag_text`, `tag_subtypes`,
`tag_card`) and parsing helpers (`parse_mtgjson_card`, etc.) live in
`test_card_fields_tagging.py`.
"""
import numpy as np
import pytest

from backend.card_data import CardFields


# ---------------------------------------------------------------------------
# Vocabulary accessors
# ---------------------------------------------------------------------------


class TestVocabularyAccessors:
    """The list accessors return non-empty, sorted, lowercase tokens."""

    def test_card_types_nonempty_and_sorted(self):
        """`card_types` is a non-empty, sorted, lowercase list."""
        types = CardFields.card_types()
        assert types == sorted(types)
        assert len(types) > 0
        assert all(t == t.lower() for t in types)

    def test_card_types_contains_known_values(self):
        """`card_types` includes the well-known card types."""
        types = CardFields.card_types()
        for known in ("creature", "land", "artifact", "enchantment", "instant",
                      "sorcery", "planeswalker", "battle"):
            assert known in types

    def test_card_supertypes_contains_legendary(self):
        """`card_supertypes` includes legendary and basic."""
        sup = CardFields.card_supertypes()
        assert "legendary" in sup
        assert "basic" in sup

    def test_card_subtypes_contains_common_creature_types(self):
        """`card_subtypes` includes common creature subtypes."""
        subs = CardFields.card_subtypes()
        for t in ("elf", "goblin", "dragon", "human", "wizard"):
            assert t in subs

    def test_color_identities_sorted_and_correct(self):
        """`color_identities` is the six WUBRG+C symbols, sorted."""
        colors = CardFields.color_identities()
        assert colors == sorted(["W", "G", "U", "B", "R", "C"])
        assert len(colors) == 6

    def test_card_tags_sorted_unique(self):
        """`card_tags` is sorted and free of duplicates."""
        tags = CardFields.card_tags()
        assert tags == sorted(tags)
        assert len(tags) == len(set(tags))

    def test_rarities_sorted_and_known(self):
        """`rarities` is sorted and includes the standard rarities."""
        rares = CardFields.rarities()
        assert rares == sorted(rares)
        for known in ("common", "uncommon", "rare", "mythic"):
            assert known in rares


# ---------------------------------------------------------------------------
# Set accessors (must match list contents)
# ---------------------------------------------------------------------------


class TestSetAccessors:
    """Set accessors return Python sets matching their list counterparts."""

    def test_card_types_set_matches_list(self):
        """`card_types_set` matches the `card_types` list."""
        assert CardFields.card_types_set() == set(CardFields.card_types())

    def test_card_supertypes_set_matches_list(self):
        """`card_supertypes_set` matches the `card_supertypes` list."""
        assert CardFields.card_supertypes_set() == set(CardFields.card_supertypes())

    def test_creature_subtypes_set_is_subset_of_subtypes(self):
        """Creature subtypes are a subset of all subtypes."""
        assert CardFields.creature_subtypes_set() <= set(CardFields.card_subtypes())

    def test_color_identities_set_matches(self):
        """`color_identities_set` matches the `color_identities` list."""
        assert CardFields.color_identities_set() == set(CardFields.color_identities())

    def test_tags_general_sets_are_sets(self):
        """Each general tag maps to a set of phrases."""
        for tag, phrases in CardFields.tags_general_sets().items():
            assert isinstance(phrases, set), f"tag {tag} not a set"

    def test_tags_joint_sets_inner_are_sets(self):
        """Each joint tag's pairs are sets."""
        for _, pairs in CardFields.tags_joint_sets().items():
            for pair in pairs:
                assert isinstance(pair, set)

    def test_tags_regex_sets_are_sets(self):
        """Each regex tag maps to a set of patterns."""
        for _, pats in CardFields.tags_regex_sets().items():
            assert isinstance(pats, set)


# ---------------------------------------------------------------------------
# Basic land helpers
# ---------------------------------------------------------------------------


class TestBasicLandHelpers:
    """Recognition and normalization of basic land names."""

    @pytest.mark.parametrize("name", ["Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"])
    def test_is_basic_land_true_for_all_basics(self, name):
        """Every canonical basic land name is recognized."""
        assert CardFields.is_basic_land(name) is True

    @pytest.mark.parametrize("name", ["plains", "ISLAND", "  Swamp  "])
    def test_is_basic_land_case_and_whitespace_insensitive(self, name):
        """Recognition ignores case and surrounding whitespace."""
        assert CardFields.is_basic_land(name) is True

    @pytest.mark.parametrize("name", ["Sol Ring", "Atraxa", "", "Snow-Covered Plains"])
    def test_is_basic_land_false_for_non_basic(self, name):
        """Non-basics (including snow basics) are not recognized."""
        assert CardFields.is_basic_land(name) is False

    def test_basic_lands_returns_lowercase(self):
        """`basic_lands` returns lowercase names."""
        basics = CardFields.basic_lands()
        assert basics == {b.lower() for b in basics}

    def test_color_basic_land_map_complete(self):
        """The color→basic-land map covers all six color symbols."""
        m = CardFields.color_basic_land_map()
        assert m["W"] == "Plains"
        assert m["U"] == "Island"
        assert m["B"] == "Swamp"
        assert m["R"] == "Mountain"
        assert m["G"] == "Forest"
        assert m["C"] == "Wastes"

    def test_basic_type_name_resolves_lowercase(self):
        """`basic_type_name` title-cases a known basic type."""
        assert CardFields.basic_type_name("forest") == "Forest"
        assert CardFields.basic_type_name("mountain") == "Mountain"

    def test_basic_type_name_unknown_returns_wastes(self):
        """`basic_type_name` falls back to Wastes for unknown names."""
        assert CardFields.basic_type_name("Strip Mine") == "Wastes"

    def test_basic_land_type_strips_whitespace_and_case(self):
        """`basic_land_type` normalizes whitespace and case."""
        assert CardFields.basic_land_type("  FOREST  ") == "Forest"


# ---------------------------------------------------------------------------
# Rarity maps
# ---------------------------------------------------------------------------


class TestRarityMaps:
    """`rarity_map` and `rarity_to_index` invert each other and are 1-indexed."""

    def test_rarity_map_is_one_indexed(self):
        """`rarity_map` is keyed from 1, not 0."""
        m = CardFields.rarity_map()
        assert 0 not in m
        assert 1 in m

    def test_rarity_to_index_round_trip(self):
        """`rarity_to_index` inverts `rarity_map`."""
        idx_map = CardFields.rarity_to_index()
        name_map = CardFields.rarity_map()
        for name, idx in idx_map.items():
            assert name_map[idx] == name

    def test_rarity_to_index_count_matches_rarities(self):
        """`rarity_to_index` has one entry per rarity."""
        assert len(CardFields.rarity_to_index()) == len(CardFields.rarities())


# ---------------------------------------------------------------------------
# all_tags + tag_synergy_map
# ---------------------------------------------------------------------------


class TestTagSynergyAndAllTags:
    """`all_tags` covers every tag source; the synergy map is well-formed."""

    def test_all_tags_includes_general_joint_regex_subtype(self):
        """`all_tags` includes tags from every tag source."""
        all_t = CardFields.all_tags()
        for tag in CardFields.general_tags().keys():
            assert tag in all_t
        for tag in CardFields.joint_tags().keys():
            assert tag in all_t
        for tag in CardFields.regex_tags().keys():
            assert tag in all_t
        for tag in CardFields.subtype_tags().keys():
            assert tag in all_t

    def test_tag_synergy_map_values_are_sets_of_strings(self):
        """Each synergy entry is a set of strings excluding the tag itself."""
        synergy = CardFields.tag_synergy_map()
        for tag, neighbors in synergy.items():
            assert isinstance(neighbors, set)
            assert all(isinstance(n, str) for n in neighbors)
            assert tag not in neighbors, f"tag {tag} should not be its own neighbor"


# ---------------------------------------------------------------------------
# Predicate factories
# ---------------------------------------------------------------------------


class TestPredicateFactories:
    """Predicate combinators return callables with expected truth tables."""

    @staticmethod
    def _always_true(_name, _vec):
        """A predicate that is always True."""
        return True

    @staticmethod
    def _always_false(_name, _vec):
        """A predicate that is always False."""
        return False

    def test_pred_dim_matches_matching_shape(self):
        """`pred_dim` is True only when the vector length matches."""
        p = CardFields.pred_dim(5)
        assert p("x", np.zeros(5)) is True
        assert p("x", np.zeros(4)) is False

    def test_pred_regex_case_insensitive_by_default(self):
        """`pred_regex` matches names case-insensitively by default."""
        p = CardFields.pred_regex(r"^atraxa")
        assert p("Atraxa, Praetors' Voice", np.zeros(1)) is True
        assert p("atraxa", np.zeros(1)) is True
        assert p("Magnus the Red", np.zeros(1)) is False

    def test_pred_not_inverts_predicate(self):
        """`pred_not` inverts the wrapped predicate."""
        p = CardFields.pred_not(self._always_true)
        assert p("x", np.zeros(1)) is False

    def test_pred_all_returns_true_iff_every_inner_true(self):
        """`pred_all` is True only when every inner predicate is True."""
        assert CardFields.pred_all(self._always_true, self._always_true)("x", np.zeros(1)) is True
        assert CardFields.pred_all(self._always_true, self._always_false)("x", np.zeros(1)) is False

    def test_pred_any_returns_true_if_any_inner_true(self):
        """`pred_any` is True when any inner predicate is True."""
        assert CardFields.pred_any(self._always_false, self._always_false)("x", np.zeros(1)) is False
        assert CardFields.pred_any(self._always_false, self._always_true)("x", np.zeros(1)) is True

    def test_predicates_compose_correctly(self):
        """Combined predicates respect both name and dimension constraints."""
        is_creature_name = CardFields.pred_regex(r"creature")
        is_8d = CardFields.pred_dim(8)
        combined = CardFields.pred_all(is_creature_name, is_8d)
        assert combined("creature x", np.zeros(8)) is True
        assert combined("creature x", np.zeros(7)) is False
        assert combined("artifact x", np.zeros(8)) is False
