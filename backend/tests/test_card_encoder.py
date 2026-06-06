"""Tests for `backend.card_data.card_encoder.CardEncoder`.

The default constructor downloads a SentenceTransformer model on first use,
which is heavy. All tests here construct the encoder with `embed_model_name=None`
so the text-embedding suffix is skipped and the structural part of the encoding
can be tested cheaply.
"""
import numpy as np
import pytest

from backend.card_data import CardEncoder, CardFields

from .conftest import make_card


# ---------------------------------------------------------------------------
# rarity_to_int
# ---------------------------------------------------------------------------


class TestRarityToInt:
    """`rarity_to_int` mirrors `CardFields.rarity_to_index`."""

    def setup_method(self):
        self.enc = CardEncoder(embed_model_name=None)

    @pytest.mark.parametrize("rarity,expected", list(CardFields.rarity_to_index().items()))
    def test_known_rarities_round_trip(self, rarity, expected):
        assert self.enc.rarity_to_int(rarity) == expected

    def test_unknown_rarity_raises(self):
        with pytest.raises(ValueError):
            self.enc.rarity_to_int("super-mythic-prismatic-foil")


# ---------------------------------------------------------------------------
# encode - basic structural correctness (no embedding)
# ---------------------------------------------------------------------------


class TestEncodeStructure:
    """Verify the layout of the encoded vector (excluding embedding suffix)."""

    def setup_method(self):
        self.enc = CardEncoder(embed_model_name=None)

    @property
    def _expected_len(self):
        # types + supertypes + subtypes + mana(1) + colors + rarity(1)
        return (
            len(self.enc.card_types)
            + len(self.enc.card_supertypes)
            + len(self.enc.all_subtypes)
            + 1
            + len(self.enc.color_identities)
            + 1
        )

    def test_encode_returns_name_and_array(self, sample_card):
        name, vec = self.enc.encode(sample_card)
        assert name == sample_card.card_name
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32

    def test_encoded_length_matches_no_embed(self, sample_card):
        _, vec = self.enc.encode(sample_card)
        assert vec.shape == (self._expected_len,)

    def test_creature_card_sets_creature_bit(self, legendary_creature_card):
        _, vec = self.enc.encode(legendary_creature_card)
        creature_idx = self.enc.card_types.index("creature")
        assert vec[creature_idx] == 1.0

    def test_legendary_supertype_bit(self, legendary_creature_card):
        _, vec = self.enc.encode(legendary_creature_card)
        offset = len(self.enc.card_types)
        legendary_idx = self.enc.card_supertypes.index("legendary")
        assert vec[offset + legendary_idx] == 1.0

    def test_mana_cost_position(self, sample_card):
        _, vec = self.enc.encode(sample_card)
        mana_idx = len(self.enc.card_types) + len(self.enc.card_supertypes) + len(self.enc.all_subtypes)
        assert vec[mana_idx] == float(sample_card.mana_cost)

    def test_rarity_position(self, sample_card):
        _, vec = self.enc.encode(sample_card)
        rarity_idx = self._expected_len - 1
        expected = self.enc.rarity_to_int(sample_card.rarity)
        assert vec[rarity_idx] == float(expected)

    def test_color_identity_bits_set(self):
        card = make_card(color_identity=["W", "B"], mana_cost=2)
        _, vec = self.enc.encode(card)
        offset = (
            len(self.enc.card_types)
            + len(self.enc.card_supertypes)
            + len(self.enc.all_subtypes)
            + 1
        )
        w_idx = self.enc.color_identities.index("W")
        b_idx = self.enc.color_identities.index("B")
        u_idx = self.enc.color_identities.index("U")
        assert vec[offset + w_idx] == 1.0
        assert vec[offset + b_idx] == 1.0
        assert vec[offset + u_idx] == 0.0


class TestEncodeColorlessFallback:
    """When `color_identity` is empty and mana>0, the colorless 'C' bit is set."""

    def setup_method(self):
        self.enc = CardEncoder(embed_model_name=None)

    def test_empty_color_identity_with_mana_sets_colorless(self):
        card = make_card(color_identity=[], mana_cost=1)
        _, vec = self.enc.encode(card)
        offset = (
            len(self.enc.card_types)
            + len(self.enc.card_supertypes)
            + len(self.enc.all_subtypes)
            + 1
        )
        c_idx = self.enc.color_identities.index("C")
        assert vec[offset + c_idx] == 1.0

    def test_empty_color_identity_with_zero_mana_no_color_bits(self, basic_land_card):
        # Basic land: empty color identity AND mana_cost == 0
        _, vec = self.enc.encode(basic_land_card)
        offset = (
            len(self.enc.card_types)
            + len(self.enc.card_supertypes)
            + len(self.enc.all_subtypes)
            + 1
        )
        color_slice = vec[offset : offset + len(self.enc.color_identities)]
        assert all(bit == 0.0 for bit in color_slice)


class TestEncodeSubtypes:
    """Subtype bits flip on for known subtypes."""

    def setup_method(self):
        self.enc = CardEncoder(embed_model_name=None)

    def test_subtype_bits(self):
        card = make_card(card_subtypes=["elf", "warrior"])
        _, vec = self.enc.encode(card)
        offset = len(self.enc.card_types) + len(self.enc.card_supertypes)
        elf_idx = self.enc.all_subtypes.index("elf")
        warrior_idx = self.enc.all_subtypes.index("warrior")
        wizard_idx = self.enc.all_subtypes.index("wizard")
        assert vec[offset + elf_idx] == 1.0
        assert vec[offset + warrior_idx] == 1.0
        assert vec[offset + wizard_idx] == 0.0


class TestEncodeDeterminism:
    """Re-encoding the same card without an embedding model is deterministic."""

    def test_deterministic_no_embed(self):
        enc = CardEncoder(embed_model_name=None)
        card = make_card()
        _, v1 = enc.encode(card)
        _, v2 = enc.encode(card)
        np.testing.assert_array_equal(v1, v2)
