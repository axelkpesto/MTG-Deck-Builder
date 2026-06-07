"""Tests for `backend.card_data.card_decoder.CardDecoder`.

Covers: `decode`, `decode_to_dict`, title casing (via the public Name output),
`int_to_rarity`, `slice`, `item_from_vector`, `constrain_logits`,
`land_mask_from_vectors`, `color_identity_mask_from_vectors`, and
`mana_value_from_vectors`.
"""
import numpy as np
import pytest
import torch

from backend.card_data import CardDecoder, CardFields

from .conftest import make_encoded_vector


# ---------------------------------------------------------------------------
# Title casing (exercised through the public decode_to_dict Name field)
# ---------------------------------------------------------------------------


class TestTitleCase:
    """Card-name title casing keeps transition words lowercase except at start."""

    decoder: CardDecoder

    def setup_method(self):
        """Construct a decoder for each test."""
        self.decoder = CardDecoder()

    def _name(self, raw: str) -> str:
        """Return the title-cased Name produced by decoding `raw`."""
        return self.decoder.decode_to_dict(raw, make_encoded_vector())["Name"]

    def test_simple_title_case(self):
        """Each word of a plain name is capitalized."""
        assert self._name("sol ring") == "Sol Ring"

    def test_transitions_lowercased_mid_string(self):
        """Transition words stay lowercase when not at the start."""
        assert self._name("city of brass") == "City of Brass"
        assert self._name("seat of the synod") == "Seat of the Synod"

    def test_leading_transition_word_capitalized(self):
        """A leading transition word is still capitalized."""
        assert self._name("of one mind") == "Of One Mind"

    def test_single_word(self):
        """A single-word name is capitalized."""
        assert self._name("forest") == "Forest"

    def test_apostrophes_and_punctuation_preserved(self):
        """Names with punctuation title-case without error."""
        assert "Atraxa" in self._name("atraxa, praetors' voice")


# ---------------------------------------------------------------------------
# int_to_rarity
# ---------------------------------------------------------------------------


class TestIntToRarity:
    """`int_to_rarity` mirrors `CardFields.rarity_map`."""

    decoder: CardDecoder

    def setup_method(self):
        """Construct a decoder for each test."""
        self.decoder = CardDecoder()

    @pytest.mark.parametrize("idx,name", list(CardFields.rarity_map().items()))
    def test_known_indices(self, idx, name):
        """Each known rarity index maps to its name."""
        assert self.decoder.int_to_rarity(idx) == name

    def test_unknown_returns_unknown(self):
        """An out-of-range index returns 'Unknown'."""
        assert self.decoder.int_to_rarity(99999) == "Unknown"

    def test_zero_returns_unknown(self):
        """Index 0 is unknown because the rarity map is 1-indexed."""
        assert self.decoder.int_to_rarity(0) == "Unknown"


# ---------------------------------------------------------------------------
# decode / decode_to_dict
# ---------------------------------------------------------------------------


class TestDecodeToDict:
    """Round-trip decoding the structured prefix of an encoded card vector."""

    decoder: CardDecoder

    def setup_method(self):
        """Construct a decoder for each test."""
        self.decoder = CardDecoder()

    def test_basic_creature_decode(self):
        """A creature vector decodes into the expected structured fields."""
        vec = make_encoded_vector(
            card_types=["creature"],
            supertypes=["legendary"],
            subtypes=["human", "wizard"],
            mana=3,
            color_identity=["U"],
            rarity_index=CardFields.rarity_to_index()["rare"],
        )
        out = self.decoder.decode_to_dict("magnus the red", vec)
        assert out["Name"] == "Magnus the Red"
        assert "Creature" in out["Types"]
        assert "Legendary" in out["Supertypes"]
        assert "Human" in out["Subtypes"]
        assert "Wizard" in out["Subtypes"]
        assert out["Mana Cost"] == "3"
        assert "U" in out["Color Identity"]
        assert out["Rarity"] == "Rare"

    def test_basic_land_decode(self):
        """A basic-land vector decodes with zero mana and empty color identity."""
        vec = make_encoded_vector(
            card_types=["land"],
            supertypes=["basic"],
            subtypes=["forest"],
            mana=0,
            color_identity=[],
            rarity_index=CardFields.rarity_to_index()["common"],
        )
        out = self.decoder.decode_to_dict("forest", vec)
        assert out["Name"] == "Forest"
        assert out["Mana Cost"] == "0"
        assert out["Color Identity"] == "[]"


class TestDecodeString:
    """`decode` formats the dict as newline-delimited 'k: v' lines."""

    def test_decode_returns_string_with_all_keys(self):
        """The decoded string contains every field label."""
        decoder = CardDecoder()
        vec = make_encoded_vector(
            card_types=["creature"], mana=2,
            rarity_index=CardFields.rarity_to_index()["common"],
        )
        text = decoder.decode("x", vec)
        for k in ("Name", "Types", "Supertypes", "Subtypes",
                  "Mana Cost", "Color Identity", "Rarity"):
            assert k in text

    def test_one_line_per_field(self):
        """The decoded string has one line per field (seven total)."""
        decoder = CardDecoder()
        vec = make_encoded_vector(
            rarity_index=CardFields.rarity_to_index()["common"],
        )
        lines = decoder.decode("x", vec).splitlines()
        assert len(lines) == 7


# ---------------------------------------------------------------------------
# slice
# ---------------------------------------------------------------------------


class TestSlice:
    """`slice()` returns valid python slices for each feature group."""

    decoder: CardDecoder

    def setup_method(self):
        """Construct a decoder for each test."""
        self.decoder = CardDecoder()

    def test_slice_for_types(self):
        """The 'types' slice spans the start of the type vocabulary."""
        s = self.decoder.slice("types", 1000)
        assert isinstance(s, slice)
        assert s.start == 0
        assert s.stop == len(CardFields.card_types())

    def test_slice_for_embed_uses_tail_of_vector(self):
        """The 'embed' slice is the embedding-sized tail of the vector."""
        dim = 1000
        s = self.decoder.slice("embed", dim)
        assert s.stop == dim
        assert s.start == dim - self.decoder.embed_dim

    def test_slice_invalid_key_raises(self):
        """An unknown slice key raises `KeyError`."""
        with pytest.raises(KeyError):
            self.decoder.slice("nonexistent_field", 1000)


# ---------------------------------------------------------------------------
# item_from_vector
# ---------------------------------------------------------------------------


class TestItemFromVector:
    """Categorical decoding from a slice of a vector."""

    decoder: CardDecoder

    def setup_method(self):
        """Construct a decoder for each test."""
        self.decoder = CardDecoder()

    def test_creature_type_extracted(self):
        """An active 'creature' bit is decoded back to the type."""
        vec = make_encoded_vector(card_types=["creature"])
        types = self.decoder.item_from_vector(vec, "types")
        assert "creature" in types

    def test_no_active_returns_empty(self):
        """A vector with no active bits decodes to an empty list."""
        vec = make_encoded_vector()
        types = self.decoder.item_from_vector(vec, "types")
        assert types == []

    def test_threshold_respected(self):
        """Bits below the threshold are excluded; lowering it includes them."""
        layout_len = (
            len(CardFields.card_types())
            + len(CardFields.card_supertypes())
            + len(CardFields.card_subtypes())
            + 1
            + len(CardFields.color_identities())
            + 1
        )
        v = np.full((layout_len,), 0.4, dtype=np.float32)
        types = self.decoder.item_from_vector(v, "types", threshold=0.5)
        assert types == []

        types_low = self.decoder.item_from_vector(v, "types", threshold=0.3)
        assert len(types_low) == len(CardFields.card_types())


# ---------------------------------------------------------------------------
# land_mask_from_vectors
# ---------------------------------------------------------------------------


class TestLandMaskFromVectors:
    """Boolean mask over rows encoded as lands."""

    decoder: CardDecoder

    def setup_method(self):
        """Construct a decoder for each test."""
        self.decoder = CardDecoder()

    def test_mask_picks_out_land_rows(self):
        """The mask is True for land rows and False otherwise."""
        land_vec = make_encoded_vector(card_types=["land"])
        creature_vec = make_encoded_vector(card_types=["creature"])
        mat = torch.from_numpy(np.stack([land_vec, creature_vec]))
        mask = self.decoder.land_mask_from_vectors(mat)
        assert mask.tolist() == [True, False]

    def test_empty_input(self):
        """An empty matrix yields an empty mask."""
        empty = torch.zeros((0, 1000))
        mask = self.decoder.land_mask_from_vectors(empty)
        assert mask.shape == (0,)


# ---------------------------------------------------------------------------
# color_identity_mask_from_vectors
# ---------------------------------------------------------------------------


class TestColorIdentityMaskFromVectors:
    """Per-color identity mask matrix."""

    decoder: CardDecoder

    def setup_method(self):
        """Construct a decoder for each test."""
        self.decoder = CardDecoder()

    def test_color_mask_shape_and_contents(self):
        """The mask has the right shape and flags the present colors."""
        a = make_encoded_vector(color_identity=["U"])
        b = make_encoded_vector(color_identity=["W", "G"])
        mat = torch.from_numpy(np.stack([a, b]))
        mask = self.decoder.color_identity_mask_from_vectors(mat)
        assert mask.shape == (2, len(CardFields.color_identities()))
        u_idx = CardFields.color_identities().index("U")
        w_idx = CardFields.color_identities().index("W")
        g_idx = CardFields.color_identities().index("G")
        assert mask[0, u_idx].item() is True
        assert mask[1, w_idx].item() is True
        assert mask[1, g_idx].item() is True


# ---------------------------------------------------------------------------
# mana_value_from_vectors
# ---------------------------------------------------------------------------


class TestManaValueFromVectors:
    """Extract and clamp mana values from a stack of encoded vectors."""

    decoder: CardDecoder

    def setup_method(self):
        """Construct a decoder for each test."""
        self.decoder = CardDecoder()

    def test_typical_mana_values_returned(self):
        """Typical mana values are returned unchanged."""
        a = make_encoded_vector(mana=2)
        b = make_encoded_vector(mana=5)
        mat = torch.from_numpy(np.stack([a, b]))
        mv = self.decoder.mana_value_from_vectors(mat)
        assert mv.tolist() == [2.0, 5.0]

    def test_mana_values_clamped_to_zero_minimum(self):
        """Negative mana values clamp to zero."""
        a = make_encoded_vector(mana=-3)
        mat = torch.from_numpy(np.stack([a]))
        mv = self.decoder.mana_value_from_vectors(mat)
        assert mv.tolist() == [0.0]

    def test_mana_values_clamped_to_30_maximum(self):
        """Very large mana values clamp to the maximum of 30."""
        a = make_encoded_vector(mana=999)
        mat = torch.from_numpy(np.stack([a]))
        mv = self.decoder.mana_value_from_vectors(mat)
        assert mv.tolist() == [30.0]


# ---------------------------------------------------------------------------
# constrain_logits
# ---------------------------------------------------------------------------


class TestConstrainLogits:
    """`constrain_logits` projects raw logits into valid encoded card vectors."""

    decoder: CardDecoder

    def setup_method(self):
        """Construct a decoder for each test."""
        self.decoder = CardDecoder()

    def _logit_dim(self):
        """Return the full logit vector length (structural fields + embedding)."""
        return (
            len(CardFields.card_types())
            + len(CardFields.card_supertypes())
            + len(CardFields.card_subtypes())
            + 1
            + len(CardFields.color_identities())
            + 1
            + self.decoder.embed_dim
        )

    def test_output_shape_matches_input(self):
        """The constrained output keeps the input shape."""
        dim = self._logit_dim()
        x = torch.zeros((2, dim))
        out = self.decoder.constrain_logits(x)
        assert out.shape == x.shape

    def test_categorical_bits_binary(self):
        """Categorical (type) bits are projected to 0.0 or 1.0."""
        dim = self._logit_dim()
        x = torch.zeros((1, dim))
        out = self.decoder.constrain_logits(x)
        types_slice = self.decoder.slice("types", dim)
        types_bits = out[0, types_slice]
        for b in types_bits.tolist():
            assert b in (0.0, 1.0)

    def test_mana_clamped_to_range(self):
        """The mana slot is clamped into [0, 16]."""
        dim = self._logit_dim()
        x = torch.full((1, dim), -50.0)
        out = self.decoder.constrain_logits(x)
        mana_slice = self.decoder.slice("mana", dim)
        assert out[0, mana_slice].item() == 0.0

        x_hi = torch.full((1, dim), 999.0)
        out_hi = self.decoder.constrain_logits(x_hi)
        assert out_hi[0, mana_slice].item() == 16.0
