"""Tests for `SimpleDeckAnalyzer` deck-stats computations.

The analyzer pulls vectors from a `VectorDatabase`. We construct a tiny
`VectorDatabase` populated with handcrafted encoded vectors so the analyzer's
land mask, color mask, and mana-value computations are exercised end-to-end.
"""
import pytest

from backend.card_data import CardDecoder, SimpleDeck, SimpleDeckAnalyzer
from backend.vector_database import VectorDatabase

from .conftest import make_encoded_vector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _vd_with_cards(spec: dict[str, dict]) -> VectorDatabase:
    """Build a VectorDatabase from a {name: encoded-vector-args} mapping."""
    vd = VectorDatabase(None, CardDecoder())
    for name, kwargs in spec.items():
        vd.add_vector(name, make_encoded_vector(**kwargs))
    return vd


@pytest.fixture(name="tag_dataset_basic")
def _tag_dataset_basic():
    """A small tag dataset keyed by card name."""
    return {
        "Atraxa": {"tags": ["control", "card_draw"]},
        "Sol Ring": {"tags": ["ramp"]},
        "Forest": {"tags": []},
    }


@pytest.fixture(name="vd_basic")
def _vd_basic():
    """A VectorDatabase with a handful of handcrafted cards."""
    return _vd_with_cards({
        "Atraxa": {"card_types": ["creature"], "color_identity": ["W", "U", "B", "G"], "mana": 4},
        "Sol Ring": {"card_types": ["artifact"], "color_identity": [], "mana": 1},
        "Forest": {"card_types": ["land"], "color_identity": [], "mana": 0},
        "Mountain": {"card_types": ["land"], "color_identity": [], "mana": 0},
        "Lightning Bolt": {"card_types": ["instant"], "color_identity": ["R"], "mana": 1},
    })


# ---------------------------------------------------------------------------
# _prepare + analyze
# ---------------------------------------------------------------------------


class TestSimpleDeckAnalyzerPrepare:
    """`_prepare` populates vectors, masks, and missing-vector list."""

    def test_present_and_missing_split(self, vd_basic, tag_dataset_basic):
        """Cards split into present (in vd) and missing names."""
        deck = SimpleDeck(
            commanders=["Atraxa"],
            cards=["Sol Ring", "Mystery Card"],
        )
        analyzer = SimpleDeckAnalyzer(deck, tag_dataset_basic, vd_basic)
        assert analyzer.prepared_deck.present_names == ["Atraxa", "Sol Ring"]
        assert analyzer.prepared_deck.missing_vectors == ["Mystery Card"]

    def test_all_names_includes_commanders_and_cards(self, vd_basic, tag_dataset_basic):
        """`all_names` includes commanders followed by mainboard cards."""
        deck = SimpleDeck(commanders=["Atraxa"], cards=["Sol Ring"])
        analyzer = SimpleDeckAnalyzer(deck, tag_dataset_basic, vd_basic)
        assert analyzer.prepared_deck.all_names == ["Atraxa", "Sol Ring"]

    def test_tags_by_card_populated_from_dataset(self, vd_basic, tag_dataset_basic):
        """`tags_by_card` reflects the supplied tag dataset."""
        deck = SimpleDeck(commanders=["Atraxa"], cards=["Sol Ring", "Forest"])
        analyzer = SimpleDeckAnalyzer(deck, tag_dataset_basic, vd_basic)
        assert analyzer.prepared_deck.tags_by_card["Atraxa"] == ["control", "card_draw"]
        assert analyzer.prepared_deck.tags_by_card["Forest"] == []

    def test_empty_deck_produces_empty_tensors(self, vd_basic):
        """An empty deck yields empty vector and land-mask tensors."""
        deck = SimpleDeck()
        analyzer = SimpleDeckAnalyzer(deck, {}, vd_basic)
        prep = analyzer.prepared_deck
        assert prep.vectors.numel() == 0
        assert prep.land_mask.numel() == 0


# ---------------------------------------------------------------------------
# analyze_tags
# ---------------------------------------------------------------------------


class TestAnalyzeTags:
    """Tag frequency and counts."""

    def test_counts_accumulate(self, vd_basic):
        """Tag counts accumulate across cards."""
        tag_dataset = {
            "Atraxa": {"tags": ["control", "card_draw"]},
            "Sol Ring": {"tags": ["control", "ramp"]},
        }
        deck = SimpleDeck(commanders=["Atraxa"], cards=["Sol Ring"])
        analyzer = SimpleDeckAnalyzer(deck, tag_dataset, vd_basic)
        result = analyzer.analyze_tags(analyzer.prepared_deck)
        assert result["tag_counts"]["control"] == 2
        assert result["tag_counts"]["card_draw"] == 1
        assert result["tag_counts"]["ramp"] == 1

    def test_tag_freq_sums_to_one(self, vd_basic):
        """Tag frequencies normalize to sum to one."""
        tag_dataset = {
            "Atraxa": {"tags": ["a", "b"]},
            "Sol Ring": {"tags": ["a"]},
        }
        deck = SimpleDeck(commanders=["Atraxa"], cards=["Sol Ring"])
        analyzer = SimpleDeckAnalyzer(deck, tag_dataset, vd_basic)
        result = analyzer.analyze_tags(analyzer.prepared_deck)
        assert pytest.approx(sum(result["tag_freq"].values()), rel=1e-6) == 1.0

    def test_no_tags_yields_empty_counts(self, vd_basic):
        """With no tag data, counts are empty."""
        deck = SimpleDeck(commanders=["Atraxa"], cards=[])
        analyzer = SimpleDeckAnalyzer(deck, {}, vd_basic)
        result = analyzer.analyze_tags(analyzer.prepared_deck)
        assert not result["tag_counts"]


# ---------------------------------------------------------------------------
# analyze_color_distribution
# ---------------------------------------------------------------------------


class TestAnalyzeColorDistribution:
    """Color counts per WUBRG bucket."""

    def test_counts_per_color(self, vd_basic, tag_dataset_basic):
        """Each color's count reflects the cards contributing it."""
        deck = SimpleDeck(commanders=["Atraxa"], cards=["Lightning Bolt", "Sol Ring"])
        analyzer = SimpleDeckAnalyzer(deck, tag_dataset_basic, vd_basic)
        out = analyzer.analyze_color_distribution(analyzer.prepared_deck)
        counts = out["colors"]["counts"]
        assert counts["W"] == 1
        assert counts["U"] == 1
        assert counts["B"] == 1
        assert counts["G"] == 1
        assert counts["R"] == 1

    def test_percent_is_count_divided_by_total(self, vd_basic, tag_dataset_basic):
        """Color percentages equal count divided by present-card total."""
        deck = SimpleDeck(commanders=["Atraxa"], cards=["Lightning Bolt"])
        analyzer = SimpleDeckAnalyzer(deck, tag_dataset_basic, vd_basic)
        out = analyzer.analyze_color_distribution(analyzer.prepared_deck)
        percent = out["colors"]["percent"]
        assert percent["W"] == pytest.approx(0.5)
        assert percent["U"] == pytest.approx(0.5)
        assert percent["B"] == pytest.approx(0.5)
        assert percent["G"] == pytest.approx(0.5)
        assert percent["R"] == pytest.approx(0.5)

    def test_empty_deck_zero_counts(self, vd_basic):
        """An empty deck has zero counts for every color."""
        deck = SimpleDeck()
        analyzer = SimpleDeckAnalyzer(deck, {}, vd_basic)
        out = analyzer.analyze_color_distribution(analyzer.prepared_deck)
        assert all(c == 0 for c in out["colors"]["counts"].values())


# ---------------------------------------------------------------------------
# analyze_curve
# ---------------------------------------------------------------------------


class TestAnalyzeCurve:
    """Mana-curve histogram with 0..6+ buckets."""

    def test_curve_buckets_have_length_7(self, vd_basic, tag_dataset_basic):
        """The curve has seven count and percent buckets."""
        deck = SimpleDeck(commanders=["Atraxa"], cards=["Sol Ring", "Forest"])
        analyzer = SimpleDeckAnalyzer(deck, tag_dataset_basic, vd_basic)
        out = analyzer.analyze_curve(analyzer.prepared_deck)
        assert len(out["mana_curve"]["counts"]) == 7
        assert len(out["mana_curve"]["percent"]) == 7

    def test_curve_assigns_correct_buckets(self, vd_basic, tag_dataset_basic):
        """Each card lands in its mana-value bucket."""
        deck = SimpleDeck(commanders=["Atraxa"], cards=["Sol Ring", "Forest"])
        analyzer = SimpleDeckAnalyzer(deck, tag_dataset_basic, vd_basic)
        counts = analyzer.analyze_curve(analyzer.prepared_deck)["mana_curve"]["counts"]
        assert counts[0] == 1  # Forest
        assert counts[1] == 1  # Sol Ring
        assert counts[4] == 1  # Atraxa

    def test_high_cmc_clamped_to_six(self):
        """A high mana value is clamped into the 6+ bucket."""
        vd = _vd_with_cards({
            "Big Spell": {"card_types": ["sorcery"], "mana": 12},
        })
        deck = SimpleDeck(commanders=[], cards=["Big Spell"])
        analyzer = SimpleDeckAnalyzer(deck, {}, vd)
        counts = analyzer.analyze_curve(analyzer.prepared_deck)["mana_curve"]["counts"]
        assert counts[6] == 1

    def test_empty_deck_zero_curve(self, vd_basic):
        """An empty deck has an all-zero curve."""
        deck = SimpleDeck()
        analyzer = SimpleDeckAnalyzer(deck, {}, vd_basic)
        out = analyzer.analyze_curve(analyzer.prepared_deck)
        assert out["mana_curve"]["counts"] == [0] * 7


# ---------------------------------------------------------------------------
# analyze_lands_and_basics
# ---------------------------------------------------------------------------


class TestAnalyzeLandsAndBasics:
    """Counts of lands, basics, and basic-type histograms."""

    def test_counts_lands_and_basics(self, vd_basic, tag_dataset_basic):
        """Land and basic-land counts are tallied."""
        deck = SimpleDeck(commanders=["Atraxa"], cards=["Forest", "Mountain", "Sol Ring"])
        analyzer = SimpleDeckAnalyzer(deck, tag_dataset_basic, vd_basic)
        out = analyzer.analyze_lands_and_basics(analyzer.prepared_deck)
        assert out["lands"]["land_count"] == 2
        assert out["lands"]["basic_count"] == 2

    def test_basic_types_histogram(self, vd_basic, tag_dataset_basic):
        """Basic-land types are histogrammed by name."""
        deck = SimpleDeck(commanders=[], cards=["Forest", "Mountain"])
        analyzer = SimpleDeckAnalyzer(deck, tag_dataset_basic, vd_basic)
        out = analyzer.analyze_lands_and_basics(analyzer.prepared_deck)
        types = out["lands"]["basic_types"]
        assert types.get("Forest") == 1
        assert types.get("Mountain") == 1

    def test_basic_ratio(self, vd_basic, tag_dataset_basic):
        """The basic ratio is 1.0 when all lands are basics."""
        deck = SimpleDeck(commanders=[], cards=["Forest", "Mountain"])
        analyzer = SimpleDeckAnalyzer(deck, tag_dataset_basic, vd_basic)
        out = analyzer.analyze_lands_and_basics(analyzer.prepared_deck)
        assert out["lands"]["basic_ratio"] == 1.0

    def test_basic_ratio_zero_when_no_lands(self, vd_basic, tag_dataset_basic):
        """The basic ratio is 0.0 when there are no lands."""
        deck = SimpleDeck(commanders=[], cards=["Sol Ring"])
        analyzer = SimpleDeckAnalyzer(deck, tag_dataset_basic, vd_basic)
        out = analyzer.analyze_lands_and_basics(analyzer.prepared_deck)
        assert out["lands"]["basic_ratio"] == 0.0


# ---------------------------------------------------------------------------
# Top-level analyze()
# ---------------------------------------------------------------------------


class TestTopLevelAnalyze:
    """`analyze()` returns the four expected sections."""

    def test_returns_all_sections(self, vd_basic, tag_dataset_basic):
        """`analyze` returns tags, color_distribution, curve, and lands."""
        deck = SimpleDeck(commanders=["Atraxa"], cards=["Forest", "Sol Ring"])
        analyzer = SimpleDeckAnalyzer(deck, tag_dataset_basic, vd_basic)
        out = analyzer.analyze()
        for key in ("tags", "color_distribution", "curve", "lands"):
            assert key in out
