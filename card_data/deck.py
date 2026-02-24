"""Deck data models and analysis helpers."""

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import torch

from .card import Card
from .card_decoder import CardDecoder
from .card_fields import CardFields


class Deck:
    """Deck container for fully decoded card objects."""

    def __init__(self, deck_id: str | None, colors: List[str], color_percentages: Dict[str, float], bracket: int, deck_format: str | None, commanders: List[Card], companions: List[Card], mainboard_count: int, cards: List[Tuple[Card, int]]) -> None:
        """Initialize a deck model from structured card objects."""
        self.id: str = deck_id
        self.colors: List[str] = colors
        self.color_percentages: Dict[str, float] = color_percentages
        self.bracket: int = bracket
        self.mainboard_count: int = mainboard_count
        self.cards: List[Tuple[Card, int]] = cards
        self.cards_expanded: List[Card] = [card for card, count in self.cards for _ in range(count)]
        self.format: str = deck_format
        self.commanders: List[Card] = commanders
        self.companions: List[Card] = companions
        self.all_cards = self.commanders + self.companions + self.cards_expanded

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Deck):
            return self.cards == value.cards
        return False

    def __str__(self) -> str:
        return f"id: {self.id}\ncolors: {self.colors}\ncolor_percentages: {self.color_percentages}\nbracket: {self.bracket}\ncommander: {[commander.card_name for commander in self.commanders]}\ncompanion: {[companion.card_name for companion in self.companions]}\ncards: {[(card.card_name, quantity) for card, quantity in self.cards[:5]]}...\nmainboard_count: {self.mainboard_count}\n"

    def __len__(self) -> int:
        """Return total deck size including commanders/companions."""
        return sum(qty for _, qty in self.cards) + len(self.commanders) + len(self.companions)

    def get_attributes(self) -> Dict:
        """Serialize deck to a plain dictionary."""
        return {
            'id': self.id,
            'colors': self.colors,
            'color_percentages': self.color_percentages,
            'bracket': self.bracket,
            'format': self.format,
            'commanders': [commander.get_attributes() for commander in self.commanders],
            'companions': [companion.get_attributes() for companion in self.companions],
            'mainboard_count': self.mainboard_count,
            'cards': [{'card':card.get_attributes(), 'quantity': qty} for card, qty in self.cards],
        }

    def to_json(self) -> str:
        """Serialize deck to formatted JSON."""
        return json.dumps(self.get_attributes(), indent=4)

    def to_tensor(self, encoder) -> torch.Tensor:
        """Encode all cards in the deck into a stacked tensor."""
        return torch.stack([torch.tensor(encoder.encode(x)[1], dtype=torch.float32) for x in self.all_cards])

    def shape_deck(self, commander_colors: List[str]) -> None:
        """Pad or trim deck cards to 99 entries using basics."""
        if len(self.cards) >= 99:
            self.cards = self.cards[:99]
            return
        basics = Deck.basic_lands_from_colors(commander_colors)
        need = 99 - len(self.cards)
        self.cards += (basics * ((need // len(basics)) + 1))[:need]

    @staticmethod
    def basic_lands_from_colors(colors: List[str]) -> List[str]:
        """Map deck colors to recommended basic land names."""
        return [CardFields.color_basic_land_map()[color] for color in colors if color in CardFields.color_basic_land_map()]

class SimpleDeck:
    """Lightweight deck model using only card names."""

    def __init__(self, deck_id: str | None = None, commanders: List[str] | None = None, cards: List[str] | None = None) -> None:
        """Initialize simple deck structure from card names."""
        self.id: str = deck_id
        self.commanders: List[str] = commanders or []
        self.cards: List[str] = cards or []

    def __str__(self) -> str:
        return f"id: {self.id}\ncommanders: {self.commanders}\ncards: {self.cards}\n"

    def get_attributes(self) -> Dict:
        """Serialize simple deck to a plain dictionary."""
        return {
            'id': self.id,
            'commanders': self.commanders,
            'cards': self.cards,
        }

    def to_json(self) -> str:
        """Serialize simple deck to formatted JSON."""
        return json.dumps(self.get_attributes(), indent=4)

    def to_tensor_stack(self, vd) -> torch.Tensor:
        """Stack vectors for all known cards in the simple deck."""
        vecs = []
        for name in (self.commanders + self.cards):
            try:
                vecs.append(vd.find_vector(vd.find_id(name)).float())
            except (KeyError, AttributeError, RuntimeError, TypeError, ValueError):
                continue
        return torch.stack(vecs, dim=0)

    @staticmethod
    def from_json(obj: Dict) -> 'SimpleDeck':
        """Build `SimpleDeck` from JSON-compatible object."""
        return SimpleDeck(
            deck_id=str(obj["id"]),
            commanders=list(obj["commanders"]),
            cards=list(obj.get("cards", [])),
        )

    @staticmethod
    def load_json_file(path: str) -> List["SimpleDeck"]:
        """Load a list of `SimpleDeck` entries from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        items = raw.values() if isinstance(raw, dict) else raw
        return [SimpleDeck.from_json(x) for x in items]

    def __len__(self) -> int:
        return len(self.cards) + len(self.commanders)

    def __eq__(self, value: object) -> bool:
        if isinstance(value, SimpleDeck):
            return self.cards == value.cards and self.commanders == value.commanders
        return False

    def shape_deck(self, commander_colors: List[str]) -> None:
        """Pad or trim cards to 99 entries using basic lands."""
        if len(self.cards) >= 99:
            self.cards = self.cards[:99]
            return
        basics = SimpleDeck.basic_lands_from_colors(commander_colors)
        need = 99 - len(self.cards)
        self.cards += (basics * ((need // len(basics)) + 1))[:need]

    @staticmethod
    def basic_lands_from_colors(colors: List[str]) -> List[str]:
        """List of Lands """
        return [CardFields.color_basic_land_map()[color] for color in colors if color in CardFields.color_basic_land_map()]

@dataclass
class PreparedDeckData:
    """Precomputed vector/tags/features used by deck analyzers."""

    all_names: List[str]
    present_names: List[str]
    vectors: torch.Tensor
    missing_vectors: List[str]
    tags_by_card: Dict[str, List[str]]

    land_mask: torch.Tensor
    mana_value: torch.Tensor
    color_mask: torch.Tensor

class SimpleDeckAnalyzer:
    """Analyze deck composition, colors, mana curve, and tags."""

    def __init__(self, deck: "SimpleDeck", tag_dataset: dict[str, dict[str, list]], vd):
        """Initialize analyzer and cache prepared deck features."""
        self.deck = deck
        self.tag_data = tag_dataset
        self.vd = vd
        self.decoder = CardDecoder()

        basics = CardFields.basic_lands()
        self._basic_set_lower = {str(x).strip().lower() for x in basics}

        self.prepared_deck = self._prepare()

    def analyze(self):
        """Return full analysis payload for the prepared deck."""
        out = {}
        out["tags"] = self.analyze_tags(self.prepared_deck)
        out["color_distribution"] = self.analyze_color_distribution(self.prepared_deck)
        out["curve"] = self.analyze_curve(self.prepared_deck)
        out["lands"] = self.analyze_lands_and_basics(self.prepared_deck)
        return out

    def _get_tags(self, name: str) -> List[str]:
        """Extract known tags for a card name from tag dataset."""
        payload = self.tag_data.get(name)
        if isinstance(payload, dict):
            tags = payload.get("tags", [])
            if isinstance(tags, list):
                return [str(t) for t in tags if isinstance(t, (str, int, float))]
        return []

    def _prepare(self) -> PreparedDeckData:
        """Prepare deck vectors, masks, and tag lookup caches."""
        all_names = list(self.deck.commanders) + list(self.deck.cards)

        tags_by_card: Dict[str, List[str]] = {}
        vecs: List[torch.Tensor] = []
        present_names: List[str] = []
        missing_vectors: List[str] = []

        for name in all_names:
            tags_by_card[name] = self._get_tags(name)

            try:
                v = self.vd.find_vector(name)
            except (KeyError, AttributeError, RuntimeError, TypeError, ValueError):
                v = None

            if v is None:
                missing_vectors.append(name)
                continue

            vecs.append(v)
            present_names.append(name)

        if len(vecs) == 0:
            vectors = torch.empty((0, 0))
            land_mask = torch.empty((0,), dtype=torch.bool)
            mana_value = torch.empty((0,), dtype=torch.float32)
            color_mask = torch.empty((0, len(list(CardFields.color_identities()))), dtype=torch.bool)
        else:
            vectors = torch.stack(vecs, dim=0)
            land_mask = self.decoder.land_mask_from_vectors(vectors, threshold=0.5)
            mana_value = self.decoder.mana_value_from_vectors(vectors)
            color_mask = self.decoder.color_identity_mask_from_vectors(vectors, threshold=0.5)

        return PreparedDeckData(
            all_names=all_names,
            present_names=present_names,
            vectors=vectors,
            missing_vectors=missing_vectors,
            tags_by_card=tags_by_card,
            land_mask=land_mask,
            mana_value=mana_value,
            color_mask=color_mask,
        )

    def analyze_tags(self, prep: PreparedDeckData) -> Dict[str, Any]:
        """Compute tag counts and normalized frequencies."""
        tag_counts = defaultdict(int)

        for _, tags in prep.tags_by_card.items():
            for t in tags:
                tag_counts[t] += 1

        total = sum(tag_counts.values())
        tag_freq = {t: c / max(1, total) for t, c in tag_counts.items()}

        return {
            "tag_counts": dict(tag_counts),
            "tag_freq": tag_freq,
        }

    def analyze_color_distribution(self, prep: PreparedDeckData) -> Dict[str, Any]:
        """Compute color identity distribution across present vectors."""
        wanted = ["W", "U", "B", "R", "G"]
        color_idents = list(CardFields.color_identities())
        pos = {c: (color_idents.index(c) if c in color_idents else None) for c in wanted}

        counts = {}
        for c in wanted:
            p = pos[c]
            counts[c] = 0 if p is None or prep.color_mask.numel() == 0 else int(prep.color_mask[:, p].sum().item())

        total_cards = int(prep.vectors.size(0)) if prep.vectors.numel() else 0
        percent = {c: counts[c] / max(1, total_cards) for c in wanted}

        return {"colors": {"counts": counts, "percent": percent}}

    def analyze_curve(self, prep: PreparedDeckData) -> Dict[str, Any]:
        """Compute 0-6+ mana curve counts and percentages."""
        total_cards = int(prep.vectors.size(0)) if prep.vectors.numel() else 0
        if prep.mana_value.numel() == 0:
            curve = [0] * 7
            curve_pct = [0.0] * 7
            return {"mana_curve": {"counts": curve, "percent": curve_pct}}

        mv = prep.mana_value
        buckets = torch.round(mv).to(torch.int64)
        buckets = torch.clamp(buckets, min=0, max=6)

        counts = torch.bincount(buckets, minlength=7)  # (7,)
        curve = counts.detach().cpu().tolist()

        curve_pct = [c / max(1, total_cards) for c in curve]
        return {"mana_curve": {"counts": curve, "percent": curve_pct}}

    def analyze_lands_and_basics(self, prep: PreparedDeckData) -> Dict[str, Any]:
        """Compute land totals and basic-land composition stats."""
        land_count = int(prep.land_mask.sum().item()) if prep.land_mask.numel() else 0

        basic_types: Dict[str, int] = {}
        basic_count = 0
        for name in prep.present_names:
            nm = name.strip().lower()
            if nm in self._basic_set_lower:
                basic_count += 1
                try:
                    t = CardFields.basic_type_name(nm)
                except (KeyError, AttributeError, RuntimeError, TypeError, ValueError):
                    t = name
                basic_types[t] = basic_types.get(t, 0) + 1

        basic_ratio = (basic_count / max(1, land_count)) if land_count > 0 else 0.0

        return {
            "lands": {
                "land_count": land_count,
                "basic_count": basic_count,
                "basic_types": basic_types,
                "basic_ratio": basic_ratio,
            }
        }
