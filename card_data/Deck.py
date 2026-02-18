from .Card import Card
from .Card_Fields import CardFields
from .Card_Decoder import CardDecoder
import torch
import json
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
import torch
import json
from dataclasses import dataclass


class Deck(object):
    def __init__(self, id: str, colors: List[str], color_percentages: Dict[str, float], bracket: int, format: str, commanders: List[Card], companions: List[Card], mainboard_count: int, cards: List[Tuple[Card, int]]) -> None:
        self.id: str = id
        self.colors: List[str] = colors
        self.color_percentages: Dict[str, float] = color_percentages
        self.bracket: int = bracket
        self.mainboard_count: int = mainboard_count
        self.cards: List[Tuple[Card, int]] = cards
        self.cards_expanded: List[Card] = [card for card, count in self.cards for _ in range(count)]
        self.format: str = format
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
        return sum([qty for _, qty in self.cards])+len(self.commanders)+len(self.companions)
    
    def get_attributes(self) -> Dict:
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
        return json.dumps(self.get_attributes(), indent=4)
    
    def to_tensor(self, encoder) -> torch.Tensor:
        return torch.stack([torch.tensor(encoder.encode(x)[1], dtype=torch.float32) for x in self.all_cards])
    
    def shape_deck(self, commander_colors: List[str]) -> None:
        if len(self.cards) >= 99:
            self.cards = self.cards[:99]
            return
        basics = Deck.basic_lands_from_colors(commander_colors)
        need = 99 - len(self.cards)
        self.cards += (basics * ((need // len(basics)) + 1))[:need]

    @staticmethod
    def basic_lands_from_colors(colors: List[str]) -> List[str]:
        return [CardFields.color_basic_land_map()[color] for color in colors if color in CardFields.color_basic_land_map()]
    
class SimpleDeck(object):
    def __init__(self, id: str, commanders: List[str], cards: List[str]) -> None:
        self.id: str = id
        self.commanders: List[str] = commanders
        self.cards: List[str] = cards

    def __str__(self) -> str:
        return f"id: {self.id}\ncommanders: {self.commanders}\ncards: {self.cards}\n"
    
    def get_attributes(self) -> Dict:
        return {
            'id': self.id,
            'commanders': self.commanders,
            'cards': self.cards,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.get_attributes(), indent=4)

    def to_tensor_stack(self, vd) -> torch.Tensor:
        vecs = []
        for name in (self.commanders + self.cards):
            try:
                vecs.append(vd.find_vector(vd.find_id(name)).float())
            except Exception:
                continue
        return torch.stack(vecs, dim=0)
    
    @staticmethod
    def from_json(obj: Dict) -> 'SimpleDeck':
        return SimpleDeck(id=str(obj["id"]), commanders=list(obj["commanders"]), cards=list(obj.get("cards", [])))

    @staticmethod
    def load_json_file(path: str) -> List["SimpleDeck"]:
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
        if len(self.cards) >= 99:
            self.cards = self.cards[:99]
            return
        basics = SimpleDeck.basic_lands_from_colors(commander_colors)
        need = 99 - len(self.cards)
        self.cards += (basics * ((need // len(basics)) + 1))[:need]

    @staticmethod
    def basic_lands_from_colors(colors: List[str]) -> List[str]:
        return [CardFields.color_basic_land_map()[color] for color in colors if color in CardFields.color_basic_land_map()]

@dataclass
class PreparedDeckData:
    all_names: List[str]
    present_names: List[str]
    X: torch.Tensor
    missing_vectors: List[str]
    tags_by_card: Dict[str, List[str]]

    land_mask: torch.Tensor
    mana_value: torch.Tensor
    color_mask: torch.Tensor

class SimpleDeckAnalyzer(object):
    def __init__(self, deck: "SimpleDeck", tag_dataset: dict[str, dict[str, list]], vd):
        self.deck = deck
        self.tag_data = tag_dataset
        self.vd = vd
        self.decoder = CardDecoder()

        basics = CardFields.basic_lands()
        self._basic_set_lower = {str(x).strip().lower() for x in basics}

        self.prepared_deck = self._prepare()

    def analyze(self):
        out = {}
        out["tags"] = self.analyze_tags(self.prepared_deck)
        out["color_distribution"] = self.analyze_color_distribution(self.prepared_deck)
        out["curve"] = self.analyze_curve(self.prepared_deck)
        out["lands"] = self.analyze_lands_and_basics(self.prepared_deck)
        return out

    def _get_tags(self, name: str) -> List[str]:
        payload = self.tag_data.get(name)
        if isinstance(payload, dict):
            tags = payload.get("tags", [])
            if isinstance(tags, list):
                return [str(t) for t in tags if isinstance(t, (str, int, float))]
        return []

    def _prepare(self) -> PreparedDeckData:
        all_names = list(self.deck.commanders) + list(self.deck.cards)

        tags_by_card: Dict[str, List[str]] = {}
        vecs: List[torch.Tensor] = []
        present_names: List[str] = []
        missing_vectors: List[str] = []

        for name in all_names:
            tags_by_card[name] = self._get_tags(name)

            try:
                v = self.vd.find_vector(name)
            except Exception:
                v = None

            if v is None:
                missing_vectors.append(name)
                continue

            vecs.append(v)
            present_names.append(name)

        if len(vecs) == 0:
            X = torch.empty((0, 0))
            land_mask = torch.empty((0,), dtype=torch.bool)
            mana_value = torch.empty((0,), dtype=torch.float32)
            color_mask = torch.empty((0, len(list(CardFields.color_identities()))), dtype=torch.bool)
        else:
            X = torch.stack(vecs, dim=0)
            land_mask = self.decoder.land_mask_from_vectors(X, threshold=0.5)
            mana_value = self.decoder.mana_value_from_vectors(X)
            color_mask = self.decoder.color_identity_mask_from_vectors(X, threshold=0.5)

        return PreparedDeckData(
            all_names=all_names,
            present_names=present_names,
            X=X,
            missing_vectors=missing_vectors,
            tags_by_card=tags_by_card,
            land_mask=land_mask,
            mana_value=mana_value,
            color_mask=color_mask,
        )

    def analyze_tags(self, prep: PreparedDeckData) -> Dict[str, Any]:
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
        wanted = ["W", "U", "B", "R", "G"]
        color_idents = list(CardFields.color_identities())
        pos = {c: (color_idents.index(c) if c in color_idents else None) for c in wanted}

        counts = {}
        for c in wanted:
            p = pos[c]
            counts[c] = 0 if p is None or prep.color_mask.numel() == 0 else int(prep.color_mask[:, p].sum().item())

        M = int(prep.X.size(0)) if prep.X.numel() else 0
        percent = {c: counts[c] / max(1, M) for c in wanted}

        return {"colors": {"counts": counts, "percent": percent}}

    def analyze_curve(self, prep: PreparedDeckData) -> Dict[str, Any]:
        """
        Fast mana curve:
          - bucket = round(mv), clamp into [0..6] where 6 means "6+"
          - use torch.bincount to count per-bucket (no Python loop over cards)
        """
        M = int(prep.X.size(0)) if prep.X.numel() else 0
        if prep.mana_value.numel() == 0:
            curve = [0] * 7
            curve_pct = [0.0] * 7
            return {"mana_curve": {"counts": curve, "percent": curve_pct}}

        # Keep computation on whatever device mana_value is on.
        mv = prep.mana_value
        buckets = torch.round(mv).to(torch.int64)
        buckets = torch.clamp(buckets, min=0, max=6)

        counts = torch.bincount(buckets, minlength=7)  # (7,)
        curve = counts.detach().cpu().tolist()

        curve_pct = [c / max(1, M) for c in curve]
        return {"mana_curve": {"counts": curve, "percent": curve_pct}}

    def analyze_lands_and_basics(self, prep: PreparedDeckData) -> Dict[str, Any]:
        land_count = int(prep.land_mask.sum().item()) if prep.land_mask.numel() else 0

        basic_types: Dict[str, int] = {}
        basic_count = 0
        for name in prep.present_names:
            nm = name.strip().lower()
            if nm in self._basic_set_lower:
                basic_count += 1
                try:
                    t = CardFields.basic_type_name(nm)
                except Exception:
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