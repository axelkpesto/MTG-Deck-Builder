from card_data.Card import Card
from card_data.Card_Fields import CardFields
import torch
import json
from typing import List, Tuple, Dict

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