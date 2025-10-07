from card_data.Card import Card
from card_data.Card_Encoder import CardEncoder
import torch
import json

class Deck(object):
    def __init__(self, id: str, colors: list[str], color_percentages: dict[str, float], bracket: int, format: str, commanders: list[Card], companions: list[Card], mainboard_count: int, cards: list[tuple[Card, int]]) -> None:
        self.id: str = id
        self.colors: list[str] = colors
        self.color_percentages: dict[str, float] = color_percentages
        self.bracket: int = bracket
        self.mainboard_count: int = mainboard_count
        self.cards: list[tuple[Card, int]] = cards
        self.cards_expanded: list[Card] = [card for card, count in self.cards for _ in range(count)]
        self.format: str  = format
        self.commanders: list[Card] = commanders
        self.companions: list[Card] = companions
        self.all_cards = self.commanders + self.companions + self.cards_expanded
        self.encoder = CardEncoder()

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Deck):
            return self.cards == value.cards
        return False

    def __str__(self) -> str:
        return f"id: {self.id}\ncolors: {self.colors}\ncolor_percentages: {self.color_percentages}\nbracket: {self.bracket}\ncommander: {[commander.card_name for commander in self.commanders]}\ncompanion: {[companion.card_name for companion in self.companions]}\ncards: {[(card.card_name, quantity) for card, quantity in self.cards[:5]]}...\nmainboard_count: {self.mainboard_count}\n"
    
    def __len__(self) -> int:
        return sum([qty for _, qty in self.cards])+len(self.commanders)+len(self.companions)
    
    def get_attributes(self) -> dict:
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
    
    def to_tensor(self) -> torch.tensor:
        return torch.stack([torch.tensor(self.encoder.encode(x)[1], dtype=torch.float32) for x in self.all_cards])
