import json

class Card(object):
    def __init__(self, commander_legal: bool, card_name: str, card_types: list[str], card_supertypes: str, card_subtypes: list[str], mana_cost: int, mana_cost_exp: str, color_identity: list[str], defense: str, rarity: str, text: str, rank: str, power: str, toughness: str, loyalty: str, id: str) -> None:
        self.commander_legal: bool = commander_legal #Is the Card Legal in Commander? True if property exists and is legal, False otherwise.
        self.card_name: str = card_name #Name of the Card. None if property doesn't exist.
        self.card_types: list[str] =  card_types #Card Types (EX: CREATURE) of the Card. None if property doesn't exist.
        self.card_supertypes: str = card_supertypes #Supertypes (EX: LEGENDARY) of the Card. None if property doesn't exist.
        self.card_subtypes: list[str] = card_subtypes #Subtypes (EX: GOBLIN) of the Card. None if property doesn't exist.
        self.mana_cost: int = mana_cost #Numerical Mana Cost of the Card. None if property doesn't exist.
        self.mana_cost_exp: str = mana_cost_exp #Expanded Mana Cost of the Card. None if property doesn't exist.
        self.color_identity: list[str] = color_identity #Colors contained in the Expanded Mana Cost of the Card. None if property doesn't exist.
        self.defense: str = defense #Defense of a Battle Card. None if the Card is not a Battle.
        self.rarity: str = rarity #Rarity (EX: UNCOMMON) of the Card. None if property doesn't exist.
        self.text: str = text #Text of the Card. None if property doesn't exist.
        self.rank: str = rank #EDHREC Rank of the Card. None if property doesn't exist.
        self.power: str = power #Power of a Creature Card. None if Card is not a Creature.
        self.toughness: str = toughness #Toughness of a Creature Card. None if Card is not a Creature.
        self.loyalty: str = loyalty #Starting Loyalty of a Planeswalker Card. None if Card is not a Planeswalker.
        self.id: str = id #MultiverseID Identifier

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Card):
            return self.card_name == value.card_name
        return False

    def __str__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.get_attributes().items())

    def __hash__(self) -> int:
        return hash(self.rank)
    
    def __len__(self) -> int:
        return 1
    
    def get_attributes(self) -> dict:
        """
        get_attributes maps the name of attributes to the value of the attribute.

        return: dictionary of string:attribute
        """
        return {
            'card_name':self.card_name,
            'card_types':self.card_types,
            'card_supertypes':self.card_supertypes,
            'card_subtypes':self.card_subtypes,
            'mana_cost':self.mana_cost,
            'mana_cost_exp':self.mana_cost_exp,
            'color_identity':self.color_identity,
            'defense':self.defense,
            'rarity':self.rarity,
            'text':self.text,
            'rank':self.rank,
            'power':self.power,
            'toughness':self.toughness,
            'loyalty':self.loyalty,
            'id':self.id,
            # 'tags':CardFields.tag_card(self),
        }

    def to_json(self) -> str:
        """
        Converts the Card object into a JSON string.
        
        return: A JSON string representing the Card object.
        """
        return json.dumps(self.get_attributes(), indent=4)


if __name__ == "__main__":
    from MTG_DATABASE import DataSet
    ds = DataSet()