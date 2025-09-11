
from Card_Fields import CardFields
import numpy as np

class CardDecoder(object):
    def __init__(self, embed_dim: int | None = 384):
        self.card_types = CardFields.card_types()
        self.card_supertypes = CardFields.card_supertypes()
        self.all_subtypes = CardFields.card_subtypes()
        self.color_identities = CardFields.color_identities()
        self.tags = CardFields.card_tags()
        self.embed_dim = embed_dim

    def decode(self, card_name: str, encoded_vector: np.ndarray) -> str:
        d = self.decode_to_dict(card_name, encoded_vector)
        return "\n".join([f"{k}: {v}" for k, v in d.items()])

    def decode_to_dict(self, card_name: str, encoded_vector: np.ndarray) -> dict[str, str]:
        idx = 0

        card_types = [self.card_types[i] for i, v in enumerate(encoded_vector[idx:idx + len(self.card_types)]) if v == 1]
        idx += len(self.card_types)

        supertypes = [self.card_supertypes[i] for i, v in enumerate(encoded_vector[idx:idx + len(self.card_supertypes)]) if v == 1]
        idx += len(self.card_supertypes)

        subtypes = [self.all_subtypes[i] for i, v in enumerate(encoded_vector[idx:idx + len(self.all_subtypes)]) if v == 1]
        idx += len(self.all_subtypes)

        mana_cost = int(encoded_vector[idx])
        idx += 1

        color_identity = [self.color_identities[i] for i, v in enumerate(encoded_vector[idx:idx + len(self.color_identities)]) if v == 1]
        idx += len(self.color_identities)

        rarity = self.int_to_rarity(int(encoded_vector[idx]))
        idx += 1
        
        if self.embed_dim:
            text_embed = encoded_vector[idx:idx + self.embed_dim]

        return {
            "Name": card_name.capitalize(),
            "Types": str(card_types).capitalize(),
            "Supertypes": str(supertypes).capitalize(),
            "Subtypes": str(subtypes).capitalize(),
            "Mana Cost": str(mana_cost),
            "Color Identity": str(color_identity),
            "Rarity": str(rarity).capitalize(),
            "Text Embedding (truncated)": str(text_embed[:5]) + "..."
        }

    def int_to_rarity(self, rarity_int: int) -> str:
        return {
            1: 'Common',
            2: 'Uncommon',
            3: 'Rare',
            4: 'Mythic',
            5: 'Timeshifted',
            6: 'Masterpiece',
            7: 'Special'
        }.get(rarity_int, 'Unknown')
