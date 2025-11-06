from sentence_transformers import SentenceTransformer
from card_data.Card import Card
from card_data.Card_Fields import CardFields
import numpy as np
from typing import Optional, Tuple

class CardEncoder(object):
    def __init__(self, embed_model_name: Optional[str] ="all-MiniLM-L6-v2"):
        self.embed_model_name = embed_model_name
        self.card_types = CardFields.card_types()
        self.card_supertypes = CardFields.card_supertypes()
        self.all_subtypes = CardFields.card_subtypes()
        self.color_identities = CardFields.color_identities()
        self.tags = CardFields.card_tags()
        if embed_model_name:
            self.embed_model = SentenceTransformer(embed_model_name)

    def encode(self, crd: Card) -> Tuple[str, np.ndarray]:
        ret = []

        # Type encodings
        cd = [0] * len(self.card_types)
        for i in range(len(self.card_types)):
            if self.card_types[i] in crd.card_types:
                cd[i] = 1
        ret += cd

        cd = [0] * len(self.card_supertypes)
        for i in range(len(self.card_supertypes)):
            if self.card_supertypes[i] in crd.card_supertypes:
                cd[i] = 1
        ret += cd

        cd = [0] * len(self.all_subtypes)
        for i in range(len(self.all_subtypes)):
            if self.all_subtypes[i] in crd.card_subtypes:
                cd[i] = 1
        ret += cd

        # Mana cost
        ret += [int(crd.mana_cost)]

        # Color identity
        cd = [0] * len(self.color_identities)
        if not crd.color_identity:
            if crd.mana_cost > 0:
                cd[self.color_identities.index("C")] = 1
        else:
            for i in range(len(self.color_identities)):
                if self.color_identities[i].upper() in crd.color_identity:
                    cd[i] = 1
        ret += cd

        # Rarity
        ret += [self.rarity_to_int(crd.rarity)]

        # Text embedding
        if self.embed_model_name:
            embed = self.embed_model.encode(crd.text if crd.text else "", normalize_embeddings=True)
            ret += embed.tolist()

        return (crd.card_name, np.array(ret, dtype=np.float32))

    def rarity_to_int(self, rarity: str) -> int:
        return CardFields.rarity_to_index().get(rarity, Exception("Rarity Not Found"))