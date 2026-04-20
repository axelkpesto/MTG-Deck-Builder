from typing import Optional, Tuple

import numpy as np

from backend.card_data.card import Card
from backend.card_data.card_fields import CardFields

class CardEncoder:

    def __init__(self, embed_model_name: Optional[str] ="all-MiniLM-L6-v2"):
        """Initialize encoder metadata and optionally load a text embedding model.

        Args:
            embed_model_name: SentenceTransformer model name for text embeddings.
                Pass None to skip embedding and produce shorter vectors.

        Returns:
            None
        """
        self.embed_model_name = embed_model_name
        self.card_types = CardFields.card_types()
        self.card_supertypes = CardFields.card_supertypes()
        self.all_subtypes = CardFields.card_subtypes()
        self.color_identities = CardFields.color_identities()
        self.tags = CardFields.card_tags()
        if embed_model_name:
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer(embed_model_name)

    def encode(self, crd: Card) -> Tuple[str, np.ndarray]:
        """Encode a card into a (card_name, feature_vector) pair.

        Args:
            crd: The Card object to encode.

        Returns:
            A tuple of (card_name, float32 numpy array) representing the card.
        """
        ret = []

        cd = [0] * len(self.card_types)
        for i, card_type in enumerate(self.card_types):
            if card_type in crd.card_types:
                cd[i] = 1
        ret += cd

        cd = [0] * len(self.card_supertypes)
        for i, card_supertype in enumerate(self.card_supertypes):
            if card_supertype in crd.card_supertypes:
                cd[i] = 1
        ret += cd

        cd = [0] * len(self.all_subtypes)
        for i, card_subtype in enumerate(self.all_subtypes):
            if card_subtype in crd.card_subtypes:
                cd[i] = 1
        ret += cd

        ret += [int(crd.mana_cost)]

        cd = [0] * len(self.color_identities)
        if not crd.color_identity:
            if crd.mana_cost > 0:
                cd[self.color_identities.index("C")] = 1
        else:
            for i, color_identity in enumerate(self.color_identities):
                if color_identity.upper() in crd.color_identity:
                    cd[i] = 1
        ret += cd

        ret += [self.rarity_to_int(crd.rarity)]

        if self.embed_model_name:
            embed = self.embed_model.encode(crd.text if crd.text else "", normalize_embeddings=True)
            ret += embed.tolist()

        return crd.card_name, np.array(ret, dtype=np.float32)

    def rarity_to_int(self, rarity: str) -> int:
        """Map a rarity string to its encoded integer index.

        Args:
            rarity: Rarity string (e.g. 'common', 'rare').

        Returns:
            Integer index corresponding to the rarity.

        Raises:
            ValueError: If the rarity string is not recognized.
        """
        try:
            return CardFields.rarity_to_index()[rarity]
        except KeyError as e:
            raise ValueError(f"Unknown rarity: {rarity!r}") from e
