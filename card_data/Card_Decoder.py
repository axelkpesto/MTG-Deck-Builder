from card_data.Card_Fields import CardFields
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional

class CardDecoder(object):
    def __init__(self, embed_dim: Optional[int] = 686):
        self.card_types = CardFields.card_types()
        self.card_supertypes = CardFields.card_supertypes()
        self.all_subtypes = CardFields.card_subtypes()
        self.color_identities = CardFields.color_identities()
        self.rarities = CardFields.rarities()
        self.embed_dim = embed_dim

        self.field_map = {
            "types": self.card_types,
            "supertypes": self.card_supertypes,
            "subtypes": self.all_subtypes,
            "mana": self.color_identities,
            "colors": self.color_identities,
            "rarity": self.rarities
        }

        self.slice_map = {
            "types": (0, len(self.card_types)),
            "supertypes": (len(self.card_types), len(self.card_types) + len(self.card_supertypes)),
            "subtypes": (len(self.card_types) + len(self.card_supertypes), len(self.card_types) + len(self.card_supertypes) + len(self.all_subtypes)),
            "mana": (len(self.card_types) + len(self.card_supertypes) + len(self.all_subtypes), len(self.card_types) + len(self.card_supertypes) + len(self.all_subtypes) + 1),
            "colors": (len(self.card_types) + len(self.card_supertypes) + len(self.all_subtypes) + 1, len(self.card_types) + len(self.card_supertypes) + len(self.all_subtypes) + 1 + len(self.color_identities)),
            "rarity": (len(self.card_types) + len(self.card_supertypes) + len(self.all_subtypes) + 1 + len(self.color_identities), len(self.card_types) + len(self.card_supertypes) + len(self.all_subtypes) + 1 + len(self.color_identities) + 1),
            "embed": ("tail", embed_dim),
            "head_len": len(self.card_types) + len(self.card_supertypes) + len(self.all_subtypes) + 1 + len(self.color_identities) + 1
        }

    def decode(self, card_name: str, encoded_vector: np.ndarray) -> str:
        d = self.decode_to_dict(card_name, encoded_vector)
        return "\n".join([f"{k}: {v}" for k, v in d.items()])

    def decode_to_dict(self, card_name: str, encoded_vector: np.ndarray) -> Dict[str, str]:
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

        return {
            "Name": card_name.capitalize(),
            "Types": str([x.capitalize() for x in card_types]),
            "Supertypes": str([x.capitalize() for x in supertypes]),
            "Subtypes": str([x.capitalize() for x in subtypes]),
            "Mana Cost": str(mana_cost),
            "Color Identity": str([x.capitalize() for x in color_identity]),
            "Rarity": str(rarity).capitalize(),
        }

    def int_to_rarity(self, rarity_int: int) -> str:
        return CardFields.rarity_map().get(rarity_int, 'Unknown')
    
    def slice(self, key: str, dim: int) -> slice:
        if key == "embed":
            return slice(dim - self.embed_dim, dim)
        a, b = self.slice_map[key]
        return slice(a, b)

    def item_from_vector(self, vec: torch.Tensor, value: str, threshold: float = 0.5) -> List[str]:
        vec = np.asarray(vec)
        s = self.slice(value, vec.shape[-1])
        hot = vec[s] > threshold
        return [c for c, on in zip(self.field_map[value], hot) if on]
    
    def constrain_logits(self, X: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        dim = X.size(-1)
        s_types       = self.slice("types", dim)
        s_supertypes  = self.slice("supertypes", dim)
        s_subtypes    = self.slice("subtypes", dim)
        s_mana        = self.slice("mana", dim)
        s_colors      = self.slice("colors", dim)
        s_rarity      = self.slice("rarity", dim)
        s_embed       = self.slice("embed", dim)

        types      = (torch.sigmoid(X[..., s_types])      >= threshold).to(X.dtype)
        supertypes = (torch.sigmoid(X[..., s_supertypes]) >= threshold).to(X.dtype)
        subtypes   = (torch.sigmoid(X[..., s_subtypes])   >= threshold).to(X.dtype)
        colors     = (torch.sigmoid(X[..., s_colors])     >= threshold).to(X.dtype)

        mana = torch.round(X[..., s_mana])
        mana = torch.clamp(mana, 0, 16)

        rarity = torch.round(X[..., s_rarity])
        rarity = torch.clamp(rarity, 1, len(self.rarities) + 1)

        embed = F.normalize(X[..., s_embed], dim=-1)

        Y = torch.cat([types, supertypes, subtypes, mana, colors, rarity, embed], dim=-1)
        return Y