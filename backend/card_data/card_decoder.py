"""Utilities for decoding and constraining encoded MTG card vectors."""

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from card_data.card_fields import CardFields

class CardDecoder:
    """Decode vectorized cards into readable fields and constrained tensors."""

    def __init__(self, embed_dim: int = 686):
        """Initialize field maps and slice definitions for encoded vectors."""
        self.card_types = CardFields.card_types()
        self.card_supertypes = CardFields.card_supertypes()
        self.all_subtypes = CardFields.card_subtypes()
        self.color_identities = CardFields.color_identities()
        self.rarities = CardFields.rarities()
        self.embed_dim: int = int(embed_dim)

        self.field_map: Dict[str, list[str]] = {
            "types": self.card_types,
            "supertypes": self.card_supertypes,
            "subtypes": self.all_subtypes,
            "mana": self.color_identities,
            "colors": self.color_identities,
            "rarity": self.rarities
        }

        self.slice_map: Dict[str, tuple[int, int]] = {
            "types": (0, len(self.card_types)),
            "supertypes": (len(self.card_types), len(self.card_types) + len(self.card_supertypes)),
            "subtypes": (len(self.card_types) + len(self.card_supertypes), len(self.card_types) + len(self.card_supertypes) + len(self.all_subtypes)),
            "mana": (len(self.card_types) + len(self.card_supertypes) + len(self.all_subtypes), len(self.card_types) + len(self.card_supertypes) + len(self.all_subtypes) + 1),
            "colors": (len(self.card_types) + len(self.card_supertypes) + len(self.all_subtypes) + 1, len(self.card_types) + len(self.card_supertypes) + len(self.all_subtypes) + 1 + len(self.color_identities)),
            "rarity": (len(self.card_types) + len(self.card_supertypes) + len(self.all_subtypes) + 1 + len(self.color_identities), len(self.card_types) + len(self.card_supertypes) + len(self.all_subtypes) + 1 + len(self.color_identities) + 1),
        }

    def decode(self, card_name: str, encoded_vector: np.ndarray) -> str:
        """Decode a card vector to a human-readable multiline string."""
        d = self.decode_to_dict(card_name, encoded_vector)
        return "\n".join([f"{k}: {v}" for k, v in d.items()])

    def decode_to_dict(self, card_name: str, encoded_vector: np.ndarray) -> Dict[str, str]:
        """Decode a card vector to a dictionary of display fields."""
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
        """Convert encoded rarity integer into rarity string."""
        return CardFields.rarity_map().get(rarity_int, 'Unknown')

    def slice(self, key: str, dim: int) -> slice:
        """Return the slice for a feature group given full vector dimension."""
        if key == "embed":
            return slice(dim - self.embed_dim, dim)
        a, b = self.slice_map[key]
        return slice(a, b)

    def item_from_vector(self, vec: torch.Tensor, value: str, threshold: float = 0.5) -> List[str]:
        """Return active categorical items from a vector feature slice."""
        vec_np = np.asarray(vec)
        s = self.slice(value, vec_np.shape[-1])
        hot = vec_np[s] > threshold
        return [c for c, on in zip(self.field_map[value], hot) if on]

    def constrain_logits(self, logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Project raw logits into valid card-feature ranges and masks."""
        dim = logits.size(-1)
        s_types       = self.slice("types", dim)
        s_supertypes  = self.slice("supertypes", dim)
        s_subtypes    = self.slice("subtypes", dim)
        s_mana        = self.slice("mana", dim)
        s_colors      = self.slice("colors", dim)
        s_rarity      = self.slice("rarity", dim)
        s_embed       = self.slice("embed", dim)

        types      = (torch.sigmoid(logits[..., s_types])      >= threshold).to(logits.dtype)
        supertypes = (torch.sigmoid(logits[..., s_supertypes]) >= threshold).to(logits.dtype)
        subtypes   = (torch.sigmoid(logits[..., s_subtypes])   >= threshold).to(logits.dtype)
        colors     = (torch.sigmoid(logits[..., s_colors])     >= threshold).to(logits.dtype)

        mana = torch.round(logits[..., s_mana])
        mana = torch.clamp(mana, 0, 16)

        rarity = torch.round(logits[..., s_rarity])
        rarity = torch.clamp(rarity, 1, len(self.rarities) + 1)

        embed = F.normalize(logits[..., s_embed], dim=-1)

        constrained = torch.cat([types, supertypes, subtypes, mana, colors, rarity, embed], dim=-1)
        return constrained

    def land_mask_from_vectors(self, node_features: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return boolean mask of rows predicted/encoded as lands."""
        land_type_index = CardFields.card_types().index("land")
        type_slice_start, _ = self.slice_map["types"]
        return node_features[:, type_slice_start + land_type_index] >= threshold

    def color_identity_mask_from_vectors(self, node_features: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return boolean color-identity mask from encoded node features."""
        color_slice_start, color_slice_end = self.slice_map["colors"]
        return node_features[:, color_slice_start:color_slice_end] >= threshold

    def mana_value_from_vectors(self, node_features: torch.Tensor) -> torch.Tensor:
        """Return clamped mana values from encoded node features."""
        mana_slice_start, mana_slice_end = self.slice_map["mana"]
        mv = node_features[:, mana_slice_start:mana_slice_end].squeeze(-1)
        return torch.clamp(mv, min=0.0, max=30.0)
