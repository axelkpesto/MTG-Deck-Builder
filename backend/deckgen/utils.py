"""Shared utility helpers for deck generation, training, and stats parsing."""

import os
import json
from typing import Any, List
import random
import numpy as np
import torch
from card_data import CardFields

def mana_value_bucket(mv: float) -> int:
    """Bucket mana value into range 0..6, where 6 means 6+."""
    mv_i = int(round(float(mv)))
    return 6 if mv_i >= 6 else max(0, mv_i)


def safe_read_json(path: str) -> Any:
    """Read and parse JSON from a path, raising if file is missing."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and Torch RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clamp_int(v: int, lo: int, hi: int) -> int:
    """Clamp an integer value to the inclusive range [lo, hi]."""
    return max(lo, min(hi, v))


def is_basic_land_name(name: str) -> bool:
    """Return whether a card name is recognized as a basic land."""
    return name.strip().lower() in CardFields.basic_lands()


def basic_land_type(name: str) -> str:
    """Map a basic land name to its normalized basic-land type string."""
    return CardFields.basic_type_name(name.strip().lower())


def duplicate_penalty(extra: int, lam: float, power: float) -> float:
    """Compute a power-law penalty for duplicates above target amount."""
    if extra <= 0 or lam <= 0:
        return 0.0
    return float(lam) * (float(extra) ** float(power))


def allowed_basic_land_types(commander_colors: torch.Tensor) -> List[str]:
    """Return allowed basic land names for the commander's color identity."""
    color_idents = CardFields.color_identities()
    mapping = CardFields.color_basic_land_map()

    allowed: List[str] = []
    for i, c in enumerate(color_idents):
        if c == "C":
            continue
        if bool(commander_colors[i].item()) and c in mapping:
            allowed.append(mapping[c])

    return allowed if allowed else ["Wastes"]


def extract_land_count(stats: dict) -> int:
    """Extract total land count from analyzer stats payload."""
    return int(stats["lands"]["lands"]["land_count"])


def extract_basic_ratio(stats: dict) -> float:
    """Extract basic-land ratio from analyzer stats payload."""
    return float(stats["lands"]["lands"]["basic_ratio"])


def extract_curve_counts(stats: dict) -> List[int]:
    """Extract mana-curve bucket counts from analyzer stats payload."""
    return [int(x) for x in stats["curve"]["mana_curve"]["counts"]]


def extract_tag_count(stats: dict, tag: str) -> int:
    """Extract count for a single tag from analyzer stats payload."""
    return int(stats["tags"]["tag_counts"].get(tag, 0))
