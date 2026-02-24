"""Configuration dataclasses for deck generation and training."""

from dataclasses import dataclass, field
from typing import Tuple
from config import CONFIG


@dataclass(frozen=True)
class TagPenalty:
    """Penalty schedule for a single gameplay tag count target range."""

    tag: str
    min_target: int
    max_target: int
    soft_penalty: float
    hard_penalty: float
    hard_buffer: int = 0


@dataclass(frozen=True)
class DeckGenPaths:
    """Filesystem paths used by deck generation runtime."""

    nodes_pt: str = CONFIG.datasets["GRAPH_NODES_DATA_PATH"]
    edges_pt: str = CONFIG.datasets["GRAPH_EDGES_DATA_PATH"]
    vectors_pt: str = CONFIG.datasets["VECTOR_DATABASE_PATH"]
    decks_json: str = CONFIG.datasets["DECKS_DATASET_PATH"]
    tags_json: str = CONFIG.datasets["TAGS_DATASET_PATH"]
    ckpt_pt: str = CONFIG.models["GEN_MODEL_PATH"]


@dataclass(frozen=True)
class GenConfig:
    """Generation-time hyperparameters and target constraints."""

    deck_size: int = 100
    temperature: float = 0.85
    top_k: int = 512

    # Candidate pool
    neighbor_k: int = 600
    candidate_budget: int = 8000
    explore_random: int = 2000

    # Similarity (for learned targets)
    similar_commander_k: int = 50
    min_samples: int = 25

    # Quantiles for learned targets (from SimpleDeckAnalyzer stats)
    land_q: float = 0.50
    land_cap_q: float = 0.90
    basic_ratio_q: float = 0.50
    ramp_q: float = 0.50
    tag_min_q: float = 0.25
    tag_max_q: float = 0.75

    # Hard clamps for learned targets (explicit so generator doesn't need getattr fallbacks)
    land_min: int = 32
    land_max: int = 45

    basic_ratio_min: float = 0.10
    basic_ratio_max: float = 0.65

    basics_min: int = 6
    basics_max: int = 30

    # Land pressure
    land_boost: float = 1.25
    land_penalty_after_target: float = 5.0

    # Basics behavior
    basic_min_boost: float = 1.2
    basic_type_boost: float = 1.4
    basic_force_boost: float = 6.0
    basic_land_penalty_when_over_ratio: float = 2.0
    force_basic_when_behind: bool = True

    # Duplicate basic-land penalty (discourage excessive one basic type)
    dup_penalty_lambda: float = 0.12
    dup_penalty_power: float = 0.6
    dup_penalty_cap: float = 1.25

    # Curve shaping
    curve_penalty: float = 1.25

    # Ramp moderation (soft/hard cap)
    ramp_tag: str = "ramp"
    ramp_min: int = 9
    ramp_max: int = 14
    ramp_soft_penalty: float = 1.25
    ramp_hard_penalty: float = 3.25
    ramp_hard_buffer: int = 2

    # "Common" tag moderation (soft/hard cap)
    common_tag_penalties: Tuple[TagPenalty, ...] = field(
        default_factory=lambda: (
            TagPenalty(tag="card_draw", min_target=8, max_target=14, soft_penalty=0.9, hard_penalty=2.1, hard_buffer=2),
            TagPenalty(tag="removal", min_target=8, max_target=14, soft_penalty=0.9, hard_penalty=2.1, hard_buffer=2),
            TagPenalty(tag="interaction", min_target=6, max_target=12, soft_penalty=0.9, hard_penalty=2.1, hard_buffer=2),
            TagPenalty(tag="boardwipe", min_target=2, max_target=4, soft_penalty=1.1, hard_penalty=2.6, hard_buffer=1),
        )
    )

    # Tags that should not define "strategy"
    non_strategy_tags: Tuple[str, ...] = (
        "ramp",
        "removal",
        "card_draw",
        "control",
        "interaction",
        "boardwipe",
    )

@dataclass(frozen=True)
class DeckTrainConfig:
    """Training hyperparameters for the deck generation model."""

    seed: int = 7
    epochs: int = 5
    steps_per_epoch: int = 3000

    lr: float = 3e-4
    weight_decay: float = 1e-4

    hidden_dim: int = 256
    node_dim: int = 256
    state_dim: int = 256
    gnn_layers: int = 3

    # sequence training
    max_prefix_len: int = 48
    num_negatives: int = 2048
    temperature: float = 1.0

    dropout: float = 0.12

    combo_loss_weight: float = 2.0

    # logging
    log_every: int = 0

    # output extras
    save_node_embeddings: bool = True
