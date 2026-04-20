"""Pre-loaded graph tensors, metadata, and deck stats for deck generation."""
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from torch_geometric.data import Data

from backend.card_data import CardDecoder, CardFields, SimpleDeck
from backend.vector_database import VectorDatabase

from .config import DeckGenPaths, GenConfig

decoder = CardDecoder()


class DeckGenAssets:
    """Container for pre-computed graph tensors and generation metadata."""

    def __init__(self, node_names: List[str], node_to_index: Dict[str, int], graph: Data, is_land_node: torch.Tensor, color_identity_mask: torch.Tensor, mana_value_by_node: torch.Tensor, tag_map: Dict[str, List[str]], neighbors_by_node: List[torch.Tensor], commander_indices: List[int], by_commander_stats: Dict[str, List[dict]], global_stats: List[dict]) -> None:
        """Store pre-computed graph tensors and generation metadata.

        Args:
            node_names: Ordered list of card names corresponding to graph nodes.
            node_to_index: Mapping from card name to node index.
            graph: PyG Data object containing node features and edge data.
            is_land_node: Boolean tensor marking land-typed nodes.
            color_identity_mask: Boolean tensor of shape (N, num_colors) for color identity.
            mana_value_by_node: Float tensor of mana values for each node.
            tag_map: Mapping from card name to list of gameplay tags.
            neighbors_by_node: Top-k neighbor indices for each node.
            commander_indices: Deduplicated list of commander node indices.
            by_commander_stats: Per-commander aggregated deck analyzer stats.
            global_stats: All deck analyzer stats pooled globally.

        Returns:
            None
        """
        self.node_names = node_names
        self.node_to_index = node_to_index
        self.graph = graph
        self.is_land_node = is_land_node
        self.color_identity_mask = color_identity_mask
        self.mana_value_by_node = mana_value_by_node
        self.tag_map = tag_map
        self.neighbors_by_node = neighbors_by_node
        self.commander_indices = commander_indices
        self.by_commander_stats = by_commander_stats
        self.global_stats = global_stats


def build_node_features(vd: VectorDatabase, node_names: List[str]) -> torch.Tensor:
    """Build the node feature matrix from the vector database.

    Args:
        vd: Loaded VectorDatabase instance.
        node_names: Ordered list of card names to build features for.

    Returns:
        Float32 tensor of shape (N, vector_dim) with one row per card.
    """
    first_vec = vd.get_vector(node_names[0])
    if first_vec is None:
        raise RuntimeError("Vector database does not contain the first graph node.")

    dim = int(first_vec.numel()) if isinstance(first_vec, torch.Tensor) else int(len(first_vec))
    x = torch.zeros((len(node_names), dim), dtype=torch.float32)
    for i, name in enumerate(node_names):
        vector = vd.get_vector(name)
        x[i] = vector.float().view(-1)
    return x


def collect_deck_stats(decks: List[SimpleDeck], node_to_index: Dict[str, int], tag_map: Dict[str, List[str]], vd: VectorDatabase, is_land_node: Optional[torch.Tensor] = None, mana_value_by_node: Optional[torch.Tensor] = None) -> tuple[Dict[str, List[dict]], List[dict], List[int]]:  # pylint: disable=unused-argument
    """Aggregate deck stats globally and per commander using O(1) graph lookups.

    Args:
        decks: List of SimpleDeck objects from the training dataset.
        node_to_index: Mapping from card name to graph node index.
        tag_map: Mapping from card name to list of gameplay tags.
        vd: Loaded VectorDatabase instance (unused, kept for API compatibility).
        is_land_node: Boolean tensor of shape (N,) marking land nodes.
        mana_value_by_node: Float tensor of shape (N,) with mana values.

    Returns:
        A tuple of (by_commander_stats, global_stats, dedup_commander_indices).
    """
    basic_set_lower = {str(x).strip().lower() for x in CardFields.basic_lands()}

    # Transfer tensors to CPU numpy once — avoids 1M GPU-CPU syncs inside the loop.
    is_land_arr = is_land_node.cpu().numpy() if is_land_node is not None else None
    mana_arr = mana_value_by_node.cpu().numpy() if mana_value_by_node is not None else None

    by_commander_stats: Dict[str, List[dict]] = {}
    global_stats: List[dict] = []
    commander_indices: List[int] = []

    for deck in decks:
        if not deck.commanders:
            continue

        commander = deck.commanders[0]
        commander_idx = node_to_index.get(commander)
        if commander_idx is None:
            continue

        tag_counts: Dict[str, int] = defaultdict(int)
        land_count = 0
        basic_count = 0
        curve_counts = [0] * 7

        for name in list(deck.commanders) + list(deck.cards):
            idx = node_to_index.get(name)
            if idx is None:
                continue

            for t in tag_map.get(name, []):
                tag_counts[t] += 1

            if is_land_arr is not None and is_land_arr[idx]:
                land_count += 1
                if name.strip().lower() in basic_set_lower:
                    basic_count += 1

            if mana_arr is not None:
                mv = int(round(float(mana_arr[idx])))
                curve_counts[min(6, max(0, mv))] += 1

        basic_ratio = basic_count / land_count if land_count > 0 else 0.0

        stats = {
            "tags": {"tag_counts": dict(tag_counts)},
            "curve": {"mana_curve": {"counts": curve_counts, "percent": []}},
            "lands": {"lands": {
                "land_count": land_count,
                "basic_count": basic_count,
                "basic_ratio": basic_ratio,
            }},
        }

        global_stats.append(stats)
        by_commander_stats.setdefault(commander, []).append(stats)
        commander_indices.append(int(commander_idx))

    seen: set[int] = set()
    dedup_commander_indices: List[int] = []
    for idx in commander_indices:
        if idx in seen:
            continue
        seen.add(idx)
        dedup_commander_indices.append(idx)

    return by_commander_stats, global_stats, dedup_commander_indices


def build_neighbor_index(edge_index: torch.Tensor, edge_attr: torch.Tensor, node_count: int, neighbor_k: int) -> List[torch.Tensor]:
    """Build top-k outgoing neighbor index for each node.

    Args:
        edge_index: Long tensor of shape (2, E) with source and destination indices.
        edge_attr: Float tensor of shape (E, edge_dim) with edge attributes.
        node_count: Total number of nodes in the graph.
        neighbor_k: Maximum number of neighbors to retain per node.

    Returns:
        List of length node_count where each element is a long tensor of neighbor indices.
    """
    neighbors: List[torch.Tensor] = [torch.empty(0, dtype=torch.long) for _ in range(node_count)]
    src_cpu = edge_index[0].detach().cpu()
    dst_cpu = edge_index[1].detach().cpu()
    strength_cpu = edge_attr.detach().cpu().sum(dim=1)

    order = torch.argsort(src_cpu, stable=True)
    src_cpu = src_cpu[order]
    dst_cpu = dst_cpu[order]
    strength_cpu = strength_cpu[order]

    unique_src, counts = torch.unique_consecutive(src_cpu, return_counts=True)
    start = 0
    for source_idx, count in zip(unique_src.tolist(), counts.tolist()):
        end = start + count
        dst_seg = dst_cpu[start:end]
        w_seg = strength_cpu[start:end]
        k = min(int(neighbor_k), int(dst_seg.numel()))
        if k > 0:
            top = torch.topk(w_seg, k=k, largest=True).indices
            neighbors[source_idx] = dst_seg[top].clone()
        start = end

    return neighbors


def load_assets(paths: DeckGenPaths, device: torch.device, gen: GenConfig, vector_db: Optional[VectorDatabase] = None) -> DeckGenAssets:
    """Load serialized graph artifacts and assemble a DeckGenAssets instance.

    Args:
        paths: DeckGenPaths dataclass pointing to all required data files.
        device: Torch device to place tensors on.
        gen: Generation config providing neighbor_k and other parameters.
        vector_db: Optional pre-loaded VectorDatabase; loaded from disk if None.

    Returns:
        A fully initialized DeckGenAssets ready for deck generation.
    """
    nodes = torch.load(paths.nodes_pt, map_location="cpu", weights_only=False)
    edges = torch.load(paths.edges_pt, map_location="cpu", weights_only=False)

    node_names: List[str] = nodes["node_names"]
    node_to_index: Dict[str, int] = nodes["node_to_idx"]

    edge_index = edges["edge_index"].long().to(device)
    edge_attr = edges["edge_attr"].float().to(device)

    vd = vector_db if vector_db is not None else VectorDatabase.load_static(paths.vectors_pt)

    if os.path.isfile(paths.node_features_pt):
        x = torch.load(paths.node_features_pt, map_location="cpu", weights_only=True)
    else:
        x = build_node_features(vd, node_names)
        torch.save(x, paths.node_features_pt)

    graph = Data(x=x.to(device), edge_index=edge_index, edge_attr=edge_attr)

    is_land = decoder.land_mask_from_vectors(graph.x).to(device)
    color_mask = decoder.color_identity_mask_from_vectors(graph.x).to(device)
    mana_value = decoder.mana_value_from_vectors(graph.x).to(device)

    with open(paths.tags_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    tag_map: Dict[str, List[str]] = {k: list(v["tags"]) for k, v in raw.items()}

    decks = SimpleDeck.load_json_file(paths.decks_json)

    by_commander_stats, global_stats, dedup_cmd = collect_deck_stats(
        decks=decks,
        node_to_index=node_to_index,
        tag_map=tag_map,
        vd=vd,
        is_land_node=is_land,
        mana_value_by_node=mana_value,
    )

    neighbors = build_neighbor_index(
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_count=len(node_names),
        neighbor_k=int(gen.neighbor_k),
    )

    return DeckGenAssets(
        node_names=node_names,
        node_to_index=node_to_index,
        graph=graph,
        is_land_node=is_land,
        color_identity_mask=color_mask,
        mana_value_by_node=mana_value,
        tag_map=tag_map,
        neighbors_by_node=neighbors,
        commander_indices=dedup_cmd,
        by_commander_stats=by_commander_stats,
        global_stats=global_stats,
    )
