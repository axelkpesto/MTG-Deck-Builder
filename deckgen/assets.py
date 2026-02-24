"""Load and assemble graph/model assets for deck generation."""

import json
from typing import Dict, List, Optional

import torch
from torch_geometric.data import Data

from vector_database import VectorDatabase
from card_data import CardDecoder, SimpleDeck, SimpleDeckAnalyzer

from .config import DeckGenPaths, GenConfig

decoder = CardDecoder()


class DeckGenAssets:
    """In-memory container for graph tensors and generation metadata."""

    def __init__(self, node_names: List[str], node_to_index: Dict[str, int], graph: Data, is_land_node: torch.Tensor, color_identity_mask: torch.Tensor, mana_value_by_node: torch.Tensor, tag_map: Dict[str, List[str]], neighbors_by_node: List[torch.Tensor], commander_indices: List[int], by_commander_stats: Dict[str, List[dict]], global_stats: List[dict]) -> None:
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
    """Build the node feature matrix from the vector database."""
    first_vec = vd.get_vector(node_names[0])
    if first_vec is None:
        raise RuntimeError("Vector database does not contain the first graph node.")

    dim = int(first_vec.numel()) if isinstance(first_vec, torch.Tensor) else int(len(first_vec))
    x = torch.zeros((len(node_names), dim), dtype=torch.float32)
    for i, name in enumerate(node_names):
        vector = vd.get_vector(name)
        x[i] = vector.float().view(-1)
    return x


def collect_deck_stats(decks: List[SimpleDeck], node_to_index: Dict[str, int], tag_map: Dict[str, List[str]], vd: VectorDatabase) -> tuple[Dict[str, List[dict]], List[dict], List[int]]:
    """Aggregate analyzer stats globally and by commander."""
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

        analyzer = SimpleDeckAnalyzer(deck, tag_map, vd)
        stats = analyzer.analyze()

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
    """Build top-k outgoing neighbor indices for each node."""
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
    """Load serialized graph artifacts and derived generation assets."""
    nodes = torch.load(paths.nodes_pt, map_location="cpu", weights_only=False)
    edges = torch.load(paths.edges_pt, map_location="cpu", weights_only=False)

    node_names: List[str] = nodes["node_names"]
    node_to_index: Dict[str, int] = nodes["node_to_idx"]

    edge_index = edges["edge_index"].long().to(device)
    edge_attr = edges["edge_attr"].float().to(device)

    vd = vector_db if vector_db is not None else VectorDatabase.load_static(paths.vectors_pt)

    x = build_node_features(vd, node_names)

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
