import json
from typing import Dict, List, Optional

import torch
from torch_geometric.data import Data

from Vector_Database import VectorDatabase
from card_data import CardDecoder, SimpleDeck, SimpleDeckAnalyzer

from .config import DeckGenPaths, GenConfig

decoder = CardDecoder()


class DeckGenAssets:
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


def load_assets(paths: DeckGenPaths, device: torch.device, gen: GenConfig, vector_db: Optional[VectorDatabase] = None) -> DeckGenAssets:
    nodes = torch.load(paths.nodes_pt, map_location="cpu", weights_only=False)
    edges = torch.load(paths.edges_pt, map_location="cpu", weights_only=False)

    node_names: List[str] = nodes["node_names"]
    node_to_index: Dict[str, int] = nodes["node_to_idx"]

    edge_index = edges["edge_index"].long().to(device)
    edge_attr = edges["edge_attr"].float().to(device)

    vd = vector_db if vector_db is not None else VectorDatabase.load_static(paths.vectors_pt)

    first_vec = vd.get_vector(node_names[0])
    if first_vec is None:
        raise RuntimeError(f"{paths.vectors_pt} does not contain vectors for node '{node_names[0]}'")

    dim = int(first_vec.numel()) if isinstance(first_vec, torch.Tensor) else int(len(first_vec))
    x = torch.zeros((len(node_names), dim), dtype=torch.float32)

    for i, n in enumerate(node_names):
        v = vd.get_vector(n)
        x[i] = v.float().view(-1)

    graph = Data(x=x.to(device), edge_index=edge_index, edge_attr=edge_attr)

    is_land = decoder.land_mask_from_vectors(graph.x).to(device)
    color_mask = decoder.color_identity_mask_from_vectors(graph.x).to(device)
    mana_value = decoder.mana_value_from_vectors(graph.x).to(device)

    with open(paths.tags_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    tag_map: Dict[str, List[str]] = {k: list(v["tags"]) for k, v in raw.items()}

    decks = SimpleDeck.load_json_file(paths.decks_json)

    by_commander_stats: Dict[str, List[dict]] = {}
    global_stats: List[dict] = []
    commander_indices: List[int] = []

    skipped_no_commander = 0
    skipped_missing_commander = 0

    for d in decks:
        if not d.commanders:
            skipped_no_commander += 1
            continue

        cmd = d.commanders[0]
        cmd_idx = node_to_index.get(cmd)
        if cmd_idx is None:
            skipped_missing_commander += 1
            continue

        analyzer = SimpleDeckAnalyzer(d, tag_map, vd)
        stats = analyzer.analyze()

        global_stats.append(stats)
        by_commander_stats.setdefault(cmd, []).append(stats)
        commander_indices.append(int(cmd_idx))

    seen = set()
    dedup_cmd: List[int] = []
    for idx in commander_indices:
        if idx in seen:
            continue
        seen.add(idx)
        dedup_cmd.append(idx)

    neighbors: List[torch.Tensor] = [torch.empty(0, dtype=torch.long) for _ in range(len(node_names))]
    src_cpu = edge_index[0].detach().cpu()
    dst_cpu = edge_index[1].detach().cpu()
    strength_cpu = edge_attr.detach().cpu().sum(dim=1)

    order = torch.argsort(src_cpu, stable=True)
    src_cpu = src_cpu[order]
    dst_cpu = dst_cpu[order]
    strength_cpu = strength_cpu[order]

    unique_src, counts = torch.unique_consecutive(src_cpu, return_counts=True)
    start = 0
    for s, c in zip(unique_src.tolist(), counts.tolist()):
        end = start + c
        dst_seg = dst_cpu[start:end]
        w_seg = strength_cpu[start:end]
        k = min(int(gen.neighbor_k), int(dst_seg.numel()))
        if k > 0:
            top = torch.topk(w_seg, k=k, largest=True).indices
            neighbors[s] = dst_seg[top].clone()
        start = end

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
