"""Build graph node/edge tensors from deck, combo, and tag datasets."""

import json
import math
import heapq
from dataclasses import dataclass
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, List, Tuple, Any

import torch

from vector_database import VectorDatabase
from card_data import CardFields

from config import CONFIG

@dataclass(frozen=True)
class Config:
    """Runtime configuration for graph construction inputs and outputs."""

    decks_path: str = CONFIG.datasets["DECKS_DATASET_PATH"]
    variants_path: str = CONFIG.datasets["COMBO_DATASET_PATH"]
    tags_path: str = CONFIG.datasets["TAGS_DATASET_PATH"]
    commander_cards_path: str = CONFIG.datasets["CARDS_DATASET_PATH"]
    vector_path: str = CONFIG.datasets["VECTOR_DATABASE_PATH"]
    out_nodes_pt: str = CONFIG.datasets["GRAPH_NODES_DATA_PATH"]
    out_edges_pt: str = CONFIG.datasets["GRAPH_EDGES_DATA_PATH"]

    min_deck_occurrences: int = 1

    min_common_tags: int = 2
    max_cards_per_tag: int = 2000

    topk_per_node: int = 300


EDGE_FEATURES = [
    "cooccur",
    "combo",
    "tag_shared",
    "tag_synergy",
    "commander_cond",
]


def _safe_int(x: Any, default: int = 10**9) -> int:
    """Best-effort integer conversion with a fallback default."""

    try:
        if x is None:
            return default
        if isinstance(x, bool):
            return default
        if isinstance(x, (int, float)):
            return int(x)
        s = str(x).strip()
        if s == "" or s.lower() == "none":
            return default
        return int(float(s))
    except (TypeError, ValueError, OverflowError):
        return default


def load_commander_cards(path: str) -> Dict[str, Dict[str, Any]]:
    """Load commander card metadata keyed by card name from JSON."""

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[warn] {path} not found. Node metadata will be sparse.")
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            name = row.get("card_name")
            if not isinstance(name, str) or not name:
                continue
            out[name] = row
    elif isinstance(data, dict):
        for name, row in data.items():
            if isinstance(row, dict):
                out[str(name)] = row
    return out


def co_occurance_graph(deck_card_lists: List[List[str]], min_deck_occurrences: int) -> Dict[str, Dict[str, int]]:
    """Build a weighted undirected co-occurrence graph from deck lists."""

    cooccur_counter = Counter()

    for deck in deck_card_lists:
        counts = Counter(deck)

        unique = sorted(counts.keys())
        for i, a in enumerate(unique):
            ca = counts[a]
            for j in range(i + 1, len(unique)):
                b = unique[j]
                cb = counts[b]
                cooccur_counter[(a, b)] += ca * cb

    graph: Dict[str, Dict[str, int]] = defaultdict(dict)
    for (card_a, card_b), count in cooccur_counter.items():
        if count >= min_deck_occurrences:
            graph[card_a][card_b] = count
            graph[card_b][card_a] = count

    return graph


def commander_co_occurance(decks_obj: Dict[str, Any], min_deck_occurrences: int) -> Dict[str, Dict[str, int]]:
    """Map each commander to cards that frequently appear in its decks."""

    graph = defaultdict(dict)
    commander_deck_map = defaultdict(list)

    for _, deck in decks_obj.items():
        commander = deck.get("commanders", [None])[0]
        cards = set(deck.get("cards", []))
        if commander:
            commander_deck_map[commander].append(cards)

    for commander, decks_for_commander in commander_deck_map.items():
        counts = Counter()
        for card_set in decks_for_commander:
            for card in card_set:
                counts[card] += 1

        for card, count in counts.items():
            if count >= min_deck_occurrences:
                graph[commander][card] = count

    return graph


def combo_graph(variants: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """Build a combo co-usage graph from combo variant records."""

    graph = defaultdict(dict)

    for combo in variants.get("results", []):
        legalities = combo.get("legalities", combo.get("legality", {}))
        if isinstance(legalities, dict) and legalities.get("commander") is not True:
            continue

        uses = combo.get("uses", [])
        if not isinstance(uses, list) or len(uses) < 2:
            continue

        names = []
        for u in uses:
            if not isinstance(u, dict):
                continue
            card = u.get("card", None)
            if isinstance(card, dict) and isinstance(card.get("name"), str):
                names.append(card["name"])
            elif isinstance(u.get("name"), str):
                names.append(u["name"])

        names = sorted(set(n for n in names if n))
        if len(names) < 2:
            continue

        for a, b in combinations(names, 2):
            graph[a][b] = graph[a].get(b, 0) + 1
            graph[b][a] = graph[b].get(a, 0) + 1

    return graph


def tag_graph(tag_data: Dict[str, Dict], min_common: int = 2, max_cards_per_tag: int = 3000) -> Dict[str, Dict[str, int]]:
    """Build graph edges for cards that share the same tags."""

    tag_to_cards = defaultdict(list)
    for card, payload in tag_data.items():
        for tag in payload.get("tags", []):
            tag_to_cards[tag].append(card)

    pair_counts = defaultdict(int)
    for tag, cards in tag_to_cards.items():
        if max_cards_per_tag and len(cards) > max_cards_per_tag:
            continue
        cards = sorted(set(cards))
        for a, b in combinations(cards, 2):
            pair_counts[(a, b)] += 1

    graph = defaultdict(dict)
    for (a, b), c in pair_counts.items():
        if c >= min_common:
            graph[a][b] = c
            graph[b][a] = c

    return graph


def synergy_tag_graph(tag_data: Dict[str, Dict], min_common: int = 1, max_cards_per_tag: int = 3000) -> Dict[str, Dict[str, int]]:  # pylint: disable=too-many-branches
    """Build directed edges from tags to cards that match synergistic tags."""

    tag_to_cards = defaultdict(list)
    for card, payload in tag_data.items():
        for tag in payload.get("tags", []):
            tag_to_cards[tag].append(card)
    for t in list(tag_to_cards.keys()):
        tag_to_cards[t] = list(dict.fromkeys(tag_to_cards[t]))

    graph = defaultdict(dict)

    for card_a, payload in tag_data.items():
        tags_a = set(payload.get("tags", []))
        if not tags_a:
            continue

        counts = defaultdict(int)

        for t in tags_a:
            syn_tags = CardFields.tag_synergy_map().get(t)
            if not syn_tags:
                continue

            for syn in syn_tags:
                cards_b = tag_to_cards.get(syn, [])
                if max_cards_per_tag and len(cards_b) > max_cards_per_tag:
                    continue

                for card_b in cards_b:
                    if card_b == card_a:
                        continue
                    counts[card_b] += 1

        for card_b, c in counts.items():
            if c >= min_common:
                graph[card_a][card_b] = c

    return graph


def build_edge_tensors(node_to_idx: Dict[str, int], cooccur_graph: Dict[str, Dict[str, int]], combo_graph_data: Dict[str, Dict[str, int]], tag_graph_data: Dict[str, Dict[str, int]], synergy_tag_data: Dict[str, Dict[str, int]], commander_graph: Dict[str, Dict[str, int]], topk_per_node: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge edge sources and features into `edge_index` and `edge_attr` tensors."""

    src_list: List[int] = []
    dst_list: List[int] = []
    feat_list: List[List[float]] = []

    get_idx = node_to_idx.get

    for card_a, ia in node_to_idx.items():
        co_n = cooccur_graph.get(card_a, {})
        cb_n = combo_graph_data.get(card_a, {})
        tg_n = tag_graph_data.get(card_a, {})
        sg_n = synergy_tag_data.get(card_a, {})
        cm_n = commander_graph.get(card_a, {})

        neighbors = set(co_n) | set(cb_n) | set(tg_n) | set(sg_n) | set(cm_n)
        neighbors.discard(card_a)

        candidates = []
        for card_b in neighbors:
            ib = get_idx(card_b)
            if ib is None:
                continue

            f0 = float(co_n.get(card_b, 0.0))
            f1 = float(cb_n.get(card_b, 0.0))
            f2 = float(tg_n.get(card_b, 0.0))
            f3 = float(sg_n.get(card_b, 0.0))
            f4 = float(cm_n.get(card_b, 0.0))

            if f0 == 0.0 and f1 == 0.0 and f2 == 0.0 and f3 == 0.0 and f4 == 0.0:
                continue

            if f1 > 0.0:
                f1 = 1.0 + math.log1p(f1)

            score = f0 + (3.0 * f1) + f2 + f3 + f4
            candidates.append((score, ib, [f0, f1, f2, f3, f4]))

        if not candidates:
            continue

        if topk_per_node and len(candidates) > topk_per_node:
            candidates = heapq.nlargest(topk_per_node, candidates, key=lambda x: x[0])

        for _, ib, feats in candidates:
            src_list.append(ia)
            dst_list.append(ib)
            feat_list.append(feats)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(feat_list, dtype=torch.float32)
    return edge_index, edge_attr


def main():
    """Run the graph build pipeline and persist node/edge tensor artifacts."""

    cfg = Config()

    with open(cfg.decks_path, "r", encoding="utf-8") as f:
        decks_obj = json.load(f)

    with open(cfg.variants_path, "r", encoding="utf-8") as f:
        variants_obj = json.load(f)

    with open(cfg.tags_path, "r", encoding="utf-8") as f:
        tag_data = json.load(f)

    commander_cards = load_commander_cards(cfg.commander_cards_path)

    print("[info] Data loaded.")

    vector_data = VectorDatabase.load_static(cfg.vector_path)
    node_names = list(vector_data.keys())
    node_to_idx = {n: i for i, n in enumerate(node_names)}
    print(f"[info] Vector data loaded. Nodes: {len(node_names)}")

    ranks: List[int] = []
    cmc: List[int] = []
    types: List[List[str]] = []
    subtypes: List[List[str]] = []
    colors: List[List[str]] = []

    for name in node_names:
        row = commander_cards.get(name, {})
        ranks.append(_safe_int(row.get("rank", None), default=10**9))
        cmc.append(_safe_int(row.get("mana_cost", 0), default=0))

        t = row.get("card_types", [])
        st = row.get("card_subtypes", [])
        ci = row.get("color_identity", [])

        types.append(list(t) if isinstance(t, list) else [])
        subtypes.append(list(st) if isinstance(st, list) else [])
        colors.append(list(ci) if isinstance(ci, list) else [])

    finite_ranks = [r for r in ranks if r < 10**8]
    rank_min = min(finite_ranks) if finite_ranks else 1
    rank_max = max(finite_ranks) if finite_ranks else 10**6

    node_meta = {
        "rank": ranks,
        "mana_cost": cmc,
        "card_types": types,
        "card_subtypes": subtypes,
        "color_identity": colors,
        "rank_min": int(rank_min),
        "rank_max": int(rank_max),
    }

    deck_card_lists: List[List[str]] = []
    for _, deck in decks_obj.items():
        cards = deck.get("cards", [])
        if isinstance(cards, list) and cards:
            deck_card_lists.append(cards)

    commander_graph = commander_co_occurance(decks_obj, cfg.min_deck_occurrences)

    cooccur_graph = co_occurance_graph(deck_card_lists, cfg.min_deck_occurrences)

    combo_graph_data = combo_graph(variants_obj)

    tag_graph_data = tag_graph(
        tag_data,
        min_common=cfg.min_common_tags,
        max_cards_per_tag=cfg.max_cards_per_tag,
    )

    synergy_tag_data = synergy_tag_graph(
        tag_data,
        min_common=cfg.min_common_tags,
        max_cards_per_tag=cfg.max_cards_per_tag,
    )

    edge_index, edge_attr = build_edge_tensors(
        node_to_idx=node_to_idx,
        cooccur_graph=cooccur_graph,
        combo_graph_data=combo_graph_data,
        tag_graph_data=tag_graph_data,
        synergy_tag_data=synergy_tag_data,
        commander_graph=commander_graph,
        topk_per_node=cfg.topk_per_node,
    )
    
    torch.save(
        {"node_names": node_names, "node_to_idx": node_to_idx, "node_meta": node_meta},
        cfg.out_nodes_pt,
    )
    torch.save(
        {"edge_index": edge_index, "edge_attr": edge_attr, "edge_feat_names": EDGE_FEATURES},
        cfg.out_edges_pt,
    )

    print(f"Wrote nodes: {cfg.out_nodes_pt}")
    print(f"Wrote edges: {cfg.out_edges_pt}")


if __name__ == "__main__":
    main()
