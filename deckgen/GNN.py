import os
import random
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from Vector_Database import VectorDatabase
from card_data import CardDecoder, SimpleDeck, CardFields
from deckgen.config import DeckGenPaths, DeckTrainConfig
from deckgen.model import CommanderDeckGNN
from deckgen.utils import mana_value_bucket, set_seed

card_decoder = CardDecoder()

def load_graph(paths: DeckGenPaths, device: torch.device) -> Tuple[List[str], Dict[str, int], torch.Tensor, torch.Tensor]:
    nodes_blob = torch.load(paths.nodes_pt, map_location="cpu", weights_only=False)
    edges_blob = torch.load(paths.edges_pt, map_location="cpu", weights_only=False)

    node_names: List[str] = nodes_blob["node_names"]
    node_to_index: Dict[str, int] = nodes_blob["node_to_idx"]

    edge_index = edges_blob["edge_index"].long().to(device)
    edge_attr = edges_blob["edge_attr"].float().to(device)

    return node_names, node_to_index, edge_index, edge_attr


def build_node_feature_matrix(node_names: List[str], paths: DeckGenPaths, device: torch.device) -> torch.Tensor:
    vector_db = VectorDatabase.load_static(paths.vectors_pt)

    reference_vec = None
    for name in node_names:
        if name in vector_db:
            reference_vec = vector_db[name]
            break
    if reference_vec is None:
        raise RuntimeError("vector_data.pt appears empty or does not match graph_nodes.pt")

    if isinstance(reference_vec, torch.Tensor):
        vector_dim = int(reference_vec.numel())
    else:
        vector_dim = int(np.asarray(reference_vec).reshape(-1).shape[0])

    feature_matrix = torch.zeros((len(node_names), vector_dim), dtype=torch.float32)

    missing = 0
    for i, name in enumerate(node_names):
        vec = vector_db.get(name)
        if vec is None:
            missing += 1
            continue
        if not isinstance(vec, torch.Tensor):
            vec = torch.tensor(np.asarray(vec, dtype=np.float32))
        feature_matrix[i] = vec.detach().cpu().float().view(-1)

    if missing:
        print(f"Missing vectors for {missing}/{len(node_names)} nodes; filled with zeros.")

    return feature_matrix.to(device)


def load_simple_decks(decks_json_path: str) -> List[SimpleDeck]:
    return SimpleDeck.load_json_file(decks_json_path)


def deck_to_training_sequence(deck: SimpleDeck, node_to_index: Dict[str, int]) -> Optional[Tuple[int, List[int]]]:
    if not deck.commanders:
        return None

    commander_name = deck.commanders[0]
    commander_index = node_to_index.get(commander_name)
    if commander_index is None:
        return None

    seen_nonbasic: Set[str] = set()
    card_indices: List[int] = []

    for card_name in deck.cards:
        idx = node_to_index.get(card_name)
        if idx is None:
            continue

        if CardFields.is_basic_land(card_name):
            card_indices.append(idx)
            continue

        if card_name in seen_nonbasic:
            continue
        seen_nonbasic.add(card_name)
        card_indices.append(idx)

    card_indices = [i for i in card_indices if i != commander_index]
    if len(card_indices) < 2:
        return None

    return commander_index, card_indices

def build_combo_edge_set(edge_index: torch.Tensor, edge_attr: torch.Tensor, combo_col: int = 1) -> Set[Tuple[int, int]]:
    edge_index_cpu = edge_index.detach().cpu()
    edge_attr_cpu = edge_attr.detach().cpu()

    src_list = edge_index_cpu[0].tolist()
    dst_list = edge_index_cpu[1].tolist()
    combo_strength = edge_attr_cpu[:, combo_col].tolist()

    combo_edges: Set[Tuple[int, int]] = set()
    for src, dst, strength in zip(src_list, dst_list, combo_strength):
        if float(strength) > 0.0:
            combo_edges.add((int(src), int(dst)))
    return combo_edges


def precompute_negative_buckets(*, num_nodes: int, mana_value_cpu: np.ndarray, is_land_cpu: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[int, int], np.ndarray]]:
    mv_bucket_by_node = np.zeros((num_nodes,), dtype=np.int8)
    for i in range(num_nodes):
        mv_bucket_by_node[i] = np.int8(mana_value_bucket(float(mana_value_cpu[i])))

    type_bucket_by_node = np.where(is_land_cpu.astype(bool), 0, 1).astype(np.int8)

    buckets: Dict[Tuple[int, int], List[int]] = {(mv, t): [] for mv in range(7) for t in range(2)}
    for i in range(num_nodes):
        buckets[(int(mv_bucket_by_node[i]), int(type_bucket_by_node[i]))].append(i)

    bucket_to_indices: Dict[Tuple[int, int], np.ndarray] = {}
    for key, arr in buckets.items():
        bucket_to_indices[key] = np.asarray(arr, dtype=np.int32)

    return mv_bucket_by_node, type_bucket_by_node, bucket_to_indices


def sample_hard_negative_candidates(*, num_nodes: int, true_index: int, banned_indices: Set[int], num_negatives: int, device: torch.device, mv_bucket_by_node: np.ndarray, type_bucket_by_node: np.ndarray, bucket_to_indices: Dict[Tuple[int, int], np.ndarray]) -> torch.Tensor:
    true_mv_bucket = int(mv_bucket_by_node[true_index])
    true_type_bucket = int(type_bucket_by_node[true_index])
    same_bucket = bucket_to_indices.get((true_mv_bucket, true_type_bucket))

    negatives: List[int] = []

    if same_bucket is not None and same_bucket.size > 0:
        need = num_negatives
        replace = same_bucket.size < need

        for _ in range(8):
            if need <= 0:
                break
            sampled = np.random.choice(same_bucket, size=need, replace=replace)
            for idx in sampled.tolist():
                idx = int(idx)
                if idx == true_index or idx in banned_indices:
                    continue
                negatives.append(idx)
                if len(negatives) >= num_negatives:
                    break
            need = num_negatives - len(negatives)

    tries = 0
    while len(negatives) < num_negatives and tries < num_negatives * 100:
        idx = int(np.random.randint(0, num_nodes))
        tries += 1
        if idx == true_index or idx in banned_indices:
            continue
        negatives.append(idx)

    if len(negatives) < num_negatives:
        padding = np.random.randint(0, num_nodes, size=(num_negatives - len(negatives),), dtype=np.int32).tolist()
        negatives.extend(int(x) for x in padding)

    candidate_array = np.empty((1 + num_negatives,), dtype=np.int32)
    candidate_array[0] = int(true_index)
    candidate_array[1:] = np.asarray(negatives[:num_negatives], dtype=np.int32)
    return torch.tensor(candidate_array, dtype=torch.long, device=device)


def train_one_epoch(*, model: CommanderDeckGNN, graph: Data, training_examples: List[Tuple[int, List[int]]], cfg: DeckTrainConfig, optimizer: torch.optim.Optimizer, scaler: Optional[torch.amp.GradScaler], combo_edges: Set[Tuple[int, int]], mv_bucket_by_node: np.ndarray, type_bucket_by_node: np.ndarray, bucket_to_indices: Dict[Tuple[int, int], np.ndarray]) -> float:
    model.train()
    num_nodes = int(graph.x.size(0))

    total_loss = 0.0
    steps = 0

    autocast_enabled = (graph.x.device.type == "cuda")

    for _ in range(cfg.steps_per_epoch):
        commander_index, card_sequence = random.choice(training_examples)
        if len(card_sequence) < 2:
            continue

        max_t = min(cfg.max_prefix_len, len(card_sequence) - 1)
        prefix_len = random.randint(1, max_t)

        prefix_indices = card_sequence[:prefix_len]
        target_index = card_sequence[prefix_len]
        previous_index = prefix_indices[-1] if prefix_indices else commander_index

        banned = set(prefix_indices)
        banned.add(commander_index)

        prefix_tensor = (
            torch.tensor(prefix_indices, dtype=torch.long, device=graph.x.device)
            if prefix_indices
            else torch.empty((0,), dtype=torch.long, device=graph.x.device)
        )

        candidate_indices = sample_hard_negative_candidates(
            num_nodes=num_nodes,
            true_index=target_index,
            banned_indices=banned,
            num_negatives=cfg.num_negatives,
            device=graph.x.device,
            mv_bucket_by_node=mv_bucket_by_node,
            type_bucket_by_node=type_bucket_by_node,
            bucket_to_indices=bucket_to_indices,
        )

        labels = torch.zeros((1,), dtype=torch.long, device=graph.x.device)

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=autocast_enabled):
            node_embeddings = model.encode(graph.x, graph.edge_index, graph.edge_attr)

            commander_vec = node_embeddings[commander_index]
            pool_vec = (
                node_embeddings[prefix_tensor].mean(dim=0)
                if prefix_tensor.numel() > 0
                else torch.zeros_like(commander_vec)
            )
            state = model.policy.make_state(commander_vec, pool_vec)

            logits = model.policy.score_candidates(node_embeddings, state, candidate_indices) / max(1e-6, cfg.temperature)
            loss = F.cross_entropy(logits.unsqueeze(0), labels)

            if (previous_index, target_index) in combo_edges:
                loss = loss * float(cfg.combo_loss_weight)

        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += float(loss.item())
        steps += 1

        if cfg.log_every and steps % int(cfg.log_every) == 0:
            print(f"[train] step={steps} loss={float(loss.item()):.4f}")

    return total_loss / max(1, steps)


@torch.inference_mode()
def compute_node_embeddings(model: CommanderDeckGNN, graph: Data) -> torch.Tensor:
    model.eval()
    return model.encode(graph.x, graph.edge_index, graph.edge_attr)


def save_checkpoint(*, path: str, model: CommanderDeckGNN, train_cfg: DeckTrainConfig, node_embeddings: Optional[torch.Tensor]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    payload: Dict[str, Any] = {
        "format_version": 2,
        "model_state_dict": model.state_dict(),
        "train_cfg": train_cfg.__dict__,
    }
    
    if node_embeddings is not None:
        payload["node_embeddings"] = node_embeddings.detach().cpu()

    torch.save(payload, path)


def main() -> None:
    paths = DeckGenPaths()
    cfg = DeckTrainConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    set_seed(cfg.seed)

    node_names, node_to_index, edge_index, edge_attr = load_graph(paths, device=device)
    node_features = build_node_feature_matrix(node_names, paths, device=device)
    graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr).to(device)

    decks = load_simple_decks(paths.decks_json)
    training_examples: List[Tuple[int, List[int]]] = []
    for deck in decks:
        ex = deck_to_training_sequence(deck, node_to_index)
        if ex is not None:
            training_examples.append(ex)

    if not training_examples:
        raise RuntimeError("No training examples matched graph nodes. Check naming consistency.")

    mana_value_by_node = card_decoder.mana_value_from_vectors(graph.x).detach()
    is_land_node = card_decoder.land_mask_from_vectors(graph.x, threshold=0.5).detach()

    mv_bucket_by_node, type_bucket_by_node, bucket_to_indices = precompute_negative_buckets(
        num_nodes=int(graph.x.size(0)),
        mana_value_cpu=mana_value_by_node.detach().cpu().numpy(),
        is_land_cpu=is_land_node.detach().cpu().numpy(),
    )

    combo_edges = build_combo_edge_set(edge_index=edge_index, edge_attr=edge_attr, combo_col=1)

    model = CommanderDeckGNN(
        in_dim=int(graph.x.size(1)),
        edge_dim=int(graph.edge_attr.size(1)),
        hidden_dim=cfg.hidden_dim,
        node_dim=cfg.node_dim,
        state_dim=cfg.state_dim,
        num_layers=cfg.gnn_layers,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    for epoch in range(cfg.epochs):
        avg_loss = train_one_epoch(
            model=model,
            graph=graph,
            training_examples=training_examples,
            cfg=cfg,
            optimizer=optimizer,
            scaler=scaler,
            combo_edges=combo_edges,
            mv_bucket_by_node=mv_bucket_by_node,
            type_bucket_by_node=type_bucket_by_node,
            bucket_to_indices=bucket_to_indices,
        )
        print(f"epoch={epoch} avg_loss={avg_loss:.4f}")

    node_embeddings = None
    if cfg.save_node_embeddings:
        node_embeddings = compute_node_embeddings(model, graph)

    save_checkpoint(
        path=paths.ckpt_pt,
        model=model,
        train_cfg=cfg,
        node_embeddings=node_embeddings,
    )


if __name__ == "__main__":
    main()
