"""Deck generation logic driven by graph embeddings and heuristic constraints."""

from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from card_data import CardFields

from deckgen.assets import DeckGenAssets
from deckgen.config import GenConfig
from deckgen.model import CommanderDeckGNN
from deckgen.utils import clamp_int, mana_value_bucket, basic_land_type, is_basic_land_name, allowed_basic_land_types, duplicate_penalty, extract_basic_ratio, extract_curve_counts, extract_land_count, extract_tag_count

def pool_stats_for_commander(commander_name: str, commander_index: int, assets: DeckGenAssets, node_embeddings: torch.Tensor, gen: GenConfig) -> List[dict]:
    """Collect training stats for commander-specific and similar commanders."""
    pooled: List[dict] = []
    pooled.extend(assets.by_commander_stats.get(commander_name, []))

    cmd_ids = assets.commander_indices
    if cmd_ids:
        cand = torch.tensor(cmd_ids, device=node_embeddings.device, dtype=torch.long)
        cmd_vec = node_embeddings[commander_index]
        cmd_vec = cmd_vec / (cmd_vec.norm(p=2) + 1e-8)

        cand_vecs = node_embeddings[cand]
        cand_vecs = cand_vecs / (cand_vecs.norm(p=2, dim=1, keepdim=True) + 1e-8)
        sims = cand_vecs @ cmd_vec

        # remove self if present
        self_pos = (cand == commander_index).nonzero(as_tuple=False)
        if self_pos.numel() > 0:
            sims[self_pos[0, 0]] = -1.0

        k = min(int(gen.similar_commander_k), int(sims.numel()))
        if k > 0:
            _, top_pos = torch.topk(sims, k=k)
            for p in top_pos.tolist():
                idx = int(cand[p].item())
                name = assets.node_names[idx]
                pooled.extend(assets.by_commander_stats.get(name, []))

    # Fallback to global if still empty
    if not pooled:
        pooled = list(assets.global_stats)

    return pooled


def learn_land_profile(pooled: List[dict], gen: GenConfig) -> Tuple[int, int, float, int]:
    """Learn land and basics targets from pooled historical deck stats."""
    land_counts = [int(extract_land_count(s)) for s in pooled]
    basic_ratios = [float(extract_basic_ratio(s)) for s in pooled]

    a = np.asarray(land_counts, dtype=np.float32)
    land_target = int(np.quantile(a, float(gen.land_q)))
    land_cap = int(np.quantile(a, float(gen.land_cap_q)))

    b = np.asarray(basic_ratios, dtype=np.float32)
    basic_ratio_target = float(np.quantile(b, float(gen.basic_ratio_q)))

    land_target = clamp_int(land_target, gen.land_min, gen.land_max)
    land_cap = clamp_int(max(land_cap, land_target), land_target, gen.land_max)
    basic_ratio_target = float(np.clip(basic_ratio_target, gen.basic_ratio_min, gen.basic_ratio_max))

    basics_target = int(round(land_target * basic_ratio_target))
    basics_target = clamp_int(basics_target, gen.basics_min, gen.basics_max)
    return land_target, land_cap, basic_ratio_target, basics_target


def learn_curve_target(pooled: List[dict]) -> List[int]:
    """Learn target mana-curve bucket counts from pooled deck stats."""
    curves = [c for c in (extract_curve_counts(s) for s in pooled) if len(c) >= 7]
    a = np.asarray(curves, dtype=np.float32)
    med = np.median(a, axis=0)
    return [int(round(x)) for x in med[:7].tolist()]


def learn_tag_bounds(pooled: List[dict], tag: str, gen: GenConfig, fallback_min: int, fallback_max: int) -> Tuple[int, int]:
    """Learn lower/upper bounds for a tag count from pooled stats."""
    vals = [int(extract_tag_count(s, tag)) for s in pooled]
    a = np.asarray(vals, dtype=np.float32)
    lo = int(np.quantile(a, float(gen.tag_min_q)))
    hi = int(np.quantile(a, float(gen.tag_max_q)))
    hi = max(hi, lo)
    return max(0, lo if lo is not None else fallback_min), max(0, hi if hi is not None else fallback_max)


def learn_ramp_target(pooled: List[dict], gen: GenConfig) -> int:
    """Learn preferred ramp count from pooled deck statistics."""
    vals = [int(extract_tag_count(s, gen.ramp_tag)) for s in pooled]
    a = np.asarray(vals, dtype=np.float32)
    t = int(np.quantile(a, float(gen.ramp_q)))
    return clamp_int(t, gen.ramp_min, gen.ramp_max)


class DeckState:
    """Mutable generation state tracked while constructing a deck."""

    def __init__(self, deck_indices: List[int], nonbasic_already_picked: torch.Tensor) -> None:
        """Initialize counters and cached values for one generation run."""
        self.deck_indices = deck_indices
        self.nonbasic_already_picked = nonbasic_already_picked

        self.selected_sum: Optional[torch.Tensor] = None
        self.selected_count: int = 0
        self.lands_picked: int = 0
        self.basics_picked: int = 0
        self.curve_counts: List[int] = [0] * 7
        self.tag_counts: Dict[str, int] = defaultdict(int)
        self.dominant_strategy_tags: List[str] = []
        self.ramp_count: int = 0
        self.basics_picked_by_type: Dict[str, int] = defaultdict(int)


class DeckGenerator:
    """Generate a deck list for a commander using model scores and constraints."""

    def __init__(self, model: CommanderDeckGNN, assets: DeckGenAssets, commander_name: str, gen: GenConfig, allow_duplicates: bool, node_embeddings: Optional[torch.Tensor] = None):
        """Prepare model state, targets, masks, and helper caches."""
        self.model = model
        self.assets = assets
        self.commander_name = commander_name
        self.gen = gen
        self.allow_duplicates = bool(allow_duplicates)

        self.graph = assets.graph
        self.device = self.graph.x.device
        self.num_nodes = int(self.graph.x.size(0))

        self.commander_index = int(assets.node_to_index[commander_name])
        self.neighbors_by_node = [n.to(self.device, non_blocking=True).long() for n in self.assets.neighbors_by_node]

        self.node_embeddings = node_embeddings if node_embeddings is not None else model.encode(self.graph.x, self.graph.edge_index, self.graph.edge_attr)

        pooled = pool_stats_for_commander(commander_name=commander_name, commander_index=self.commander_index, assets=assets, node_embeddings=self.node_embeddings, gen=gen)
        self.land_target, self.land_cap, self.basic_ratio_target, self.basics_target = learn_land_profile(pooled, gen)
        self.curve_target = learn_curve_target(pooled)
        self.ramp_target = learn_ramp_target(pooled, gen)

        self.illegal_by_color, self.card_colors, self.commander_colors = self.compute_color_legality()

        self.is_basic_node = torch.tensor(
            [is_basic_land_name(n) for n in self.assets.node_names],
            dtype=torch.bool,
            device=self.device,
        )
        self.land_is_legal = self.assets.is_land_node & ~self.illegal_by_color

        # lands with 2+ colors in identity
        color_counts = self.card_colors.float().sum(dim=1)
        self.is_fixing_land = self.assets.is_land_node & ~self.is_basic_node & (color_counts >= 2.0)
        self.is_utility_land = self.assets.is_land_node & ~self.is_basic_node & ~self.is_fixing_land

        # Allowed basics + per-type targets
        self.allowed_basics = allowed_basic_land_types(self.commander_colors)
        self.allowed_basic_tensor = self.compute_allowed_basic_tensor(self.allowed_basics)
        self.basic_targets_by_land_type = self.build_basic_targets_by_type(self.allowed_basics, self.basics_target)

        basic_types = sorted({basic_land_type(name) for name in CardFields.basic_lands()})
        self.basic_type_to_idx = {t: i for i, t in enumerate(basic_types)}
        self.node_basic_type_idx = torch.full((self.num_nodes,), -1, dtype=torch.long, device=self.device)
        for i, name in enumerate(self.assets.node_names):
            if is_basic_land_name(name):
                t = basic_land_type(name)
                idx = self.basic_type_to_idx.get(t)
                if idx is not None:
                    self.node_basic_type_idx[i] = int(idx)

        self.tag_map = self.assets.tag_map
        self.is_ramp_node = self.build_tag_mask(self.gen.ramp_tag)
        self.strategy_tag_nodes: Dict[str, torch.Tensor] = {}
        self.strategy_centroid_cache: Dict[Tuple[str, ...], Optional[torch.Tensor]] = {}

        self.common_tag_masks: Dict[str, torch.Tensor] = {}
        for tp in self.gen.common_tag_penalties:
            self.common_tag_masks[tp.tag] = self.build_tag_mask(tp.tag)

        self.common_tag_bounds: Dict[str, Tuple[int, int]] = {}
        for tp in self.gen.common_tag_penalties:
            self.common_tag_bounds[tp.tag] = learn_tag_bounds(pooled, tp.tag, gen, fallback_min=tp.min_target, fallback_max=tp.max_target)

        self.non_strategy = set(self.gen.non_strategy_tags)
        tag_to_nodes: Dict[str, List[int]] = defaultdict(list)
        for idx, name in enumerate(self.assets.node_names):
            if bool(self.assets.is_land_node[idx].item()):
                continue
            if bool(self.is_ramp_node[idx].item()):
                continue
            for t in self.tag_map.get(name, []):
                tag_to_nodes[t].append(idx)
        for t, idxs in tag_to_nodes.items():
            if idxs:
                self.strategy_tag_nodes[t] = torch.tensor(idxs, device=self.device, dtype=torch.long)

    def build_tag_mask(self, tag: str) -> torch.Tensor:
        """Create a boolean node mask for cards containing a given tag."""
        mask = torch.zeros((self.num_nodes,), dtype=torch.bool, device=self.device)
        for i, name in enumerate(self.assets.node_names):
            if tag in self.tag_map.get(name, []):
                mask[i] = True
        return mask

    def compute_color_legality(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-node commander color legality and color masks."""
        commander_colors = self.assets.color_identity_mask[self.commander_index].clone()

        if "C" in CardFields.color_identities():
            colorless_pos = CardFields.color_identities().index("C")
            commander_colors[colorless_pos] = False
            card_colors = self.assets.color_identity_mask.clone()
            card_colors[:, colorless_pos] = False
        else:
            card_colors = self.assets.color_identity_mask

        illegal_by_color = (card_colors & ~commander_colors.unsqueeze(0)).any(dim=1)

        allowed_basics = allowed_basic_land_types(commander_colors)
        allowed_basic_tensor = self.compute_allowed_basic_tensor(allowed_basics)

        basic_legal_override = torch.zeros((self.num_nodes,), dtype=torch.bool, device=self.device)
        if allowed_basic_tensor.numel() > 0:
            basic_legal_override[allowed_basic_tensor] = True

        illegal_by_color = illegal_by_color & ~basic_legal_override
        return illegal_by_color, card_colors, commander_colors

    def compute_allowed_basic_tensor(self, allowed_basics: List[str]) -> torch.Tensor:
        """Return node indices corresponding to allowed basic land names."""
        allowed_set = {b.lower() for b in allowed_basics}
        idxs = [i for i, name in enumerate(self.assets.node_names) if name.strip().lower() in allowed_set]
        return torch.tensor(idxs, dtype=torch.long, device=self.device) if idxs else torch.empty((0,), dtype=torch.long, device=self.device)

    def build_basic_targets_by_type(self, allowed_basics: List[str], basics_target: int) -> Dict[str, int]:
        """Distribute desired basic land count across allowed basic types."""
        types = [b for b in allowed_basics if b]
        if not types:
            return {}
        per = max(1, basics_target // len(types))
        out = {t: per for t in types}
        rem = basics_target - per * len(types)
        i = 0
        while rem > 0:
            out[types[i % len(types)]] += 1
            rem -= 1
            i += 1
        return out

    def candidate_pool(self, last_picked: int) -> torch.Tensor:
        """Build the candidate node pool from neighbors and random exploration."""
        pools: List[torch.Tensor] = []

        cmd_neigh = self.neighbors_by_node[self.commander_index]
        if cmd_neigh.numel() > 0:
            pools.append(cmd_neigh)

        last_neigh = self.neighbors_by_node[last_picked]
        if last_neigh.numel() > 0:
            pools.append(last_neigh)

        if self.gen.explore_random > 0:
            pools.append(torch.randint(0, self.num_nodes, (int(self.gen.explore_random),), device=self.device, dtype=torch.long))

        if self.allowed_basic_tensor.numel() > 0:
            pools.append(self.allowed_basic_tensor)

        cand = torch.unique(torch.cat(pools, dim=0)) if pools else torch.arange(self.num_nodes, device=self.device, dtype=torch.long)

        if cand.numel() > int(self.gen.candidate_budget):
            cand = cand[torch.randperm(cand.numel(), device=self.device)[: int(self.gen.candidate_budget)]]

        return cand.long()

    def apply_land_pressure(self, logits: torch.Tensor, state: DeckState, cand: torch.Tensor) -> None:
        """Adjust logits to steer land counts and basic-land composition."""
        land = self.assets.is_land_node[cand]
        if state.lands_picked < self.land_target:
            logits[land] += float(self.gen.land_boost)
        else:
            logits[land] -= float(self.gen.land_penalty_after_target)

        ratio = (state.basics_picked / max(1, state.lands_picked)) if state.lands_picked > 0 else 0.0
        is_basic = self.is_basic_node[cand]

        if ratio < self.basic_ratio_target:
            logits[is_basic] += float(self.gen.basic_min_boost)
        else:
            logits[is_basic] -= float(self.gen.basic_land_penalty_when_over_ratio)

        if self.gen.force_basic_when_behind and state.lands_picked > 0:
            desired_basics_now = int(round(state.lands_picked * self.basic_ratio_target))
            if state.basics_picked < desired_basics_now:
                logits[is_basic] += float(self.gen.basic_force_boost)

        if self.allowed_basic_tensor.numel() > 0:
            cand_basic_type_idx = self.node_basic_type_idx[cand]
            for bt, need in self.basic_targets_by_land_type.items():
                t_idx = self.basic_type_to_idx.get(bt)
                if t_idx is None:
                    continue
                mask = cand_basic_type_idx == int(t_idx)
                if not bool(mask.any().item()):
                    continue
                have = int(state.basics_picked_by_type.get(bt, 0))
                if have < need:
                    logits[mask] += float(self.gen.basic_type_boost)
                else:
                    extra = have - need + 1
                    pen = min(
                        float(self.gen.dup_penalty_cap),
                        duplicate_penalty(extra, float(self.gen.dup_penalty_lambda), float(self.gen.dup_penalty_power)),
                    )
                    logits[mask] -= pen

    def apply_curve_pressure(self, logits: torch.Tensor, state: DeckState, cand: torch.Tensor) -> None:
        """Penalize overfilled mana-curve buckets in current candidates."""
        mv = self.assets.mana_value_by_node[cand]
        bins = torch.clamp(mv.round().long(), min=0, max=6)
        for b in range(7):
            want = int(self.curve_target[b])
            have = int(state.curve_counts[b])
            if have > want:
                logits[bins == b] -= float(self.gen.curve_penalty)

    def apply_ramp_pressure(self, logits: torch.Tensor, state: DeckState, cand: torch.Tensor) -> None:
        """Penalize ramp-tag cards once learned ramp targets are exceeded."""
        is_ramp = self.is_ramp_node[cand]
        if state.ramp_count > (self.ramp_target + int(self.gen.ramp_hard_buffer)):
            logits[is_ramp] -= float(self.gen.ramp_hard_penalty)
        elif state.ramp_count > self.ramp_target:
            logits[is_ramp] -= float(self.gen.ramp_soft_penalty)

    def apply_common_tag_pressure(self, logits: torch.Tensor, state: DeckState, cand: torch.Tensor) -> None:
        """Apply soft/hard penalties for common tag overrepresentation."""
        for tp in self.gen.common_tag_penalties:
            mask = self.common_tag_masks[tp.tag][cand]
            lo, hi = self.common_tag_bounds[tp.tag]
            have = int(state.tag_counts.get(tp.tag, 0))

            if have > (hi + int(tp.hard_buffer)):
                logits[mask] -= float(tp.hard_penalty)
            elif have > hi:
                logits[mask] -= float(tp.soft_penalty)
            elif have < lo:
                logits[mask] += 0.15

    def update_strategy_tags(self, state: DeckState) -> None:
        """Update dominant strategy tags from observed tag counts so far."""
        start_after = 10
        topn = 3

        if len(state.deck_indices) >= start_after:
            scored = [(t, c) for t, c in state.tag_counts.items() if t not in self.non_strategy]
            scored.sort(key=lambda x: x[1], reverse=True)
            state.dominant_strategy_tags = [t for t, _ in scored[:topn]]

    def strategy_centroid(self, state: DeckState) -> Optional[torch.Tensor]:
        """Return cached centroid embedding for dominant strategy tags."""
        if not state.dominant_strategy_tags:
            return None

        key = tuple(sorted(state.dominant_strategy_tags))
        if key in self.strategy_centroid_cache:
            return self.strategy_centroid_cache[key]

        parts = [self.strategy_tag_nodes[t] for t in key if t in self.strategy_tag_nodes]
        if not parts:
            self.strategy_centroid_cache[key] = None
            return None

        idxs = torch.unique(torch.cat(parts, dim=0))
        if idxs.numel() == 0:
            self.strategy_centroid_cache[key] = None
            return None
        if idxs.numel() > 256:
            idxs = idxs[:256]

        centroid = self.node_embeddings[idxs].mean(dim=0)
        self.strategy_centroid_cache[key] = centroid
        return centroid

    def base_logits(self, cand: torch.Tensor, state_vec: torch.Tensor) -> torch.Tensor:
        """Score candidates and mask out illegal-by-color cards."""
        cand = cand.long()
        logits = self.model.policy.score_candidates(self.node_embeddings, state_vec, cand).float()
        logits[self.illegal_by_color[cand]] = -1e9
        return logits

    def pick(self, cand: torch.Tensor, logits: torch.Tensor) -> Tuple[int, float]:
        """Sample a node from top-k logits using temperature-scaled softmax."""
        k = min(int(self.gen.top_k), int(cand.numel()))
        if k <= 0:
            raise RuntimeError("Empty candidate set")

        topv, topi = torch.topk(logits, k=k)
        probs = F.softmax(topv / max(1e-6, float(self.gen.temperature)), dim=0)
        choice = torch.multinomial(probs, num_samples=1).item()
        pick_pos = int(topi[choice].item())
        pick_idx = int(cand[pick_pos].item())
        pick_logit = float(topv[choice].item())
        return pick_idx, pick_logit

    def run(self) -> List[int]:
        """Run iterative sampling until deck size is reached or stalled."""
        nonbasic_already = torch.zeros((self.num_nodes,), dtype=torch.bool, device=self.device)
        state = DeckState(deck_indices=[self.commander_index], nonbasic_already_picked=nonbasic_already)

        state.selected_sum = torch.zeros((self.node_embeddings.size(1),), dtype=self.node_embeddings.dtype, device=self.device)
        state.selected_count = 0

        for _ in range(int(self.gen.deck_size) - 1):
            last = state.deck_indices[-1]
            cand = self.candidate_pool(last)

            cand = cand[cand != self.commander_index]
            if cand.numel() == 0:
                break

            if not self.allow_duplicates:
                cand_is_basic = self.is_basic_node[cand]
                dup_nonbasic = state.nonbasic_already_picked[cand] & ~cand_is_basic
                cand = cand[~dup_nonbasic]
                if cand.numel() == 0:
                    break

            commander_vec = self.node_embeddings[self.commander_index]
            pool_vec = (state.selected_sum / float(state.selected_count)) if state.selected_count > 0 else torch.zeros_like(commander_vec)
            strategy_vec = self.strategy_centroid(state)
            state_vec = self.model.policy.make_state(commander_vec, pool_vec, strategy_vec=strategy_vec)

            logits = self.base_logits(cand, state_vec)

            self.apply_land_pressure(logits, state, cand)
            self.apply_curve_pressure(logits, state, cand)
            self.apply_ramp_pressure(logits, state, cand)
            self.apply_common_tag_pressure(logits, state, cand)

            pick_idx, _ = self.pick(cand, logits)

            pick_name = self.assets.node_names[pick_idx]
            state.deck_indices.append(pick_idx)

            if not is_basic_land_name(pick_name):
                state.nonbasic_already_picked[pick_idx] = True

            if bool(self.assets.is_land_node[pick_idx].item()):
                state.lands_picked += 1
                if is_basic_land_name(pick_name):
                    state.basics_picked += 1
                    bt = basic_land_type(pick_name)
                    state.basics_picked_by_type[bt] += 1
            else:
                if not bool(self.is_ramp_node[pick_idx].item()):
                    state.selected_sum = state.selected_sum + self.node_embeddings[pick_idx]
                    state.selected_count += 1

            b = mana_value_bucket(float(self.assets.mana_value_by_node[pick_idx].item()))
            state.curve_counts[b] += 1

            for t in self.tag_map.get(pick_name, []):
                state.tag_counts[t] += 1
            if self.gen.ramp_tag in self.tag_map.get(pick_name, []):
                state.ramp_count += 1

            self.update_strategy_tags(state)

        return state.deck_indices


def generate_deck(model: CommanderDeckGNN, assets: DeckGenAssets, commander_name: str, gen: GenConfig, allow_duplicates: bool = False, node_embeddings: Optional[torch.Tensor] = None) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """Generate a deck and return name counts plus summary stats."""
    g = DeckGenerator(model=model, assets=assets, commander_name=commander_name, gen=gen, allow_duplicates=allow_duplicates, node_embeddings=node_embeddings)

    deck_indices = g.run()

    counts: Dict[str, int] = defaultdict(int)
    for idx in deck_indices:
        counts[assets.node_names[idx]] += 1

    stats: Dict[str, Any] = {
        "commander": commander_name,
        "n_cards": int(sum(counts.values())),
    }
    return dict(counts), stats
