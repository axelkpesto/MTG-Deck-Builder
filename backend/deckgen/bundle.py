import os
from typing import Dict, Optional, Tuple

import torch

from backend.deckgen.assets import DeckGenAssets, load_assets
from backend.deckgen.config import DeckGenPaths, GenConfig
from backend.deckgen.generator import CommanderCache, build_commander_cache, generate_deck
from backend.deckgen.model import CommanderDeckGNN
from backend.vector_database import VectorDatabase

class DeckGenBundle:

    def __init__(self, model: CommanderDeckGNN, assets: DeckGenAssets, gen: GenConfig, device: torch.device, node_embeddings: Optional[torch.Tensor] = None) -> None:
        """Store the model, assets, and generation config as a ready-to-use bundle.

        Args:
            model: Trained CommanderDeckGNN model in eval mode.
            assets: Pre-loaded DeckGenAssets containing graph and metadata.
            gen: Generation hyperparameters.
            device: Torch device the tensors reside on.
            node_embeddings: Optional pre-computed node embedding matrix.

        Returns:
            None
        """
        self.model = model
        self.assets = assets
        self.gen = gen
        self.device = device
        self.node_embeddings = node_embeddings
        self.commander_cache: Dict[str, CommanderCache] = {}

    @classmethod
    def load(cls, paths: Optional[DeckGenPaths] = None, gen: Optional[GenConfig] = None, device: str = "cpu", vector_db: Optional[VectorDatabase] = None) -> "DeckGenBundle":
        """Load assets and a trained checkpoint into a ready-to-use bundle.

        Args:
            paths: DeckGenPaths pointing to all required data files; defaults to DeckGenPaths().
            gen: Generation config; defaults to GenConfig().
            device: Device string (e.g. 'cpu', 'cuda').
            vector_db: Optional pre-loaded VectorDatabase to avoid reloading from disk.

        Returns:
            A fully initialized DeckGenBundle ready for deck generation.
        """
        dev = torch.device(device)
        paths = paths or DeckGenPaths()
        gen = gen or GenConfig()

        assets = load_assets(paths=paths, device=dev, gen=gen, vector_db=vector_db)

        ckpt = torch.load(paths.ckpt_pt, map_location=dev, weights_only=False)
        train = ckpt["train_cfg"]

        model = CommanderDeckGNN(
            in_dim=int(assets.graph.x.size(1)),
            edge_dim=int(assets.graph.edge_attr.size(1)),
            hidden_dim=int(train["hidden_dim"]),
            node_dim=int(train["node_dim"]),
            state_dim=int(train["state_dim"]),
            num_layers=int(train["gnn_layers"]),
            dropout=float(train["dropout"]),
        ).to(dev)

        state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict")
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        # Pre-computed node embeddings avoid running the full GNN encode at inference time.
        # GCNConv message passing creates (num_edges × hidden_dim) intermediates which can
        # exceed 32 GB RAM on large graphs.
        node_embeddings: Optional[torch.Tensor] = None
        if os.path.isfile(paths.node_embeddings_pt):
            node_embeddings = torch.load(paths.node_embeddings_pt, map_location=dev, weights_only=True)
        elif "node_embeddings" in ckpt:
            node_embeddings = ckpt["node_embeddings"].to(dev)

        return cls(model=model, assets=assets, gen=gen, device=dev, node_embeddings=node_embeddings)

    @torch.inference_mode()
    def get_node_embeddings(self) -> torch.Tensor:
        """Return cached node embeddings, computing them on first call.

        Args:
            None

        Returns:
            Float tensor of shape (N, node_dim) with one embedding per graph node.
        """
        if self.node_embeddings is None:
            self.node_embeddings = self.model.encode(self.assets.graph.x, self.assets.graph.edge_index, self.assets.graph.edge_attr)
        return self.node_embeddings

    def get_commander_cache(self, commander_name: str) -> CommanderCache:
        """Return cached commander-specific generation metadata, computing on first call.

        Args:
            commander_name: Canonical name of the commander card.

        Returns:
            CommanderCache with pre-computed targets and legality masks.
        """
        cache = self.commander_cache.get(commander_name)
        if cache is None:
            cache = build_commander_cache(
                assets=self.assets,
                commander_name=commander_name,
                commander_index=int(self.assets.node_to_index[commander_name]),
                node_embeddings=self.get_node_embeddings(),
                gen=self.gen,
            )
            self.commander_cache[commander_name] = cache
        return cache

    def generate(self, commander_name: str, allow_duplicates: bool = False) -> Tuple[Dict[str, int], Dict[str, object]]:
        """Generate a deck list and summary stats for a given commander.

        Args:
            commander_name: Canonical name of the commander card.
            allow_duplicates: If True, non-basic cards may appear more than once.

        Returns:
            A tuple of (card_counts, stats) where card_counts maps card name to quantity.
        """
        node_embeddings = self.get_node_embeddings()
        commander_cache = self.get_commander_cache(commander_name)

        return generate_deck(
            model=self.model,
            assets=self.assets,
            commander_name=commander_name,
            gen=self.gen,
            allow_duplicates=allow_duplicates,
            node_embeddings=node_embeddings,
            commander_cache=commander_cache,
        )
