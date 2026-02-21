from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from Vector_Database import VectorDatabase
from deckgen.assets import DeckGenAssets, load_assets
from deckgen.config import DeckGenPaths, GenConfig
from deckgen.generator import generate_deck
from deckgen.model import CommanderDeckGNN

class DeckGenBundle:
    def __init__(self, model: CommanderDeckGNN, assets: DeckGenAssets, gen: GenConfig, device: torch.device, node_embeddings: Optional[torch.Tensor] = None) -> None:
        self.model = model
        self.assets = assets
        self.gen = gen
        self.device = device
        self.node_embeddings = node_embeddings

    @classmethod
    def load(cls, paths: Optional[DeckGenPaths] = None, gen: Optional[GenConfig] = None, device: str = "cpu", vector_db: Optional[VectorDatabase] = None) -> "DeckGenBundle":
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

        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.eval()

        return cls(model=model, assets=assets, gen=gen, device=dev)

    @torch.inference_mode()
    def get_node_embeddings(self) -> torch.Tensor:
        if self.node_embeddings is None:
            self.node_embeddings = self.model.encode(self.assets.graph.x, self.assets.graph.edge_index, self.assets.graph.edge_attr)
        return self.node_embeddings


    def generate(self, commander_name: str, allow_duplicates: bool = False) -> Tuple[Dict[str, int], Dict[str, object]]:
        node_embeddings = self.get_node_embeddings()

        return generate_deck(
            model=self.model,
            assets=self.assets,
            commander_name=commander_name,
            gen=self.gen,
            allow_duplicates=allow_duplicates,
            node_embeddings=node_embeddings,
        )
