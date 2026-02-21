from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GlobalGraphEncoder(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int, node_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.dropout = float(dropout)

        self.input_projection = nn.Linear(in_dim, hidden_dim)

        self.edge_to_weight = nn.Sequential(
            nn.Linear(edge_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
            nn.Sigmoid(),
        )

        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True) for _ in range(int(num_layers))])

        self.output_projection = nn.Linear(hidden_dim, node_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_projection(x))
        h = F.dropout(h, p=self.dropout, training=self.training)

        edge_weight = self.edge_to_weight(edge_attr).squeeze(-1)

        for conv in self.convs:
            h2 = conv(h, edge_index, edge_weight=edge_weight)
            h = F.relu(h2 + h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        return self.output_projection(h)


class DeckPolicy(nn.Module):
    def __init__(self, node_dim: int, state_dim: int, dropout: float):
        super().__init__()

        self.state_network = nn.Sequential(
            nn.Linear(node_dim, state_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(state_dim, state_dim),
            nn.ReLU(),
        )

        self.scoring_network = nn.Sequential(
            nn.Linear(node_dim + state_dim, state_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(state_dim, 1),
        )

    def make_state(self, commander_vec: torch.Tensor, pool_vec: torch.Tensor, *, strategy_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        if strategy_vec is None:
            mixed = 0.7 * pool_vec + 0.3 * commander_vec
        else:
            mixed = 0.60 * pool_vec + 0.25 * commander_vec + 0.15 * strategy_vec
        return self.state_network(mixed)

    def score_candidates(self, node_embeddings: torch.Tensor, state: torch.Tensor, candidate_indices: torch.Tensor) -> torch.Tensor:
        candidate_embeds = node_embeddings[candidate_indices]
        repeated_state = state.unsqueeze(0).expand(candidate_embeds.size(0), -1)
        return self.scoring_network(torch.cat([candidate_embeds, repeated_state], dim=1)).squeeze(-1)


class CommanderDeckGNN(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int, node_dim: int, state_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.encoder = GlobalGraphEncoder(in_dim, edge_dim, hidden_dim, node_dim, num_layers, dropout)
        self.policy = DeckPolicy(node_dim, state_dim, dropout)

    def encode(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.encoder(node_features, edge_index, edge_attr)
