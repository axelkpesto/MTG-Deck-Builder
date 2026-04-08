import torch

from backend.config import CONFIG
from backend.deckgen.assets import load_assets
from backend.deckgen.config import DeckGenPaths, GenConfig
from backend.deckgen.model import CommanderDeckGNN


def main() -> None:
    paths = DeckGenPaths()
    gen = GenConfig()
    device = torch.device("cpu")

    print("Loading assets (graph + vector DB)...")
    assets = load_assets(paths=paths, device=device, gen=gen)

    print("Loading model checkpoint...")
    ckpt = torch.load(paths.ckpt_pt, map_location=device, weights_only=False)
    train = ckpt["train_cfg"]

    model = CommanderDeckGNN(
        in_dim=int(assets.graph.x.size(1)),
        edge_dim=int(assets.graph.edge_attr.size(1)),
        hidden_dim=int(train["hidden_dim"]),
        node_dim=int(train["node_dim"]),
        state_dim=int(train["state_dim"]),
        num_layers=int(train["gnn_layers"]),
        dropout=float(train["dropout"]),
    ).to(device)

    state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(
        f"Computing node embeddings for {assets.graph.x.size(0)} nodes "
        f"over {assets.graph.edge_index.size(1)} edges. This may use significant RAM..."
    )
    with torch.inference_mode():
        node_embeddings = model.encode(
            assets.graph.x,
            assets.graph.edge_index,
            assets.graph.edge_attr,
        )

    out_path = CONFIG.datasets["NODE_EMBEDDINGS_PATH"]
    torch.save(node_embeddings.cpu(), out_path)

    size_mb = node_embeddings.element_size() * node_embeddings.numel() / 1024 / 1024
    print(f"Saved to {out_path}")
    print(f"  Shape : {tuple(node_embeddings.shape)}")
    print(f"  dtype : {node_embeddings.dtype}")
    print(f"  Size  : {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
