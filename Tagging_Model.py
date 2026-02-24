"""Train and run a multi-label tag predictor from card vectors."""

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from vector_database import VectorDatabase
from card_data import CardEncoder, CardDecoder
from config import CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters."""

    epochs: int = 20
    lr: float = 1e-3
    use_amp: bool = True


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation configuration."""

    threshold: float = 0.5
    batch_size: int = 2048

def build_dataset() -> pd.DataFrame:
    """Build a dataframe of card names, vectors, and tag labels."""
    vd = VectorDatabase(CardEncoder(), CardDecoder())
    vd.load(CONFIG.datasets["VECTOR_DATABASE_PATH"])

    with open(CONFIG.datasets["CARDS_DATASET_PATH"], "r", encoding="utf-8") as f:
        data = json.load(f)

    card_lookup = {card["card_name"]: card.get("tags", []) for card in data}

    rows = []
    for name, vec in vd.items():
        if hasattr(vec, "detach"):
            vec = vec.detach().cpu().numpy()
        elif hasattr(vec, "cpu"):
            vec = vec.cpu().numpy()
        vec = np.asarray(vec, dtype=np.float32)
        rows.append({"name": name, "vector": vec, "tags": card_lookup.get(name, [])})
    return pd.DataFrame(rows)

def prepare_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split features/labels into train and test tensors."""
    feature_series = df["vector"]
    y: list[list[str]] = (
        df["tags"].apply(lambda t: t if isinstance(t, list) else []).tolist()
    )
    names = df["name"].tolist()

    train_series, test_series, y_train_raw, y_test_raw, _, names_test = train_test_split(
        feature_series,
        y,
        names,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    x_train = np.stack(train_series.values).astype(np.float32)
    x_test = np.stack(test_series.values).astype(np.float32)

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train_raw).astype(np.float32)
    y_test = mlb.transform(y_test_raw).astype(np.float32)

    return x_train, x_test, y_train, y_test, mlb, names_test

class VectorsDataset(Dataset):
    """Simple tensor-backed dataset for vector/tag pairs."""

    def __init__(self, features: np.ndarray, y: np.ndarray):
        self.features = torch.from_numpy(features)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        """Return one `(features, labels)` sample."""
        return self.features[idx], self.y[idx]

class MLP(nn.Module):
    """Small MLP for multi-label classification."""

    def __init__(self, input_dim: int, output_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP forward pass."""
        return self.net(x)


def _run_train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    use_cuda_amp: bool,
    scaler: torch.amp.GradScaler | None,
) -> float:
    """Run one training epoch and return mean training loss."""
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_cuda_amp):
            logits = model(xb)
            loss = criterion(logits, yb)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running += loss.item() * xb.size(0)
    return running / len(train_loader.dataset)


def _run_val_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    use_cuda_amp: bool,
) -> float:
    """Run one validation epoch and return mean validation loss."""
    model.eval()
    running = 0.0
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=use_cuda_amp):
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            running += loss.item() * xb.size(0)
    return running / len(val_loader.dataset)

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_cfg: TrainConfig = TrainConfig(),
):
    """Train the model and print epoch-level losses."""
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    use_cuda_amp = device.type == "cuda" and train_cfg.use_amp
    scaler = torch.amp.GradScaler("cuda") if use_cuda_amp else None

    for epoch in range(1, train_cfg.epochs + 1):
        train_loss = _run_train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            use_cuda_amp=use_cuda_amp,
            scaler=scaler,
        )
        val_loss = _run_val_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            use_cuda_amp=use_cuda_amp,
        )

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

@torch.no_grad()
def evaluate(
    model: nn.Module,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    eval_cfg: EvalConfig = EvalConfig(),
):
    """Evaluate model predictions and print aggregate metrics."""
    model.eval()
    model.to(device)

    probs_list = []
    for i in range(0, len(x_test), eval_cfg.batch_size):
        xb = torch.from_numpy(x_test[i : i + eval_cfg.batch_size]).to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits).float().cpu().numpy()
        probs_list.append(probs)
    y_prob = np.concatenate(probs_list, axis=0)
    y_pred = (y_prob >= eval_cfg.threshold).astype(np.int32)

    subset_acc = accuracy_score(y_test, y_pred)
    micro_p = precision_score(y_test, y_pred, average="micro", zero_division=0)
    micro_r = recall_score(y_test, y_pred, average="micro", zero_division=0)
    micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)

    macro_p = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n=== Test Metrics ===")
    print(f"Subset Accuracy (exact match): {subset_acc:.4f}")
    print(f"Micro  - Precision: {micro_p:.4f}  Recall: {micro_r:.4f}  F1: {micro_f1:.4f}")
    print(f"Macro  - Precision: {macro_p:.4f}  Recall: {macro_r:.4f}  F1: {macro_f1:.4f}")

    print("\nPer-class classification report:")
    print(classification_report(y_test, y_pred, target_names=list(class_names), zero_division=0))

def save_model(model: nn.Module, mlb: MultiLabelBinarizer, path: str, model_kwargs: dict):
    """Persist model state and metadata to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "class_names": list(mlb.classes_),
        "model_kwargs": model_kwargs,
    }
    torch.save(payload, path)
    print(f"[saved] {path}")

@torch.no_grad()
def load_model(path: str) -> tuple[nn.Module, list[str]]:
    """Load a saved model and return it with class names."""
    payload = torch.load(path, map_location=device)
    kw = payload["model_kwargs"]
    model = MLP(**kw).eval()
    model.load_state_dict(payload["state_dict"])
    return model, payload["class_names"]

def main():
    """CLI entrypoint for training/evaluating/saving tag model."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load",
        action="store_true",
        default=False,
        help="If set, load the trained model.",
    )
    parser.add_argument("--load_path", type=str, default=CONFIG.models["TAGGING_MODEL_PATH"])

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Probability threshold for predicting a tag.",
    )
    parser.add_argument("--show_n", type=int, default=8, help="How many test examples to print.")

    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="If set, save the trained model.",
    )
    parser.add_argument("--save_path", type=str, default=CONFIG.models["TAGGING_MODEL_PATH"])
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Use mixed precision on CUDA.",
    )
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    df = build_dataset()
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    x_train, x_test, y_train, y_test, mlb, _ = prepare_dataset(
        df,
        test_size=0.2,
        random_state=args.seed,
    )

    if args.load:
        model, _ = load_model(args.load_path)
    else:

        train_ds = VectorsDataset(x_train, y_train)
        test_ds = VectorsDataset(x_test, y_test)
        model = MLP(
            input_dim=x_train.shape[1],
            output_dim=y_train.shape[1],
            hidden=args.hidden,
        ).to(device)

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        train(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_cfg=TrainConfig(
                epochs=args.epochs,
                lr=args.lr,
                use_amp=args.amp,
            ),
        )

    evaluate(
        model,
        x_test=x_test,
        y_test=y_test,
        class_names=mlb.classes_,
        eval_cfg=EvalConfig(
            threshold=args.threshold,
            batch_size=2048,
        ),
    )

    if args.save:
        model_kwargs = {
            "input_dim": x_train.shape[1],
            "output_dim": y_train.shape[1],
            "hidden": args.hidden,
        }
        save_model(model, mlb, path=args.save_path, model_kwargs=model_kwargs)

if __name__ == "__main__":
    main()
