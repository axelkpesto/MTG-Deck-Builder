import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse
import os
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from Vector_Database import VectorDatabase

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_dataset() -> pd.DataFrame:
    vd = VectorDatabase()
    vd.load("datasets/vector_data.pt")
    
    with open("datasets/CommanderCards.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    card_lookup = {card['card_name']: card.get("tags", []) for card in data}

    rows = []
    for name, vec in vd.items():
        if hasattr(vec, "detach"):
            vec = vec.detach().cpu().numpy()
        elif hasattr(vec, "cpu"):
            vec = vec.cpu().numpy()
        vec = np.asarray(vec, dtype=np.float32)
        rows.append({"name": name, "vector": vec, "tags": card_lookup.get(name, [])})
    
    print(f"Built dataset with {len(rows)}")
    return pd.DataFrame(rows)


def prepare_xy(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df['vector']
    y: list[list[str]] = df['tags'].apply(lambda t: t if isinstance(t, list) else []).tolist()
    names = df['name'].tolist()

    X_train_s, X_test_s, y_train_raw, y_test_raw, _, names_test = train_test_split(
        X, y, names, test_size=test_size, random_state=random_state, shuffle=True
    )

    X_train = np.stack(X_train_s.values).astype(np.float32)
    X_test  = np.stack(X_test_s.values).astype(np.float32)

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train_raw).astype(np.float32)
    y_test  = mlb.transform(y_test_raw).astype(np.float32)

    return X_train, X_test, y_train, y_test, mlb, names_test


class VectorsDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
    
class MLP(nn.Module):
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
        return self.net(x)


def train_torch(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 20, lr: float = 1e-3, use_amp: bool = True):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    use_cuda_amp = (device.type == "cuda") and use_amp
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_cuda_amp):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * xb.size(0)

        train_loss = running / len(train_loader.dataset)

        model.eval()
        val_running = 0.0
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=use_cuda_amp):
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_running += loss.item() * xb.size(0)
        val_loss = val_running / len(val_loader.dataset)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

@torch.no_grad()
def evaluate_torch(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray, class_names: list[str], threshold: float = 0.5, batch_size: int = 2048):
    model.eval()
    model.to(device)

    probs_list = []
    for i in range(0, len(X_test), batch_size):
        xb = torch.from_numpy(X_test[i : i + batch_size]).to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits).float().cpu().numpy()
        probs_list.append(probs)
    y_prob = np.concatenate(probs_list, axis=0)
    y_pred = (y_prob >= threshold).astype(np.int32)

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

@torch.no_grad()
def print_example_predictions(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray, test_names: list[str], class_names: list[str], k: int = 8, threshold: float = 0.5):
    if k <= 0: return
    model.eval()
    model.to(device)

    k = min(k, len(X_test))
    idxs = np.random.choice(len(X_test), size=k, replace=False)

    xb = torch.from_numpy(X_test[idxs]).to(device)
    logits = model(xb)
    probs = torch.sigmoid(logits).float().cpu().numpy()

    for row, idx in enumerate(idxs):
        name = test_names[idx]
        true_idxs = np.where(y_test[idx] == 1)[0]
        pred_idxs = np.where(probs[row] >= threshold)[0]
        
        outliers = list(set(true_idxs).symmetric_difference(set(pred_idxs)))

        true_tags = [class_names[j] for j in true_idxs.tolist()]
        pred_tags = [f"{class_names[j]} ({probs[row][j]:.2f})" for j in pred_idxs.tolist()]

        if len(pred_tags) == 0:
            top3 = probs[row].argsort()[-3:][::-1]
            pred_tags = [f"{class_names[j]} ({probs[row][j]:.2f})" for j in top3]

        print(f"\n• {name}")
        print(f"  True tags: {true_tags}")
        print(f"  Predicted (@{threshold:.2f}): {pred_tags}")
        print(f"  Outliers: {[class_names[j] for j in outliers]}")

def save_model(model: nn.Module, mlb: MultiLabelBinarizer, path: str, model_kwargs: dict):
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
    payload = torch.load(path, map_location=device)
    kw = payload['model_kwargs']
    model = MLP(**kw).eval()
    model.load_state_dict(payload['state_dict'])
    return model, payload['class_names']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true", default=False, help="If set, load the trained model.")
    parser.add_argument("--load_path", type=str, default="models/tagging_mlp.pt")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)

    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for predicting a tag.")
    parser.add_argument("--show_n", type=int, default=8, help="How many test examples to print.")

    parser.add_argument("--save", action="store_true", default=True, help="If set, save the trained model.")
    parser.add_argument("--save_path", type=str, default="models/tagging_mlp.pt")
    parser.add_argument("--amp", action="store_true", default=True, help="Use mixed precision on CUDA.")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    df = build_dataset()
    df = df.sample(frac=1, random_state=(args.seed)).reset_index(drop=True)

    X_train, X_test, y_train, y_test, mlb, names_test = prepare_xy(df, test_size=0.2, random_state=args.seed)

    if args.load:
        model, _ = load_model(args.load_path)
    else:

        train_ds = VectorsDataset(X_train, y_train)
        test_ds = VectorsDataset(X_test, y_test)
        model = MLP(input_dim=X_train.shape[1], output_dim=y_train.shape[1], hidden=args.hidden).to(device)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == 'cuda'))
        val_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))
        train_torch(model, train_loader=train_loader, val_loader=val_loader, epochs=args.epochs, lr=args.lr, use_amp=args.amp)

    evaluate_torch(model, X_test=X_test, y_test=y_test, class_names=mlb.classes_, threshold=args.threshold)
    
    print_example_predictions(model, X_test=X_test, y_test=y_test, test_names=names_test, class_names=list(mlb.classes_), k=args.show_n, threshold=args.threshold)
    
    if args.save:
        model_kwargs = {
            "input_dim": X_train.shape[1],
            "output_dim": y_train.shape[1],
            "hidden": args.hidden,
        }
        save_model(model, mlb, path=args.save_path, model_kwargs=model_kwargs)

if __name__ == "__main__":
    main()