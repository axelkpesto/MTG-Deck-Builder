import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tqdm
import copy

from Vector_Database import VectorDatabase
from Card_Lib import CardEncoder, CardDecoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_dataset() -> pd.DataFrame:
    vd = VectorDatabase(CardEncoder(), CardDecoder())
    vd.load("datasets/vector_data.pt")

    with open("datasets/RatingData.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for name, vec in vd.items():
        if name not in data:
            continue
        if hasattr(vec, "detach"):
            vec = vec.detach().cpu().numpy()
        elif hasattr(vec, "cpu"):
            vec = vec.cpu().numpy()
        vec = np.asarray(vec, dtype=np.float32)

        rows.append({"name": name, "vector": vec, "rank": data.get(name, 0)})

    print(f"Built dataset with {len(rows)}")
    return pd.DataFrame(rows)

def prepare_xy(df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42):
    df = df.copy()
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype("Int64")
    df = df[df["rank"].between(1, 10)]
    X = np.stack(df["vector"].values).astype(np.float32)
    y = (df["rank"].astype(int).to_numpy() - 1).astype(np.int64)
    names = df["name"].to_numpy()

    X_tr, X_te, y_tr, y_te, _, names_te = train_test_split(X, y, names, test_size=test_size, random_state=random_state, shuffle=True, stratify=y)
    return X_tr, X_te, y_tr, y_te, names_te

class MultiClass(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x


def train(model: nn.Module,
          X_train: np.ndarray, y_train: np.ndarray,
          X_val: np.ndarray,   y_val: np.ndarray,
          loss_fn, optimizer, epochs: int = 20, batch_size: int = 64):

    # Tensors / loaders
    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    best_acc = -np.inf
    best_weights = None

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, total_seen = 0.0, 0, 0

        for xb, yb in tqdm.tqdm(train_loader, desc=f"Epoch {epoch}", mininterval=0):
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total_correct += (logits.argmax(1) == yb).sum().item()
            total_seen += xb.size(0)

        train_loss = total_loss / max(total_seen, 1)
        train_acc = total_correct / max(total_seen, 1)

        model.eval()
        with torch.no_grad():
            val_logits = model(Xv)
            val_loss = loss_fn(val_logits, yv).item()
            val_acc = (val_logits.argmax(1) == yv).float().mean().item()

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch}: train loss {train_loss:.4f}, acc {train_acc*100:.1f}% | "
              f"val loss {val_loss:.4f}, acc {val_acc*100:.1f}%")

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model


def main():
    df = build_dataset()
    X_train, X_test, y_train, y_test, names_test = prepare_xy(df)

    input_dim = X_train.shape[1]
    output_dim = 10

    model = MultiClass(input_dim=input_dim, output_dim=output_dim, hidden_dim=128).to(device)

    class_counts = np.bincount(y_train, minlength=output_dim)
    weights = class_counts.sum() / np.clip(class_counts, 1, None)
    weights = (weights / weights.mean()).astype(np.float32)
    class_weights = torch.tensor(weights, device=device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model = train(model, X_train, y_train, X_test,  y_test, loss_fn, optimizer, epochs=20, batch_size=64)

    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).to(device))
        preds = logits.argmax(1).cpu().numpy()

    y_true_labels = (y_test + 1)
    y_pred_labels = (preds + 1)

    print(f"\nAccuracy: {accuracy_score(y_true_labels, y_pred_labels):.4f}")
    print(classification_report(y_true_labels, y_pred_labels,labels=list(range(1, 11)),digits=3))

    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": 128,
        "output_dim": output_dim,
        "label_offset": 1,
    }, "models/rank_mlp.pt")
    print("\nSaved model to models/rank_mlp.pt")


if __name__ == "__main__":
    main()
