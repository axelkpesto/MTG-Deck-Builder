import torch
import numpy as np
import pandas as pd
import os
import random
from Card_Lib import CardEncoder, CardDecoder, CardFields
from typing import Callable, Iterator, Optional, Tuple, List, Dict

class VectorStore(object):
    def __init__(self, encoder, decoder) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.vector_data: Dict[str, torch.Tensor] = {}
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
    
    def __str__(self) -> str:
        return str(dict(map(lambda kv: (kv[0], self.decoder.decode(kv[0], kv[1])), self.items())))
    
    def __eq__(self, item) -> bool:
        if not isinstance(item, VectorStore): return False
        return self.vector_data.items()==item.vector_data.items()

    def __len__(self) -> int:
        return len(self.vector_data.keys())

    def __iter__(self) -> bool:
        return (x for x in self.vector_data.items())
    
    def __getitem__(self, key) -> torch.Tensor:
        if isinstance(key, str):
            return self.vector_data.get(key)
        elif isinstance(key, int):
            values = list(self.vector_data.keys())
            if 0 <= key < len(values):
                return values[key]
            else:
                raise IndexError("Index out of range")
        else:
            raise TypeError("Invalid key type for subscripting")
    
    def clear(self) -> None:
        self.vector_data = {}
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def items(self) -> List[Tuple[str, np.array]]:
        return self.vector_data.items()
    
    def keys(self) -> List[str]:
        return self.vector_data.keys()
    
    def values(self) -> List[np.array]:
        return self.vector_data.values()

    def setdefault(self, v_id: str, vector: np.array) -> None:
        return self.vector_data.setdefault(v_id, vector)

    def contains(self, value: object) -> bool:
        return value in self.vector_data

    def to_ndarray(self, *, predicate: Optional[Callable[[str, np.ndarray], bool]] = None) -> np.ndarray:
        if predicate:
            return np.array([v.cpu().numpy() for k, v in self.vector_data.items() if predicate(k, v)])
        return np.array([v.cpu().numpy() for v in self.vector_data.values()])
    
    def add_vector(self, v_id: str, vector: np.ndarray) -> None:
        if self.contains(v_id):
            return
        vector_tensor = torch.tensor(vector, dtype=torch.float32).to(self.device)
        self.vector_data[v_id] = vector_tensor

    def get(self, v_id: str, default):
        return self.vector_data[v_id] if v_id in self.vector_data else default
    
    def get_list(self, v_ids: List[str], default) -> List[torch.tensor]:
        return [self.get(v_id, default) for v_id in v_ids]

    def get_vector(self, v_id: str) -> np.ndarray:
        return self.vector_data[v_id].cpu().numpy()

    def get_vector_tup(self, v_id: str) -> Tuple[str, torch.tensor]:
        return (v_id, self.get_vector(v_id))

    def get_random_vector(self) -> Tuple[str, torch.tensor]:
        random_id: str = random.choice(list(self.vector_data.keys()))
        return (random_id, self.get_vector(v_id=random_id))

    def size(self) -> int:
        return len(self.vector_data.keys())

    def get_similar_vectors(self, q_vector: torch.tensor, n_results: int = 5) -> List[Tuple[str, torch.tensor]]:
        q_vector_tensor = torch.tensor(q_vector, dtype=torch.float32).to(self.device)

        results = []
        for vector_id, vector in self.vector_data.items():
            vector_tensor = vector.float().to(self.device)
            similarity = torch.matmul(q_vector_tensor, vector_tensor) / (
                torch.norm(q_vector_tensor) * torch.norm(vector_tensor)
            )
            results.append((vector_id, similarity.item()))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n_results + 1]

    def nearest_by_embedding(self, queries: np.ndarray, candidates: np.ndarray, slice_index: int, topk: int = 8) -> List[np.ndarray]:
        #queries: (S, D), represents S query vectors
        #candidates: (N, D), represents N candidate vectors (often whole vector DB)
        dim = candidates.shape[-1]
        s_emb = slice(slice_index, dim)
        s_head = slice(0, slice_index)

        A = queries[..., s_emb]
        B = candidates[..., s_emb]
        A = A / np.clip(np.linalg.norm(A, axis=-1, keepdims=True), 1e-8, None)
        B = B / np.clip(np.linalg.norm(B, axis=-1, keepdims=True), 1e-8, None)
        sims = A @ B.T  # (S, N)

        ah = queries[..., s_head]
        bh = candidates[..., s_head]

        out = []
        for i in range(queries.shape[0]):
            bonus = -0.02 * np.abs(ah[i] - bh).mean(axis=-1)
            scores = sims[i] + bonus
            idx = np.argpartition(-scores, topk)[:topk]
            idx = idx[np.argsort(-scores[idx])]
            out.append(idx)
        return out

    def find_vector_pair(self, q_id: str) -> Tuple[str, torch.tensor]:
        if q_id in self.vector_data:
            return self.get_vector_tup(q_id)
        else:
            for id, _ in self.vector_data.items():
                if q_id in id:
                    return self.get_vector_tup(id)
            raise KeyError(f"KeyError: {q_id}")

    def find_vector(self, q_id: str) -> torch.tensor:
        return self.find_vector_pair(q_id)[1]

    def find_id(self, q_id: str) -> str:
        return self.find_vector_pair(q_id)[0]

    def describe_vector_string(self, v_id: str) -> str:
        if self.decoder:
            return self.decoder.decode(v_id, self.get_vector(v_id=v_id))
        return f"{v_id}: {self.get_vector(v_id=v_id)}"

    def describe_vector_dict(self, v_id: str) -> Dict[str, str]:
        if self.decoder:
            return self.decoder.decode_to_dict(v_id, self.get_vector(v_id=v_id))
        return {v_id: str(self.get_vector(v_id=v_id))}
    
    def save(self, filename: str) -> None:
        cpu_dict = {k: v.detach().cpu() for k, v in self.vector_data.items()}
        torch.save(cpu_dict, filename)

    def load(self, filename: str) -> None:
        self.clear()
        self.vector_data = torch.load(filename, map_location=self.device, weights_only=False)

        for k, v in list(self.vector_data.items()):
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            self.vector_data[k] = v.to(self.device)

    def filter_iterator(self, predicate: Callable[[str, np.ndarray], bool], *, limit: Optional[int] = None) -> Iterator[Tuple[str, np.ndarray]]:
        count = 0
        for name, vec in self.items():
            try:
                if predicate(name, vec):
                    yield (name, vec)
                    count += 1
                    if limit is not None and count >= limit:
                        return
            except Exception:
                continue

    def filter(self, predicate: Callable[[str, np.ndarray], bool], *, limit: Optional[int] = None, names_only: bool = False, vectors_only: bool = False) -> List:
        out = []
        for name, vec in self.filter_iterator(predicate, limit=limit):
            if names_only:
                out.append(name)
            elif vectors_only:
                out.append(vec)
            else:
                out.append((name, vec))
        return out
    
class VectorDatabase(object):
    def __init__(self, encoder, decoder) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.vector_store: VectorStore = VectorStore(encoder, decoder)

    def __str__(self) -> str:
        return str(self.vector_store)
    
    def __eq__(self, item) -> bool:
        if not isinstance(item, VectorDatabase): return False
        return (self.vector_store == item.vector_store)

    def __len__(self) -> int:
        return len(self.vector_store)

    def __iter__(self) -> bool:
        return (x for x in self.vector_store)
    
    def __getitem__(self, key) -> torch.Tensor:
        if isinstance(key, str):
            return self.vector_store[key]
        elif isinstance(key, int):
            if 0 <= key < len(self.vector_store):
                self.vector_store[key]
            else:
                raise IndexError("Index out of range")
        else:
            raise TypeError("Invalid key type for subscripting")

    def size(self) -> int:
        return len(self.vector_store)

    def clear(self) -> None:
        self.vector_store.clear()

    def items(self) -> List[Tuple[str, np.array]]:
        return self.vector_store.items()
    
    def keys(self) -> List[str]:
        return self.vector_store.keys()
    
    def values(self) -> List[np.array]:
        return self.vector_store.values()

    def setdefault(self, v_id: str, vector: np.array) -> None:
        return self.vector_store.setdefault(v_id, vector)

    def to_ndarray(self, *, predicate: Optional[Callable[[str, np.ndarray], bool]] = None) -> np.ndarray:
        return self.vector_store.to_ndarray(predicate=predicate)

    def parse_json(self, filename: str, max_lines: int = -1) -> VectorStore:
        set_data: pd.DataFrame = self._parse_file(filename)['data'][2:]
        num_cards: int = 0
        
        for game_set in set_data:
            for card in game_set['cards']:
                if 'commander' in card['legalities'] and card['legalities']['commander'] == "Legal" and 'paper' in card['availability']:
                    v_id, vector = self.encoder.encode(CardFields.parse_mtgjson_card(card))
                    self.vector_store.add_vector(v_id, vector)
                    num_cards += 1

                    if max_lines > -1 and num_cards >= max_lines:
                        return self.vector_store
        return self.vector_store

    def contains(self, value: object) -> bool:
        return self.vector_store.contains(value)

    def add_vector(self, v_id: str, vector: np.ndarray) -> None:
        self.vector_store.add_vector(v_id, vector)
    
    def get(self, v_id, default = None) -> torch.tensor:
        return self.vector_store.get(v_id, default)

    def get_list(self, v_ids: List[str], default = None) -> List[torch.tensor]:
        return self.vector_store.get_list(v_ids, default)

    def get_vector(self, v_id: str) -> torch.tensor:
        return self.vector_store.get_vector(v_id)
    
    def get_vector_tup(self, v_id: str) -> Tuple[str, torch.tensor]:
        return self.vector_store.get_vector_tup(v_id)
    
    def get_random_vector(self) -> Tuple[str, torch.tensor]:
        return self.vector_store.get_random_vector()
    
    def get_similar_vectors(self, q_vector: torch.tensor, n_results: int = 5) -> List[Tuple[str, torch.tensor]]:
        return self.vector_store.get_similar_vectors(q_vector, n_results)
    
    def nearest_by_embedding(self, queries: np.ndarray, candidates: np.ndarray, slice: int, topk: int = 8) -> List[np.ndarray]:
        return self.vector_store.nearest_by_embedding(queries, candidates, slice, topk)

    def find_vector_pair(self, v_id: str) -> Tuple[str, torch.tensor]:
        return self.vector_store.find_vector_pair(v_id)
    
    def find_vector(self, v_id: str) -> torch.tensor:
        return self.vector_store.find_vector(v_id)

    def find_id(self, v_id: str) -> str:
        return self.vector_store.find_id(v_id)
    
    def get_vector_description(self, v_id: str) -> str:
        return self.vector_store.describe_vector_string(v_id=v_id)

    def get_vector_description_dict(self, v_id: str) -> Dict[str, str]:
        return self.vector_store.describe_vector_dict(v_id=v_id)

    def _parse_file(self, filename: str) -> pd.DataFrame:
        assert os.path.isfile(filename), f"{filename} not found."
        return pd.read_json(filename)
    
    def save(self, filename: str) -> None:
        self.vector_store.save(filename)

    def load(self, filename: str) -> None:
        self.vector_store.load(filename)

    def filter(self, predicate: Callable[[str, np.ndarray], bool], *, limit: Optional[int] = None, names_only: bool = False, vectors_only: bool = False) -> List:
        return self.vector_store.filter(predicate, limit=limit, names_only=names_only, vectors_only=vectors_only)

if __name__ == "__main__":
    vd = VectorDatabase(CardEncoder(), CardDecoder())
    vd.parse_json(filename="datasets/AllPrintings.json")
    random_vector = vd.get_random_vector()

    print(random_vector)
    print()
    print(vd.get_vector_description(random_vector[0]))

    similar_vectors = vd.get_similar_vectors(random_vector[1])

    print(vd.find_id("Horus"))
    print(vd.find_id("Magnus"))
    print(vd.find_vector_pair("Abaddon"))

    vd.save("datasets/vector_data.pt")