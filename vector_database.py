"""Vector storage and search helpers for encoded MTG cards."""

import os
import random
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from card_data import CardFields
from config import CONFIG

class VectorStore:
    """In-memory vector store with similarity search and serialization helpers."""

    def __init__(self, encoder, decoder) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.vector_data: Dict[str, torch.Tensor] = {}
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)

        self._cache_dirty: bool = True
        self._cache_ids: List[str] = []
        self._cache_matrix: Optional[torch.Tensor] = None  # (N, D) normalized float32

    def __str__(self) -> str:
        return str(dict(map(lambda kv: (kv[0], self.decoder.decode(kv[0], kv[1])), self.items())))

    def __eq__(self, item) -> bool:
        if not isinstance(item, VectorStore):
            return False
        return self.vector_data.items() == item.vector_data.items()

    def __hash__(self) -> int:
        return hash(frozenset(self.vector_data.items()))

    def __contains__(self, key) -> bool:
        return key in self.vector_data

    def __len__(self) -> int:
        return len(self.vector_data.keys())

    def __iter__(self) -> Iterator[Tuple[str, torch.Tensor]]:
        return iter(self.vector_data.items())

    def __getitem__(self, key) -> torch.Tensor:
        if isinstance(key, str):
            return self.vector_data[key]
        if isinstance(key, int):
            values = list(self.vector_data.keys())
            if 0 <= key < len(values):
                return self.vector_data[values[key]]
            raise IndexError("Index out of range")
        raise TypeError("Invalid key type for subscripting")

    def _mark_cache(self, dirty: bool = True) -> None:
        self._cache_dirty = dirty

    def _l2_normalize(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        denom = torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=eps)
        return x / denom

    def _rebuild_similarity_cache(self) -> None:
        self._cache_ids = list(self.vector_data.keys())
        if len(self._cache_ids) == 0:
            self._cache_matrix = None
            self._mark_cache(dirty=False)
            return

        mat = torch.stack(
            [self.vector_data[k].float().to(self.device) for k in self._cache_ids],
            dim=0,
        )

        mat = self._l2_normalize(mat)
        self._cache_matrix = mat
        self._mark_cache(dirty=False)

    def clear(self) -> None:
        """Remove all vectors and clear cached similarity state."""
        self.vector_data = {}
        self._mark_cache(dirty=True)
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def items(self) -> List[Tuple[str, np.array]]:
        """Return all `(id, vector)` items."""
        return list(self.vector_data.items())

    def keys(self) -> List[str]:
        """Return all vector ids."""
        return list(self.vector_data.keys())

    def values(self) -> List[np.array]:
        """Return all vectors."""
        return list(self.vector_data.values())

    def setdefault(self, v_id: str, vector: np.array) -> None:
        """Set `v_id` to `vector` if absent and return stored value."""
        self._mark_cache(dirty=True)
        return self.vector_data.setdefault(v_id, vector)

    def contains(self, value: object) -> bool:
        """Return whether an id exists in the store."""
        return value in self.vector_data

    def to_ndarray(self, *, predicate: Optional[Callable[[str, np.ndarray], bool]] = None) -> np.ndarray:
        """Return vectors as an ndarray, optionally filtered by predicate."""
        if predicate:
            return np.array(
                [v.cpu().numpy() for k, v in self.vector_data.items() if predicate(k, v)]
            )
        return np.array([v.cpu().numpy() for v in self.vector_data.values()])

    def to_dataframe(self, *, predicate: Optional[Callable[[str, np.ndarray], bool]] = None) -> pd.DataFrame:
        """Return vectors as a dataframe, optionally filtered by predicate."""
        if predicate:
            return pd.DataFrame(
                dict(
                    (k, v.cpu().numpy())
                    for k, v in self.vector_data.items()
                    if predicate(k, v)
                )
            )
        return pd.DataFrame(dict((k, v.cpu().numpy()) for k, v in self.vector_data.items()))

    def add_vector(self, v_id: str, vector: np.ndarray) -> None:
        """Add a vector by id if it does not already exist."""
        assert isinstance(v_id, str)
        if self.contains(v_id):
            return
        vector_tensor = torch.from_numpy(vector).float().to(self.device)
        self.vector_data[v_id] = vector_tensor
        self._mark_cache(dirty=True)

    def get(self, v_id: str, default: Any) -> torch.Tensor:
        """Get vector by id or return `default` if missing."""
        return self.vector_data[v_id] if v_id in self.vector_data else default

    def get_list(self, v_ids: List[str], default: Any) -> List[torch.Tensor]:
        """Get vectors for a list of ids with fallback default."""
        return [self.get(v_id, default) for v_id in v_ids]

    def get_vector(self, v_id: str) -> torch.Tensor:
        """Get vector by id."""
        return self.vector_data[v_id]

    def get_vector_tup(self, v_id: str) -> Tuple[str, torch.Tensor]:
        """Get `(id, vector)` tuple by id."""
        return v_id, self.get_vector(v_id)

    def get_random_vector(self) -> Tuple[str, torch.Tensor]:
        """Get a random `(id, vector)` pair."""
        random_id: str = random.choice(list(self.vector_data.keys()))
        return random_id, self.get_vector(v_id=random_id)

    def size(self) -> int:
        """Return number of vectors stored."""
        return len(self.vector_data)

    def get_similar_vectors(self, q_vector: torch.Tensor, n_results: int = 5) -> List[Tuple[str, float]]:
        """Return top cosine-similar vectors to query vector."""
        q = q_vector.detach() if isinstance(q_vector, torch.Tensor) else torch.tensor(q_vector)
        q = q.float().to(self.device)
        q = self._l2_normalize(q)

        if self._cache_dirty or self._cache_matrix is None:
            self._rebuild_similarity_cache()

        if self._cache_matrix is None or len(self._cache_ids) == 0:
            return []

        scores = torch.mv(self._cache_matrix, q)

        k = min(int(n_results) + 1, scores.shape[0])
        top_scores, top_idx = torch.topk(scores, k=k, largest=True, sorted=True)
        top_scores = top_scores.detach().cpu().numpy().tolist()
        top_idx = top_idx.detach().cpu().numpy().tolist()
        return [(self._cache_ids[i], float(s)) for i, s in zip(top_idx, top_scores)]

    def find_vector_pair(self, q_id: str) -> Tuple[str, torch.Tensor]:
        """Find vector by exact/partial id and return `(id, vector)`."""
        if q_id in self.vector_data:
            return self.get_vector_tup(q_id)
        for candidate_id, _ in self.vector_data.items():
            if q_id in candidate_id:
                return self.get_vector_tup(candidate_id)
        raise KeyError(f"KeyError: {q_id}")

    def find_vector(self, q_id: str) -> torch.Tensor:
        """Find vector by exact/partial id."""
        return self.find_vector_pair(q_id)[1]

    def find_id(self, q_id: str) -> str:
        """Find canonical id by exact/partial id match."""
        return self.find_vector_pair(q_id)[0]

    def describe_vector_string(self, v_id: str) -> str:
        """Return decoded vector description as text."""
        if self.decoder:
            return self.decoder.decode(v_id, self.get_vector(v_id=v_id))
        return f"{v_id}: {self.get_vector(v_id=v_id)}"

    def describe_vector_dict(self, v_id: str) -> Dict[str, str]:
        """Return decoded vector description as a dictionary."""
        if self.decoder:
            return self.decoder.decode_to_dict(v_id, self.get_vector(v_id=v_id))
        return {v_id: str(self.get_vector(v_id=v_id))}

    def save(self, filename: str) -> None:
        """Persist vectors to disk."""
        cpu_dict = {k: v.detach().cpu() for k, v in self.vector_data.items()}
        torch.save(cpu_dict, filename)

    def load(self, filename: str) -> None:
        """Load vectors from disk to current device."""
        self.clear()
        self.vector_data = torch.load(filename, map_location=self.device, weights_only=False)

        for k, v in list(self.vector_data.items()):
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            self.vector_data[k] = v.to(self.device)

        self._mark_cache(dirty=True)

    def filter_iterator(self, predicate: Callable[[str, np.ndarray], bool], *, limit: Optional[int] = None) -> Iterator[Tuple[str, np.ndarray]]:
        """Yield filtered `(id, vector)` pairs, optionally limited."""
        count = 0
        for name, vec in self.items():
            try:
                if predicate(name, vec):
                    yield (name, vec)
                    count += 1
                    if limit is not None and count >= limit:
                        return
            except (TypeError, ValueError, KeyError, AttributeError):
                continue

    def filter(self, predicate: Callable[[str, np.ndarray], bool], *, limit: Optional[int] = None, names_only: bool = False, vectors_only: bool = False) -> List:
        """Return filtered ids/vectors/pairs based on flags."""
        out = []
        for name, vec in self.filter_iterator(predicate, limit=limit):
            if names_only:
                out.append(name)
            elif vectors_only:
                out.append(vec)
            else:
                out.append((name, vec))
        return out


class VectorDatabase:
    """Facade over `VectorStore` plus dataset parsing helpers."""

    def __init__(self, encoder, decoder) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.vector_store: VectorStore = VectorStore(encoder, decoder)

    def __str__(self) -> str:
        return str(self.vector_store)

    def __eq__(self, item) -> bool:
        if not isinstance(item, VectorDatabase):
            return False
        return self.vector_store == item.vector_store

    def __hash__(self) -> int:
        return hash(self.vector_store)

    def __contains__(self, key) -> bool:
        return key in self.vector_store

    def __len__(self) -> int:
        return len(self.vector_store)

    def __iter__(self) -> Iterator[Tuple[str, torch.Tensor]]:
        return iter(self.vector_store)

    def __getitem__(self, key) -> torch.Tensor:
        if isinstance(key, str):
            return self.vector_store[key]
        if isinstance(key, int):
            if 0 <= key < len(self.vector_store):
                return self.vector_store[key]
            raise IndexError("Index out of range")
        raise TypeError("Invalid key type for subscripting")

    def size(self) -> int:
        """Return number of vectors stored."""
        return len(self.vector_store)

    def clear(self) -> None:
        """Clear all stored vectors."""
        self.vector_store.clear()

    def items(self) -> List[Tuple[str, np.array]]:
        """Return all `(id, vector)` items."""
        return self.vector_store.items()

    def keys(self) -> List[str]:
        """Return all vector ids."""
        return self.vector_store.keys()

    def values(self) -> List[np.array]:
        """Return all vectors."""
        return self.vector_store.values()

    def setdefault(self, v_id: str, vector: np.array) -> None:
        """Set default vector for id if missing."""
        return self.vector_store.setdefault(v_id, vector)

    def to_ndarray(self, *, predicate: Optional[Callable[[str, np.ndarray], bool]] = None) -> np.ndarray:
        """Return vectors as an ndarray, optionally filtered."""
        return self.vector_store.to_ndarray(predicate=predicate)

    def to_dataframe(self, *, predicate: Optional[Callable[[str, np.ndarray], bool]] = None) -> pd.DataFrame:
        """Return vectors as a dataframe, optionally filtered."""
        return self.vector_store.to_dataframe(predicate=predicate)

    def parse_json(self, filename: str, max_lines: int = -1) -> None:
        """Parse MTGJSON source and append commander-legal vectors."""
        set_data: pd.DataFrame = self._parse_file(filename)['data'][2:]
        num_cards: int = 0

        for game_set in set_data:
            for card in game_set['cards']:
                if ("commander" in card["legalities"] and card["legalities"]["commander"] == "Legal" and "paper" in card["availability"]):
                    card = CardFields.parse_mtgjson_card(card)
                    v_id, vector = self.encoder.encode(card)
                    self.vector_store.add_vector(v_id, vector)
                    num_cards += 1

                    if 0 <= max_lines <= num_cards:
                        return None
        return None

    def contains(self, value: object) -> bool:
        """Return whether an id exists in the store."""
        return self.vector_store.contains(value)

    def add_vector(self, v_id: str, vector: np.ndarray) -> None:
        """Add one vector to the store."""
        self.vector_store.add_vector(v_id, vector)

    def get_encoder(self):
        """Return encoder instance."""
        return self.encoder

    def get_decoder(self):
        """Return decoder instance."""
        return self.decoder

    def get(self, v_id: str, default: Any = None) -> torch.Tensor:
        """Get vector by id or fallback default."""
        return self.vector_store.get(v_id, default)

    def get_list(self, v_ids: List[str], default: Any = None) -> List[torch.Tensor]:
        """Get vectors for a list of ids."""
        return self.vector_store.get_list(v_ids, default)

    def get_vector(self, v_id: str) -> torch.Tensor:
        """Get vector by id."""
        return self.vector_store.get_vector(v_id)

    def get_vector_tup(self, v_id: str) -> Tuple[str, torch.Tensor]:
        """Get `(id, vector)` tuple by id."""
        return self.vector_store.get_vector_tup(v_id)

    def get_random_vector(self) -> Tuple[str, torch.Tensor]:
        """Get random `(id, vector)` pair."""
        return self.vector_store.get_random_vector()

    def get_similar_vectors(self, q_vector: torch.Tensor, n_results: int = 5) -> List[Tuple[str, float]]:
        """Return nearest vectors for a query vector."""
        return self.vector_store.get_similar_vectors(q_vector, n_results)

    def find_vector_pair(self, v_id: str) -> Tuple[str, torch.Tensor]:
        """Find vector by exact/partial id and return `(id, vector)`."""
        return self.vector_store.find_vector_pair(v_id)

    def find_vector(self, v_id: str) -> torch.Tensor:
        """Find vector by exact/partial id."""
        return self.vector_store.find_vector(v_id)

    def find_id(self, v_id: str) -> str:
        """Find canonical id by exact/partial id."""
        return self.vector_store.find_id(v_id)

    def get_vector_description(self, v_id: str) -> str:
        """Get decoded vector description as string."""
        return self.vector_store.describe_vector_string(v_id=v_id)

    def get_vector_description_dict(self, v_id: str) -> Dict[str, str]:
        """Get decoded vector description as dictionary."""
        return self.vector_store.describe_vector_dict(v_id=v_id)

    def _parse_file(self, filename: str) -> pd.DataFrame:
        """Read source JSON file into dataframe."""
        assert os.path.isfile(filename), f"{filename} not found."
        return pd.read_json(filename)

    def save(self, filename: str) -> None:
        """Persist vector store to disk."""
        self.vector_store.save(filename)

    def load(self, filename: str) -> None:
        """Load vector store from disk."""
        self.vector_store.load(filename)

    @staticmethod
    def load_static(filename: str, encoder=None, decoder=None) -> "VectorDatabase":
        """Instantiate and load a `VectorDatabase` from disk."""
        vector_db = VectorDatabase(encoder, decoder)
        vector_db.load(filename)
        return vector_db

    def filter(self, predicate: Callable[[str, np.ndarray], bool], *, limit: Optional[int] = None, names_only: bool = False, vectors_only: bool = False) -> List:
        """Filter vectors using a predicate and output mode flags."""
        return self.vector_store.filter(
            predicate,
            limit=limit,
            names_only=names_only,
            vectors_only=vectors_only,
        )

    def to_index(self) -> Dict[str, int]:
        """Return mapping from vector id to positional index."""
        return {k: i for i, k in enumerate(self.vector_store.keys())}

    @staticmethod
    def vector_to_numpy(vec: Any) -> np.ndarray:
        """Convert tensor-like vectors to a float32 NumPy array."""
        if hasattr(vec, "detach"):
            return vec.detach().cpu().numpy()
        if hasattr(vec, "cpu"):
            return vec.cpu().numpy()
        return np.asarray(vec, dtype=np.float32)


if __name__ == "__main__":
    # vd = VectorDatabase(CardEncoder(), CardDecoder())
    # vd.parse_json(filename = CONFIG.datasets["FULL_DATASET_PATH"])
    vd = VectorDatabase.load_static(CONFIG.datasets["VECTOR_DATABASE_PATH"])
    random_vector = vd.get_random_vector()

    print(random_vector)
    print()
    print(vd.get_vector_description(random_vector[0]))

    similar_vectors = vd.get_similar_vectors(random_vector[1])

    print(vd.find_id("Horus"))
    print(vd.find_id("Magnus"))
    print(vd.find_vector_pair("Abaddon"))
    print(vd.get_vector_description(vd.find_id("Mishra, Claimed by Gix")))

    # vd.save(CONFIG.datasets["VECTOR_DATABASE_PATH"])
