from Card import Card, CardEncoder
import cupy as cp
import numpy as np
import pandas as pd
import time, os

class VectorStore(object):
    def __init__(self):
        self.encoder = CardEncoder()
        self.vector_data: dict = {}
        self.vector_indexes: dict = {}

    def contains(self, value: object) -> bool:
        if isinstance(value, Card):
            return value.card_name in self.vector_data
        elif isinstance(value, str):
            return value in self.vector_data
        return False

    def add_card(self, crd: Card) -> None:
        if self.contains(crd):
            return
        crd_tuple = self.encoder.encode(crd)
        self.add_vector(crd_tuple[0], crd_tuple[1])

    def add_vector(self, v_id: str, vector: np.array) -> None:
        if self.contains(v_id):
            return
        self.vector_data[v_id] = vector
        self._update_index(v_id, vector)

    def get_vector(self, v_id: str) -> np.array:
        return self.vector_data.get(v_id, None)

    def _update_index(self, v_id: str, vector: np.array, runtime:bool = False) -> None:
            start = time.time()

            query_vector_gpu = cp.asarray(vector)
            all_vectors = cp.array(list(self.vector_data.values()))
            dot_products = cp.dot(all_vectors, query_vector_gpu)

            similarities_cpu = cp.asnumpy(dot_products / (cp.linalg.norm(all_vectors, axis=1) * cp.linalg.norm(query_vector_gpu)))

            for idx, (vector_id, _) in enumerate(self.vector_data.items()):
                if vector_id not in self.vector_indexes:
                    self.vector_indexes[vector_id] = {}
                self.vector_indexes[vector_id][v_id] = similarities_cpu[idx]

            if runtime: print("RUNTIME OF UPDATING INDEX: " + str(time.time() - start) + ", INDEX: " + str(len(self.vector_data.items())))

    def get_similar_vectors(self, q_vector: np.array, n_results: int = 5) -> list[tuple]:
        q_vector_gpu = cp.asarray(q_vector)

        results = []
        for vector_id, vector in self.vector_data.items():
            vector_gpu = cp.asarray(vector)
            similarity = cp.dot(q_vector_gpu, vector_gpu) / (
                cp.linalg.norm(q_vector_gpu) * cp.linalg.norm(vector_gpu)
            )
            results.append((vector_id, similarity.get()))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n_results]

class VectorDatabase(object):
    def __init__(self):
        self.vector_store = VectorStore()
        self.parse_json("AllPrintings.json")

    def parse_json(self, filename:str, runtime:bool = False) -> VectorStore:
        assert os.path.isfile(filename), f"{filename} not found."
        start_time = time.time()
        
        set_data = pd.read_json(filename)['data'][2:]
        
        if runtime: print("PARSING DATASET: " + str(time.time()-start_time))
        
        for game_set in set_data:
            for card in game_set['cards']:
                if 'commander' in card['legalities'] and card['legalities']['commander']=="Legal" and 'paper' in card['availability']:
                    self.vector_store.add_card(Card(card))

        return self.vector_store

    def contains(self, value: object) -> bool:
        return self.vector_store.contains(value)

    def add_card(self, crd:Card) -> None:
        self.vector_store.add_card(crd)

    def add_vector(self, v_id:str, vector:np.array) -> None:
        self.vector_store.add_vector(v_id,vector)

    def get_vector(self, v_id:str) -> np.array:
        return self.vector_store.get_vector(v_id)

    def get_similar_vectors(self, q_vector:np.array, n_results:int = 5) -> list[tuple]:
        return self.vector_store.get_similar_vectors(q_vector,n_results)

if __name__ == "__main__":
    vd = VectorDatabase()
    vd.get_vector("Murder")