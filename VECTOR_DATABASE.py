from CARD_DATA import Card, CardEncoder, CardDecoder
import cupy as cp
import numpy as np
import pandas as pd
import time, os, random

class VectorStore(object):
    def __init__(self):
        self.encoder = CardEncoder()
        self.decoder = CardDecoder()
        self.vector_data: dict = {}
        self.vector_indexes: dict = {}

    def contains(self, value: object) -> bool:
        if isinstance(value, Card):
            return value.card_name in self.vector_data
        elif isinstance(value, str):
            return value in self.vector_data
        return False

    def add_card(self, crd: Card, runtime: bool = False) -> None:
        if self.contains(crd):
            return
        crd_tuple = self.encoder.encode(crd)
        self.add_vector(crd_tuple[0], crd_tuple[1], runtime=runtime)

    def add_vector(self, v_id: str, vector: np.array, runtime=False) -> None:
        if self.contains(v_id):
            return
        self.vector_data[v_id] = vector
        self._update_index(v_id, vector, runtime=runtime)

    def get_vector(self, v_id: str) -> np.array:
        return self.vector_data.get(v_id, None)
    
    def get_random_vector(self) -> tuple[str, np.array]:
        random_id: str = random.choice(list(self.vector_data.keys()))
        return (random_id, self.get_vector(v_id=random_id))
    
    def size(self) -> int:
        return len(self.vector_data.keys())
    
    def _update_index(self, v_id: str, vector: np.array, batch_size: int = 1000, runtime: bool = False) -> None:
        start = time.time()
        
        query_vector = cp.asarray(vector)
        query_norm = cp.linalg.norm(query_vector)
        
        for i in range(0, len(self.vector_data), batch_size):
            
            all_vectors = cp.array(list(self.vector_data.values())[i:i + batch_size])
            dot_products = cp.dot(all_vectors, query_vector)
            similarities = cp.asnumpy(dot_products / (cp.linalg.norm(all_vectors, axis=1) * query_norm))
            
            for idx, (vector_id, _) in enumerate(self.vector_data.items()):                
                if i + idx < len(self.vector_data) and idx < len(similarities):
                    if vector_id not in self.vector_indexes:
                        self.vector_indexes[vector_id] = {}
                    self.vector_indexes[vector_id][v_id] = similarities[idx]

        if runtime: print("RUNTIME OF UPDATING INDEX: " + str(time.time() - start) + ", INDEX: " + str(len(self.vector_data.items())))

    def get_similar_vectors(self, q_vector: np.array, n_results: int = 5) -> list[tuple[str, np.array]]:
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
    
    def describe_vector_string(self, v_id: str) -> str:
        return self.decoder.decode_to_string(v_id,self.get_vector(v_id=v_id))

    def describe_vector_dict(self, v_id: str) -> dict:
        return self.decoder.decode_to_dict(v_id,self.get_vector(v_id=v_id))
    
class VectorDatabase(object):
    def __init__(self, RUNTIME=False):
        self.RUNTIME = RUNTIME
        self.vector_store = VectorStore()

    def parse_json(self, filename:str, runtime:bool = False, max_lines:int = None) -> VectorStore:
        assert os.path.isfile(filename), f"{filename} not found."
        

        set_data = self._parse_file(filename=filename, runtime=runtime)
        
        start_time: float = time.time()
        num_cards: int = 0

        for game_set in set_data:
            for card in game_set['cards']:
                if 'commander' in card['legalities'] and card['legalities']['commander']=="Legal" and 'paper' in card['availability']:
                    self.vector_store.add_card(Card(card), runtime=self.RUNTIME)
                    num_cards += 1

                    if max_lines is not None and num_cards>=max_lines:
                        if runtime: print("BUILDING DATABASE: " + str(time.time()-start_time))
                        return self.vector_store

        if runtime: print("BUILDING DATABASE: " + str(time.time()-start_time))
        return self.vector_store

    def contains(self, value: object) -> bool:
        return self.vector_store.contains(value)

    def add_card(self, crd:Card) -> None:
        self.vector_store.add_card(crd)

    def add_vector(self, v_id: str, vector: np.array) -> None:
        self.vector_store.add_vector(v_id,vector)

    def get_vector(self, v_id: str) -> np.array:
        return self.vector_store.get_vector(v_id)
    
    def get_random_vector(self) -> tuple[str, np.array]:
        return self.vector_store.get_random_vector()

    def get_similar_vectors(self, q_vector:np.array, n_results:int = 5) -> list[tuple[str, np.array]]:
        return self.vector_store.get_similar_vectors(q_vector,n_results)
    
    def get_vector_description(self, v_id: str) -> str:
        return self.vector_store.describe_vector_string(v_id=v_id)
    
    def get_vector_description_dict(self, v_id: str) -> dict:
        return self.vector_store.describe_vector_dict(v_id=v_id)
    

    def _parse_file(self, filename:str, runtime:bool = False) -> pd.DataFrame:
        start_time = time.time()
        set_data: pd.DataFrame = pd.read_json(filename)['data'][2:]
        if runtime: print("PARSING DATASET: " + str(time.time()-start_time))
        return set_data

if __name__ == "__main__":
    vd = VectorDatabase(False)
    vd.parse_json(filename="AllPrintings.json", runtime=True, max_lines=2500)
    random_vector: tuple[str, np.array] = vd.get_random_vector()
    
    print(random_vector)
    print()
    print(vd.get_vector_description(random_vector[0]))

    print()
    similar_vectors = vd.get_similar_vectors(random_vector[1])
    for vector in similar_vectors:
        print("\n" + vd.get_vector_description(vector[0]))
