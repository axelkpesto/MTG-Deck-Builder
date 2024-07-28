import numpy as np
# from Card import Card

class VectorStore(object):
    def __init__(self):
        self.vector_data:dict = {}
        self.vector_indexes:dict = {}

    def add_vector(self, v_id:str, vector:np.ndarray) -> None:
        self.vector_data[v_id] = vector
        self._update_index(v_id, vector)

    def get_vector(self, v_id:str) -> np.ndarray:
        return self.vector_data.get(v_id, None)

    def _update_index(self, v_id:str, vector:np.ndarray) -> None:
        for id, vct in self.vector_data.items():
            similarity = np.dot(vector, vct) / (np.linalg.norm(vector) * np.linalg.norm(vct))
            if id not in self.vector_indexes:
                self.vector_indexes[id] = {}
            self.vector_indexes[id][v_id] = similarity

    def get_similar_vectors(self, q_vector:np.ndarray, n_results:int = 5) -> list[tuple]:
        results = []
        for vector_id, vector in self.vector_data.items():
            similarity = np.dot(q_vector, vector) / (np.linalg.norm(q_vector) * np.linalg.norm(vector))
            results.append((vector_id, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n_results]