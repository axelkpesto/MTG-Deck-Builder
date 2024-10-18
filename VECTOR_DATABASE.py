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

    #Add a card to db
    def add_card(self, crd: Card) -> None:
        if self.contains(crd): #Skip if Card.name is in db
            return
        crd_tuple = self.encoder.encode(crd) #Get vector representation of Card (Tuple of name and vector)
        self.add_vector(crd_tuple[0], crd_tuple[1]) #Add Vector ->

    #Add a vector to db
    def add_vector(self, v_id: str, vector: np.array) -> None:
        if self.contains(v_id): #Skip if id in db
            return
        self.vector_data[v_id] = vector #Set Vector Data
        self._update_index(v_id, vector, runtime=True) #Update Similarities

    def get_vector(self, v_id: str) -> np.array:
        return self.vector_data.get(v_id, None)


    #Updates the similarity relations between query vector and all other vectors in db
    def _update_index(self, v_id: str, vector: np.array, runtime: bool = False, batch_size: int = 1000) -> None:
        start = time.time() #Variable for tracking time it takes to query similarities
        
        query_vector = cp.asarray(vector) #Query vector to CuPy array
        query_norm = cp.linalg.norm(query_vector) #normalized query vector to cupy
        
        #Seperate queries into batches for batch processing. Not sure if this actively does any major changes.
        for i in range(0, len(self.vector_data), batch_size):
            
            #array of our batch of vectors from our db
            all_vectors = cp.array(list(self.vector_data.values())[i:i + batch_size])
            
            #CuPy array of all dot product relations
            dot_products = cp.dot(all_vectors, query_vector)

            #CuPy array of all Cosine Similarities between query vector and db items
            similarities = cp.asnumpy(dot_products / (cp.linalg.norm(all_vectors, axis=1) * query_norm))
            
            #Update DB to save similarities
            for idx, (vector_id, _) in enumerate(self.vector_data.items()):                
                if i + idx < len(self.vector_data) and idx < len(similarities):
                    if vector_id not in self.vector_indexes:
                        self.vector_indexes[vector_id] = {}
                    self.vector_indexes[vector_id][v_id] = similarities[idx]

        #Print stuff if parameter met
        if runtime:
            print("RUNTIME OF UPDATING INDEX: " + str(time.time() - start) + ", INDEX: " + str(len(self.vector_data.items())))



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
        self.parse_json(filename="AllPrintings.json",runtime=True)

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
    
