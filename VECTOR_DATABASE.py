from CARD_DATA import Card, CardEncoder, CardDecoder
import cupy as cp
import numpy as np
import pandas as pd
import time, os, random

class VectorStore(object):
    def __init__(self):
        """
        Responsible for initializing the VectorStore object

        Parameters: 
            None
        """
        self.encoder = CardEncoder()
        self.decoder = CardDecoder()
        self.vector_data: dict = {}
        self.vector_indexes: dict = {}

    
    def contains(self, value: object) -> bool:
        """
        Checking if Value is contained in Vector Database

        Parameters: 
            value: Object
                Object to be checked for inclusion in Database
        
        Output:
            bool
                Whether the object is in the Database
        """
        if isinstance(value, Card):
            return value.card_name in self.vector_data
        elif isinstance(value, str):
            return value in self.vector_data
        return False

    def add_card(self, crd: Card, runtime: bool = False) -> None:
        """
        Adding Card to Vector Database through encoding

        Parameters: 
            crd: Card
                Card to be encoded and added to Vector Database
            runtime: bool
                Whether the runtime of adding the data should be displayed
        
        Output:
            None
        """
        if self.contains(crd):
            return
        crd_tuple = self.encoder.encode(crd)
        self.add_vector(crd_tuple[0], crd_tuple[1], runtime=runtime)

    def add_vector(self, v_id: str, vector: np.array, runtime=False) -> None:
        """
        Adding Card to Vector Database given a vector ID and the corresponding encoded vector.

        Parameters: 
            v_id: str
                ID for the vector
            vector: np.array
                Corresponding vector (np.array) to ID
            runtime: bool
                Whether the runtime of adding the data should be displayed
        
        Output:
            None
        """
        if self.contains(v_id):
            return
        self.vector_data[v_id] = vector
        self._update_index(v_id, vector, runtime=runtime)

    def get_vector(self, v_id: str) -> np.array:
        """
        Get vector that corresponds to a certain ID

        Parameters: 
            v_id: str
                ID for the vector
        
        Output:
            np.array or None
                np.array if ID is in the database, else None
        """
        return self.vector_data.get(v_id, None)
    
    def get_random_vector(self) -> tuple[str, np.array]:
        """
        Get a random (ID, Vector) pair from Database

        Parameters: 
            None
        
        Output:
            tuple[str, np.array]
                A random (ID, Vector) pair from the DataBase
        """
        random_id: str = random.choice(list(self.vector_data.keys()))
        return (random_id, self.get_vector(v_id=random_id))
    
    def size(self) -> int:
        """
        Get the current size of the database

        Parameters: 
            None
        
        Output:
            int
                Number of keys in the database
        """
        return len(self.vector_data.keys())
    
    def _update_index(self, v_id: str, vector: np.array, batch_size: int = 1000, runtime: bool = False) -> None:
        """
        Updates the stored similarity indexes for given vector

        Parameters: 
            v_id: str
                ID of Vector
            vector: np.array
                Corresponding vector to ID
            batch_size: int
                How many vectors are processed in each iteration (for optimization)
            runtime: bool
                Whether the runtime of updating vector similarities should be displayed
        
        Output:
            None
        """
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
        """
        Updates the stored similarity indexes for given vector

        Parameters: 
            q_vector: np.array
                Query vector
            n_results: int
                Number of similar vectors
        
        Output:
            list[tuple[str, np.array]]
                list of (ID, Vector) pairs
        """
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
        """
        String representation of given vector

        Parameters: 
            v_id: str
                ID of Vector
        
        Output:
            str
                String representation of vector
        """
        return self.decoder.decode_to_string(v_id,self.get_vector(v_id=v_id))

    def describe_vector_dict(self, v_id: str) -> dict:
        """
        String representation of given vector

        Parameters: 
            v_id: str
                ID of Vector
        
        Output:
            dict
                Dictionary representation of vector
        """
        return self.decoder.decode_to_dict(v_id,self.get_vector(v_id=v_id))
    
class VectorDatabase(object):
    def __init__(self, RUNTIME: bool = False):
        """
        Responsible for initializing the VectorDatabase object

        Parameters: 
            RUNTIME: bool
                Whether runtimes of functions should be displayed
        """
        self.RUNTIME = RUNTIME
        self.vector_store = VectorStore()

    def parse_json(self, filename:str, runtime:bool = False, max_lines:int = None) -> VectorStore:
        """
        Build DataBase from JSON

        Parameters: 
            filename: str
                Name of JSON file you want passed in
            runtime: bool
                Whether runtimes of function should be displayed
            max_lines: int
                Maximum number of lines in DataBase
        
        Output:
            VectorStore:
                VectorStore with first (max_lines) of JSON
        """
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
        """
        Checking if Value is contained in Vector Database

        Parameters: 
            value: Object
                Object to be checked for inclusion in Database
        
        Output:
            bool
                Whether the object is in the Database
        """
        return self.vector_store.contains(value)

    def add_card(self, crd:Card) -> None:
        """
        Adding Card to VectorStore

        Parameters: 
            crd: Card
                Card to be encoded and added to Vector Database
        
        Output:
            None
        """
        self.vector_store.add_card(crd)

    def add_vector(self, v_id: str, vector: np.array) -> None:
        """
        Adding Card to VectorStore

        Parameters: 
            v_id: str
                ID of Vector
            vector: np.array
                Corresponding Vector to ID
        
        Output:
            None
        """
        self.vector_store.add_vector(v_id,vector)

    def get_vector(self, v_id: str) -> np.array:
        return self.vector_store.get_vector(v_id)
    
    def get_random_vector(self) -> tuple[str, np.array]:
        """
        Get a random (ID, Vector) pair from Database

        Parameters: 
            None
        
        Output:
            tuple[str, np.array]
                A random (ID, Vector) pair from the DataBase
        """
        return self.vector_store.get_random_vector()

    def get_similar_vectors(self, q_vector: np.array, n_results: int = 5) -> list[tuple[str, np.array]]:
        """
        Updates the stored similarity indexes for given vector

        Parameters: 
            q_vector: np.array
                Query vector
            n_results: int
                Number of similar vectors
        
        Output:
            list[tuple[str, np.array]]
                list of (ID, Vector) pairs
        """
        return self.vector_store.get_similar_vectors(q_vector,n_results)
    
    def get_vector_description(self, v_id: str) -> str:
        """
        String representation of given vector

        Parameters: 
            v_id: str
                ID of Vector
        
        Output:
            str
                String representation of vector
        """
        return self.vector_store.describe_vector_string(v_id=v_id)
    
    def get_vector_description_dict(self, v_id: str) -> dict:
        """
        String representation of given vector

        Parameters: 
            v_id: str
                ID of Vector
        
        Output:
            dict
                Dictionary representation of vector
        """
        return self.vector_store.describe_vector_dict(v_id=v_id)

    def _parse_file(self, filename:str, runtime:bool = False) -> pd.DataFrame:
        """
        Private file parsing function for readability

        Parameters: 
            filename: str
                Name of JSON file to be parsed
            runtime: bool
                Whether the runtime of adding the data should be displayed
        
        Output:
            pd.DataFrame
                Pandas Dataframe containing data parsed from JSON
        """
        start_time = time.time()
        set_data: pd.DataFrame = pd.read_json(filename)['data'][2:]
        if runtime: print("PARSING DATASET: " + str(time.time()-start_time))
        return set_data

