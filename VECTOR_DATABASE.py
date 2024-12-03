from CARD_DATA import Card, CardEncoder, CardDecoder
import cupy as cp
import numpy as np
import pandas as pd
import time, os, random, boto3, copy

class _VectorStore(object):
    def __init__(self, vector_data: dict = None, vector_indixes: dict = None, optimized: bool = False):
        """
        Responsible for initializing the VectorStore object
        """
        self.encoder: CardEncoder = CardEncoder()
        self.decoder: CardDecoder = CardDecoder()
        self.vector_data: dict = copy.deepcopy(vector_data) if vector_data is not None else {}
        self.vector_indexes: dict = copy.deepcopy(vector_indixes) if vector_indixes is not None else {}
        self.optimized: bool = optimized
    
    def __str__(self) -> str:
        return str(dict(map(lambda kv: (kv[0], self.decoder.decode_to_string(kv[0],kv[1])), self.items())))
    
    def __eq__(self, item) -> bool:
        if not isinstance(item, _VectorStore): return False
        return self.vector_data.items()==item.vector_data.items()

    def __len__(self) -> int:
        return len(self.vector_data.keys())

    def __iter__(self) -> bool:
        return (x for x in self.vector_data.items())

    def clear(self) -> None:
        self.vector_data = {}
        self.vector_indexes = {}
    
    def copy(self) -> '_VectorStore':
        return _VectorStore \
        (
            vector_data=copy.deepcopy(self.vector_data),
            vector_indixes=copy.deepcopy(self.vector_indexes)
        )

    def items(self) -> list[tuple[str, np.array]]:
        return self.vector_data.items()
    
    def keys(self) -> list[str]:
        return self.vector_data.keys()
    
    def setdefault(self, v_id: str, vector: np.array) -> None:
        return self.vector_data.setdefault(v_id, vector)

    def values(self) -> list[np.array]:
        return self.vector_data.values()

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

    def add_vector(self, v_id: str, vector: np.ndarray, runtime: bool = False) -> None:
        """
        Adding Card to Vector Database given a vector ID and the corresponding encoded vector.

        Parameters: 
            v_id: str
                ID for the vector
            vector: np.ndarray
                Corresponding vector (np.ndarray) to ID
            runtime: bool
                Whether the runtime of adding the data should be displayed
        
        Output:
            None
        """
        if self.contains(v_id):
            return
        self.vector_data[v_id] = vector
        if self.optimized: self._update_index(v_id, vector, batch_size=self._get_optimal_batch_size(), runtime=runtime)

    def get_vector(self, v_id: str) -> np.ndarray:
        """
        Get vector that corresponds to a certain ID

        Parameters: 
            v_id: str
                ID for the vector
        
        Output:
            np.ndarray or None
                np.ndarray if ID is in the database, else None
        """
        return self.find_vector(v_id)
    
    def get_vector_tup(self, v_id: str) -> tuple[str, np.ndarray]:
        """
        Get tuple of ID and Vector that corresponds to a certain ID

        Parameters: 
            v_id: str
                ID for the vector
        
        Output:
            tuple[str,np.ndarray or None]
                tuple[str,np.ndarray] if ID is in the database, else tuple[str,None]
        """
        return (v_id, self.vector_data[v_id])

    def get_random_vector(self) -> tuple[str, np.ndarray]:
        """
        Get a random (ID, Vector) pair from Database

        Parameters: 
            None
        
        Output:
            tuple[str, np.ndarray]
                A random (ID, Vector) pair from the DataBase
        """
        return (self.find_vector_pair(random.choice(list(self.vector_data.keys()))))

    def _get_optimal_batch_size(self, max_vector_dim: int = 320, vector_value_type = cp.float16) -> int:
        free_memory, _ = cp.cuda.runtime.memGetInfo()
        vector_bytes = np.dtype(vector_value_type).itemsize * max_vector_dim
        batch_size = max(1024, free_memory // vector_bytes // 1024)
        return int(min(batch_size, len(self)+1))

    def _update_index(self, v_id: str, vector: np.ndarray, batch_size: int = 8192, runtime: bool = False) -> None:
        start = time.time()

        query_vector = cp.asarray(vector, dtype=cp.float16)
        
        if v_id not in self.vector_indexes:
            self.vector_indexes[v_id] = {}

        vector_keys = list(self.vector_data.keys())
        vector_values = cp.asarray(list(self.vector_data.values()), dtype=cp.float16)

        indexes = {}

        with cp.cuda.Stream():
            for i in range(0, len(vector_values), batch_size):
                distances = cp.sum((vector_values[i:i + batch_size].astype(cp.float16) - query_vector.astype(cp.float16)) ** 2, axis=1)
                indexes.update({key: dist for key, dist in zip(vector_keys[i:i + batch_size], distances.get())})

        self.vector_indexes[v_id] = indexes

    def get_similar_vectors(self, v_id: str, n_results: int = 5) -> list[tuple[str, np.ndarray]]:
        """
        Updates the stored similarity indexes for given vector

        Parameters: 
            q_vector: np.ndarray
                Query vector
            n_results: int
                Number of similar vectors
        
        Output:
            list[tuple[str, np.ndarray]]
                list of (ID, Vector) pairs
        """
        if self.optimized: return self.get_similar_vectors_optimized(v_id=v_id, n_results=n_results)
        else: return self.get_similar_vectors_unoptimized(v_id=v_id, n_results=n_results)
    
    def get_similar_vectors_unoptimized(self, v_id: str, n_results: int = 5) -> list[tuple[str, np.ndarray]]:
        q_vector_gpu = cp.asarray(self.find_vector(v_id), dtype=cp.float16)
        q_vector_norm = cp.linalg.norm(q_vector_gpu)

        results = []
        for vector_id, vector in self.vector_data.items():
            vector_gpu = cp.asarray(vector, dtype=cp.float16)
            vector_norm = cp.linalg.norm(vector_gpu)

            distance = cp.linalg.norm(q_vector_gpu - vector_gpu)
            relative_distance = distance / (q_vector_norm + vector_norm)
            
            results.append((vector_id, relative_distance.get()))

        results.sort(key=lambda x: x[1])
        return results[:n_results]
    
    def get_similar_vectors_optimized(self, v_id: str, n_results: int = 5) -> list[tuple[str, np.ndarray]]:
        sorted_results = sorted(self.vector_indexes[v_id].items(), key=lambda x: x[1])
        return sorted_results[:n_results]

    def find_vector_pair(self, q_id: str) -> tuple[str, np.ndarray]:
        if q_id in self.vector_data:
            return self.get_vector_tup(q_id)
        else:
            for id, _ in self.vector_data.items():
                if q_id in id:
                    return self.get_vector_tup(id)
            raise KeyError(("KeyError:" + str(q_id)))
    
    def find_vector(self, q_id: str) -> np.ndarray:
        return self.find_vector_pair(q_id)[1]
    
    def find_id(self, q_id: str) -> str:
        return self.find_vector_pair(q_id)[0]

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
        vector_pair = self.find_vector_pair(v_id)
        return self.decoder.decode_to_string(vector_pair[0], vector_pair[1])

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
        vector_pair = self.find_vector_pair(v_id)
        return self.decoder.decode_to_dict(vector_pair[0], vector_pair[1])
    
class VectorDatabase(object):
    def __init__(self, vector_store: _VectorStore = None, optimized: bool = True, runtime: bool = False):
        """
        Responsible for initializing the VectorDatabase object
        """
        if vector_store is not None: assert(isinstance(vector_store, _VectorStore)), f"Incorrect parameter vector_store: _VectorStore, passed {type(vector_store).__name__}"
        assert(isinstance(optimized, bool)), f"Incorrect parameter optimized: bool, passed {type(optimized).__name__}"
        assert(isinstance(runtime, bool)), f"Incorrect parameter runtime: bool, passed {type(runtime).__name__}"
        self._vector_store = vector_store if vector_store is not None else _VectorStore(optimized=optimized)
        self._runtime = runtime

    def __str__(self) -> str:
        return str(self._vector_store)
    
    def __eq__(self, item) -> bool:
        if not isinstance(item, VectorDatabase): return False
        return (self._vector_store == item._vector_store) and (self._runtime == item._runtime)

    def __len__(self) -> int:
        return len(self._vector_store)

    def __iter__(self) -> bool:
        return (x for x in self._vector_store)

    def size(self) -> int:
        return len(self._vector_store)

    def clear(self) -> None:
        self._vector_store.clear()
    
    def copy(self) -> 'VectorDatabase':
        return VectorDatabase(self._vector_store.copy(), self._runtime)

    def items(self) -> list[tuple[str, np.array]]:
        return self._vector_store.items()
    
    def keys(self) -> list[str]:
        return self._vector_store.keys()
    
    def values(self) -> list[np.array]:
        return self._vector_store.values()

    def setdefault(self, v_id: str, vector: np.array) -> None:
        return self._vector_store.setdefault(v_id, vector)

    def parse_json(self, filename: str, max_lines: int = -1) -> int:
        """
        Build DataBase from JSON

        Parameters:
            filename: str
                Name of JSON file you want passed in
            max_lines: int
                Maximum number of lines in DataBase
        
        Output:
            VectorStore:
                VectorStore with first (max_lines) of JSON
        """
        
        if os.path.isfile(filename):
            set_data: pd.DataFrame = self._parse_file(filename)
        else:
            set_data: pd.DataFrame = self._AWS_DATA_REQUEST(filename)

        start_time: float = time.time()
        
        for game_set in set_data:
            for card in game_set['cards']:
                if 'commander' in card['legalities'] and card['legalities']['commander']=="Legal" and 'paper' in card['availability']:
                    self._vector_store.add_card(Card(card), self._runtime)

                    if max_lines > -1 and len(self) >= max_lines:
                        if self._runtime: print(f"BUILDING DATABASE: {time.time()-start_time}, DATASET SIZE: {len(self._vector_store)}")
                        return len(self)

        if self._runtime: print(f"BUILDING DATABASE: {time.time()-start_time}, DATASET SIZE: {len(self._vector_store)}")
        return len(self)

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
        return self._vector_store.contains(value)

    def add_card(self, crd: Card) -> None:
        """
        Adding Card to VectorStore

        Parameters:
            crd: Card
                Card to be encoded and added to Vector Database
        
        Output:
            None
        """
        self._vector_store.add_card(crd)

    def add_vector(self, v_id: str, vector: np.ndarray) -> None:
        """
        Adding Card to VectorStore

        Parameters: 
            v_id: str
                ID of Vector
            vector: np.ndarray
                Corresponding Vector to ID
        
        Output:
            None
        """
        self._vector_store.add_vector(v_id,vector)

    def get_vector(self, v_id: str) -> np.ndarray:
        return self._vector_store.get_vector(v_id)
    
    def get_vector_tup(self, v_id: str) -> tuple[str, np.ndarray]:
        return self._vector_store.get_vector_tup(v_id)
    
    def get_random_vector(self) -> tuple[str, np.ndarray]:
        """
        Get a random (ID, Vector) pair from Database

        Parameters: 
            None
        
        Output:
            tuple[str, np.ndarray]
                A random (ID, Vector) pair from the DataBase
        """
        return self._vector_store.get_random_vector()
    
    def get_similar_vectors(self, q_vector: np.ndarray, n_results: int = 5) -> list[tuple[str, np.ndarray]]:
        """
        Updates the stored similarity indexes for given vector

        Parameters: 
            q_vector: np.ndarray
                Query vector
            n_results: int
                Number of similar vectors
        
        Output:
            list[tuple[str, np.ndarray]]
                list of (ID, Vector) pairs
        """
        return self._vector_store.get_similar_vectors(q_vector,n_results)
    
    def find_vector_pair(self, v_id: str) -> tuple[str,np.ndarray]:
        return self._vector_store.find_vector_pair(v_id)
    
    def find_vector(self, v_id: str) -> np.ndarray:
        return self._vector_store.find_vector(v_id)

    def find_id(self, v_id: str) -> str:
        return self._vector_store.find_id(v_id)
    
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
        return self._vector_store.describe_vector_string(v_id)
    
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
        return self._vector_store.describe_vector_dict(v_id)

    def _parse_file(self, filename: str) -> pd.DataFrame:
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
        start_time: float = time.time()
        assert os.path.isfile(filename), f"{filename} not found."
        set_data: pd.DataFrame = pd.read_json(filename)['data'][2:]
        if self._runtime: print(f"PARSING DATASET: {time.time()-start_time}")
        return set_data

    def _AWS_DATA_REQUEST(self, filename: str, _AWS_S3_BUCKET = None, _REGION_NAME = None, _AWS_ACCESS_KEY_ID = None, _AWS_SECRET_ACCESS_KEY = None) -> pd.DataFrame:
        start_time: float = time.time()
        AWS_S3_BUCKET: str = 'allcarddata' if _AWS_S3_BUCKET is None else _AWS_S3_BUCKET
        REGION_NAME: str = 'us-east-1' if _REGION_NAME is None else _REGION_NAME

        AWS_ACCESS_KEY_ID: str = str(os.getenv("AWS_ACCESS_KEY_ID")) if _AWS_ACCESS_KEY_ID is None else _AWS_SECRET_ACCESS_KEY
        AWS_SECRET_ACCESS_KEY: str = str(os.getenv("AWS_SECRET_ACCESS_KEY")) if AWS_SECRET_ACCESS_KEY is None else _AWS_SECRET_ACCESS_KEY

        s3_client: boto3.client = boto3.client(
            service_name = "s3",
            aws_access_key_id = AWS_ACCESS_KEY_ID,
            aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
            region_name = REGION_NAME
        )

        response = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=filename)
        status: int = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        assert(status==200), "Invalid Request"

        if self._runtime: print(f"PARSING DATASET: {time.time()-start_time}")
        return pd.read_json(response.get("Body"))['data'][2:]