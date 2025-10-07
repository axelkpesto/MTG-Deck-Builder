import torch
import numpy as np
import pandas as pd
import os
import random
import boto3
from Card_Lib import CardEncoder, CardDecoder, CardFields

class VectorStore(object):
    def __init__(self, encoder, decoder) -> None:
        """
        Responsible for initializing the VectorStore object

        Parameters:
            None
        """
        self.encoder = encoder
        self.decoder = decoder
        self.vector_data: dict[str, torch.Tensor] = {}
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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

    def items(self) -> list[tuple[str, np.array]]:
        return self.vector_data.items()
    
    def keys(self) -> list[str]:
        return self.vector_data.keys()
    
    def values(self) -> list[np.array]:
        return self.vector_data.values()

    def setdefault(self, v_id: str, vector: np.array) -> None:
        return self.vector_data.setdefault(v_id, vector)

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
        return value in self.vector_data

    def add_vector(self, v_id: str, vector: np.ndarray) -> None:
        """
        Adding item to Vector Database given a vector ID and the corresponding encoded vector.

        Parameters:
            v_id: str
                ID for the vector
            vector: np.ndarray
                Corresponding vector (np.ndarray) to ID

        Output:
            None
        """
        if self.contains(v_id):
            return
        vector_tensor = torch.tensor(vector, dtype=torch.float32).to(self.device)
        self.vector_data[v_id] = vector_tensor

    def get(self, v_id: str, default):
        return self.vector_data[v_id] if v_id in self.vector_data else default
    
    def get_vector(self, v_id: str) -> np.ndarray:
        """
        Get vector that corresponds to a certain ID

        Parameters:
            v_id: str
                ID for the vector

        Output:
            np.ndarray
                np.ndarray if ID is in the database
        """
        return self.vector_data[v_id].cpu().numpy()

    def get_vector_tup(self, v_id: str) -> tuple[str, torch.tensor]:
        """
        Get tuple of ID and Vector that corresponds to a certain ID

        Parameters:
            v_id: str
                ID for the vector

        Output:
            tuple[str, torch.tensor or None]
                tuple[str, torch.tensor] if ID is in the database, else tuple[str, None]
        """
        return (v_id, self.get_vector(v_id))

    def get_random_vector(self) -> tuple[str, torch.tensor]:
        """
        Get a random (ID, Vector) pair from Database

        Parameters:
            None
        
        Output:
            tuple[str, torch.tensor]
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

    def get_similar_vectors(self, q_vector: torch.tensor, n_results: int = 5) -> list[tuple[str, torch.tensor]]:
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

    def find_vector_pair(self, q_id: str) -> tuple[str, torch.tensor]:
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
        """
        String representation of given vector

        Parameters:
            v_id: str
                ID of Vector
        
        Output:
            str
                String representation of vector
        """
        if self.decoder:
            return self.decoder.decode(v_id, self.get_vector(v_id=v_id))
        return f"{v_id}: {self.get_vector(v_id=v_id)}"

    def describe_vector_dict(self, v_id: str) -> dict[str, str]:
        """
        Dictionary representation of given vector

        Parameters:
            v_id: str
                ID of Vector
        
        Output:
            dict[str, str]
                Dictionary representation of vector
        """
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

class VectorDatabase(object):
    def __init__(self, encoder, decoder) -> None:
        """
        Responsible for initializing the VectorDatabase object
        """
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

    def items(self) -> list[tuple[str, np.array]]:
        return self.vector_store.items()
    
    def keys(self) -> list[str]:
        return self.vector_store.keys()
    
    def values(self) -> list[np.array]:
        return self.vector_store.values()

    def setdefault(self, v_id: str, vector: np.array) -> None:
        return self.vector_store.setdefault(v_id, vector)

    def parse_json(self, filename: str, max_lines: int = -1) -> VectorStore:
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
        set_data: pd.DataFrame = self._parse_file(filename) \
                                 if os.path.isfile(filename) \
                                 else \
                                 self._AWS_DATA_REQUEST(filename)
        
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

    def add_vector(self, v_id: str, vector: np.ndarray) -> None:
        """
        Adding Vector to VectorStore

        Parameters:
            v_id: str
                ID of Vector
            vector: np.ndarray
                Corresponding Vector to ID
        
        Output:
            None
        """
        self.vector_store.add_vector(v_id, vector)
    
    def get(self, v_id, default = None) -> torch.tensor:
        return self.vector_store.get(v_id, default)

    def get_vector(self, v_id: str) -> torch.tensor:
        return self.vector_store.get_vector(v_id)
    
    def get_vector_tup(self, v_id: str) -> tuple[str, torch.tensor]:
        return self.vector_store.get_vector_tup(v_id)
    
    def get_random_vector(self) -> tuple[str, torch.tensor]:
        return self.vector_store.get_random_vector()
    
    def get_similar_vectors(self, q_vector: torch.tensor, n_results: int = 5) -> list[tuple[str, torch.tensor]]:
        return self.vector_store.get_similar_vectors(q_vector, n_results)
    
    def find_vector_pair(self, v_id: str) -> tuple[str, torch.tensor]:
        return self.vector_store.find_vector_pair(v_id)
    
    def find_vector(self, v_id: str) -> torch.tensor:
        return self.vector_store.find_vector(v_id)

    def find_id(self, v_id: str) -> str:
        return self.vector_store.find_id(v_id)
    
    def get_vector_description(self, v_id: str) -> str:
        return self.vector_store.describe_vector_string(v_id=v_id)

    def get_vector_description_dict(self, v_id: str) -> dict[str, str]:
        return self.vector_store.describe_vector_dict(v_id=v_id)

    def _parse_file(self, filename: str) -> pd.DataFrame:
        assert os.path.isfile(filename), f"{filename} not found."
        data = pd.read_json(filename)['data'][2:]
        return data

    def _AWS_DATA_REQUEST(self, filename: str) -> pd.DataFrame:
        AWS_S3_BUCKET: str = 'allcarddata'
        REGION_NAME: str = 'us-east-1'

        AWS_ACCESS_KEY_ID: str = str(os.getenv("AWS_ACCESS_KEY_ID"))
        AWS_SECRET_ACCESS_KEY: str = str(os.getenv("AWS_SECRET_ACCESS_KEY"))

        s3_client = boto3.client(
            service_name="s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=REGION_NAME
        )

        response = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=filename)
        status: int = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        assert status == 200, "Invalid Request"
        return pd.read_json(response.get("Body"))['data'][2:]
    
    def save(self, filename: str) -> None:
        self.vector_store.save(filename)

    def load(self, filename: str) -> None:
        self.vector_store.load(filename)

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