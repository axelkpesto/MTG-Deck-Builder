import json
from typing import Dict

def load_config() -> Dict[str, Dict[str, str]]:
    with open("config/config.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw

class Config:
    def __init__(self):
        cfg = load_config()
        self.datasets = cfg["datasets"]
        self.models = cfg["models"]

CONFIG = Config()