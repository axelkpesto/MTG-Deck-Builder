"""Load Configuration from JSON to usable dictionaries"""

import json
from pathlib import Path
from typing import Dict

def load_config() -> Dict[str, Dict[str, str]]:
    """Load Config from JSON"""
    config_path = Path(__file__).with_name("config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw

class Config:
    """Processes JSON into usable dictionaries for file access"""

    def __init__(self):
        cfg = load_config()
        self.datasets = cfg["datasets"]
        self.models = cfg["models"]

CONFIG = Config()
