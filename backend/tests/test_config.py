"""Tests for `backend.config.config` JSON-loaded configuration."""
import json

import pytest

from backend.config import CONFIG
from backend.config.config import Config, load_config


class TestLoadConfig:
    """`load_config` returns a `dict` keyed by `datasets` and `models`."""

    def test_returns_dict(self):
        cfg = load_config()
        assert isinstance(cfg, dict)

    def test_has_datasets_and_models_sections(self):
        cfg = load_config()
        assert "datasets" in cfg
        assert "models" in cfg


class TestConfigObject:
    """The `Config` instance exposes the JSON sections as attributes."""

    def test_datasets_attribute_is_dict(self):
        cfg = Config()
        assert isinstance(cfg.datasets, dict)

    def test_models_attribute_is_dict(self):
        cfg = Config()
        assert isinstance(cfg.models, dict)

    def test_module_level_config_is_instance(self):
        assert isinstance(CONFIG, Config)


class TestExpectedKeys:
    """Required dataset / model paths must be present so downstream modules import."""

    @pytest.mark.parametrize("key", [
        "FULL_DATASET_PATH",
        "CARDS_DATASET_PATH",
        "DECKS_DATASET_PATH",
        "VECTOR_DATABASE_PATH",
        "TAGS_DATASET_PATH",
        "GRAPH_NODES_DATA_PATH",
        "GRAPH_EDGES_DATA_PATH",
        "NODE_EMBEDDINGS_PATH",
        "NODE_FEATURES_PATH",
    ])
    def test_dataset_key_present(self, key):
        assert key in CONFIG.datasets

    @pytest.mark.parametrize("key", [
        "TAGGING_MODEL_PATH",
        "GEN_MODEL_PATH",
    ])
    def test_model_key_present(self, key):
        assert key in CONFIG.models

    def test_all_paths_are_strings(self):
        for v in CONFIG.datasets.values():
            assert isinstance(v, str)
        for v in CONFIG.models.values():
            assert isinstance(v, str)
