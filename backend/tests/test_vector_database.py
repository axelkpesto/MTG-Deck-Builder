"""Tests for `backend.vector_database.VectorDatabase` (facade over VectorStore).

Most behavior delegates to VectorStore; this file ensures the facade methods
exist, route through correctly, and verify the unique helpers (parse_json,
load_static, to_index, vector_to_numpy).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from backend.vector_database import VectorDatabase


# ---------------------------------------------------------------------------
# Facade pass-throughs
# ---------------------------------------------------------------------------


class TestVectorDatabasePassThroughs:
    """Methods that just forward to `vector_store`."""

    def test_len_and_size(self, populated_vector_database):
        assert len(populated_vector_database) == 5
        assert populated_vector_database.size() == 5

    def test_contains(self, populated_vector_database):
        assert "Atraxa" in populated_vector_database
        assert populated_vector_database.contains("Atraxa") is True
        assert "Missing" not in populated_vector_database

    def test_iter_yields_pairs(self, populated_vector_database):
        items = list(populated_vector_database)
        assert all(isinstance(it, tuple) and len(it) == 2 for it in items)

    def test_getitem_string(self, populated_vector_database):
        v = populated_vector_database["Atraxa"]
        assert isinstance(v, torch.Tensor)

    def test_getitem_int(self, populated_vector_database):
        v = populated_vector_database[0]
        assert isinstance(v, torch.Tensor)

    def test_getitem_int_out_of_range(self, populated_vector_database):
        with pytest.raises(IndexError):
            _ = populated_vector_database[99999]

    def test_getitem_invalid_type(self, populated_vector_database):
        with pytest.raises(TypeError):
            _ = populated_vector_database[1.5]

    def test_clear(self, populated_vector_database):
        populated_vector_database.clear()
        assert len(populated_vector_database) == 0

    def test_keys_values_items(self, populated_vector_database):
        keys = populated_vector_database.keys()
        vals = populated_vector_database.values()
        items = populated_vector_database.items()
        assert len(keys) == len(vals) == len(items)

    def test_get_with_default(self, populated_vector_database):
        assert populated_vector_database.get("Missing", default="x") == "x"

    def test_get_random_vector(self, populated_vector_database):
        name, vec = populated_vector_database.get_random_vector()
        assert name in populated_vector_database

    def test_find_methods(self, populated_vector_database):
        assert populated_vector_database.find_id("atraxa") == "Atraxa"
        v = populated_vector_database.find_vector("atraxa")
        assert isinstance(v, torch.Tensor)
        pair = populated_vector_database.find_vector_pair("atraxa")
        assert pair[0] == "Atraxa"

    def test_get_encoder_decoder(self, fake_encoder, fake_decoder):
        db = VectorDatabase(fake_encoder, fake_decoder)
        assert db.get_encoder() is fake_encoder
        assert db.get_decoder() is fake_decoder

    def test_to_ndarray_and_dataframe(self, populated_vector_database):
        assert isinstance(populated_vector_database.to_ndarray(), np.ndarray)
        assert isinstance(populated_vector_database.to_dataframe(), pd.DataFrame)

    def test_setdefault_forwards(self, empty_vector_database):
        out = empty_vector_database.setdefault("X", torch.zeros(4))
        assert "X" in empty_vector_database

    def test_get_similar_vectors_returns_pairs(self, populated_vector_database):
        # Implementation returns min(n_results + 1, db_size) entries.
        q = populated_vector_database.get_vector("Atraxa")
        results = populated_vector_database.get_similar_vectors(q, n_results=2)
        assert len(results) == min(3, len(populated_vector_database))
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        assert results[0][0] == "Atraxa"

    def test_filter_forwards(self, populated_vector_database):
        out = populated_vector_database.filter(lambda n, v: True, limit=1)
        assert len(out) == 1


# ---------------------------------------------------------------------------
# Equality and hashing
# ---------------------------------------------------------------------------


class TestEquality:
    """`__eq__` compares underlying VectorStore content."""

    def test_empty_databases_equal(self, fake_encoder, fake_decoder):
        a = VectorDatabase(fake_encoder, fake_decoder)
        b = VectorDatabase(fake_encoder, fake_decoder)
        assert a == b

    def test_equal_when_tensor_contents_match(self, fake_encoder, fake_decoder):
        a = VectorDatabase(fake_encoder, fake_decoder)
        b = VectorDatabase(fake_encoder, fake_decoder)
        v = np.zeros(4, dtype=np.float32)
        a.add_vector("X", v)
        b.add_vector("X", v)
        assert a == b

    def test_unequal_when_tensor_contents_differ(self, fake_encoder, fake_decoder):
        a = VectorDatabase(fake_encoder, fake_decoder)
        b = VectorDatabase(fake_encoder, fake_decoder)
        a.add_vector("X", np.zeros(4, dtype=np.float32))
        b.add_vector("X", np.ones(4, dtype=np.float32))
        assert a != b

    def test_unequal_against_other_type(self, empty_vector_database):
        assert (empty_vector_database == "not a db") is False


# ---------------------------------------------------------------------------
# to_index
# ---------------------------------------------------------------------------


class TestToIndex:
    """`to_index` returns a name → positional index map."""

    def test_index_count_matches_size(self, populated_vector_database):
        idx = populated_vector_database.to_index()
        assert len(idx) == len(populated_vector_database)

    def test_indices_zero_based_and_sequential(self, populated_vector_database):
        idx = populated_vector_database.to_index()
        assert sorted(idx.values()) == list(range(len(populated_vector_database)))


# ---------------------------------------------------------------------------
# vector_to_numpy (static helper)
# ---------------------------------------------------------------------------


class TestVectorToNumpy:
    """Coerce torch tensors, numpy arrays, and lists into float32 ndarrays."""

    def test_torch_tensor(self):
        t = torch.zeros(3)
        out = VectorDatabase.vector_to_numpy(t)
        assert isinstance(out, np.ndarray)

    def test_torch_tensor_with_grad(self):
        # Tensor on cpu with requires_grad=True; .detach() path
        t = torch.zeros(3, requires_grad=True)
        out = VectorDatabase.vector_to_numpy(t)
        assert isinstance(out, np.ndarray)

    def test_list_input(self):
        out = VectorDatabase.vector_to_numpy([1.0, 2.0, 3.0])
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32

    def test_numpy_input_returns_array(self):
        arr = np.array([1.0, 2.0], dtype=np.float64)
        out = VectorDatabase.vector_to_numpy(arr)
        assert isinstance(out, np.ndarray)


# ---------------------------------------------------------------------------
# Save / load and load_static
# ---------------------------------------------------------------------------


class TestSaveLoadStatic:
    """Round-trip through disk plus `load_static` classmethod."""

    def test_save_load_round_trip(self, tmp_path, populated_vector_database):
        path = str(tmp_path / "vd.pt")
        populated_vector_database.save(path)

        loaded = VectorDatabase.load_static(path)
        assert set(loaded.keys()) == set(populated_vector_database.keys())

    def test_load_clears_existing(self, tmp_path, populated_vector_database, fake_encoder, fake_decoder):
        # Save first DB, then load into a different DB that already has content.
        path = str(tmp_path / "vd.pt")
        populated_vector_database.save(path)

        other = VectorDatabase(fake_encoder, fake_decoder)
        other.add_vector("Extra", np.zeros(4, dtype=np.float32))
        assert "Extra" in other

        other.load(path)
        assert "Extra" not in other  # cleared during load


# ---------------------------------------------------------------------------
# parse_json
# ---------------------------------------------------------------------------


class TestParseJson:
    """`parse_json` ingests MTGJSON-style data and encodes commander-legal cards."""

    def _build_payload(self, tmp_path: Path) -> Path:
        # MTGJSON file shape: top-level dict with 'data' as nested set objects.
        # The parser reads ['data'][2:] and iterates each set's 'cards'. To make
        # that work via pandas.read_json, we ship a small fixture matching the
        # column-style format pandas can ingest.
        # Simplest: a dict with two padding set objects followed by ours.
        payload = {
            "data": {
                "_meta1": {"cards": []},
                "_meta2": {"cards": []},
                "TEST": {
                    "cards": [
                        {
                            "name": "Legal Card",
                            "types": ["Creature"],
                            "supertypes": [],
                            "subtypes": ["Human"],
                            "manaValue": 2,
                            "manaCost": "{1}{U}",
                            "colorIdentity": ["U"],
                            "rarity": "rare",
                            "text": "Draw a card.",
                            "edhrecRank": 1,
                            "power": "1", "toughness": "1", "loyalty": "",
                            "identifiers": {"scryfallId": "id1"},
                            "legalities": {"commander": "Legal"},
                            "availability": ["paper"],
                        },
                        {
                            "name": "Illegal Card",
                            "types": ["Creature"],
                            "supertypes": [],
                            "subtypes": [],
                            "manaValue": 1,
                            "manaCost": "{U}",
                            "colorIdentity": ["U"],
                            "rarity": "common",
                            "text": "",
                            "identifiers": {"scryfallId": "id2"},
                            "legalities": {"commander": "Banned"},
                            "availability": ["paper"],
                        },
                    ]
                },
            }
        }
        path = tmp_path / "mtgjson.json"
        path.write_text(json.dumps(payload))
        return path

    def test_only_commander_legal_paper_cards_added(self, tmp_path, fake_encoder, fake_decoder):
        # We need an encoder whose .encode returns (name, ndarray). Our fake does.
        db = VectorDatabase(fake_encoder, fake_decoder)
        path = self._build_payload(tmp_path)
        db.parse_json(str(path))
        assert "Legal Card" in db
        assert "Illegal Card" not in db

    def test_max_lines_caps_insert_count(self, tmp_path, fake_encoder, fake_decoder):
        # With max_lines=0 the loop still inserts one card (num_cards is
        # incremented BEFORE the `0 <= max_lines <= num_cards` check), then
        # returns. Exactly one card should land in the store.
        db = VectorDatabase(fake_encoder, fake_decoder)
        path = self._build_payload(tmp_path)
        db.parse_json(str(path), max_lines=0)
        assert len(db) == 1

    def test_max_lines_negative_means_unlimited(self, tmp_path, fake_encoder, fake_decoder):
        # Negative max_lines disables the cap; both commander-legal cards added.
        db = VectorDatabase(fake_encoder, fake_decoder)
        path = self._build_payload(tmp_path)
        db.parse_json(str(path), max_lines=-1)
        assert len(db) == 1  # only one card in fixture is commander-legal

    def test_missing_file_raises(self, fake_encoder, fake_decoder):
        db = VectorDatabase(fake_encoder, fake_decoder)
        with pytest.raises(AssertionError):
            db.parse_json("does-not-exist.json")


# ---------------------------------------------------------------------------
# get_list with default
# ---------------------------------------------------------------------------


class TestGetList:
    """Batch `get_list` returns vectors / default for each id."""

    def test_get_list_mixed_results(self, populated_vector_database):
        out = populated_vector_database.get_list(["Atraxa", "Missing"], default="x")
        assert isinstance(out[0], torch.Tensor)
        assert out[1] == "x"
