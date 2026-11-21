"""Tests for `backend.vector_database.VectorStore`.

Covers every dunder method plus add/get/find/filter/similarity/serialization paths.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from backend.vector_database import VectorStore


# ---------------------------------------------------------------------------
# Membership and length
# ---------------------------------------------------------------------------


class TestStoreSizeAndMembership:
    """`__len__`, `size`, `__contains__`, `contains`."""

    def test_empty_store_len_zero(self, empty_vector_store):
        """An empty store reports zero length and size."""
        assert len(empty_vector_store) == 0
        assert empty_vector_store.size() == 0

    def test_populated_store_len(self, populated_vector_store):
        """A populated store reports its entry count."""
        assert len(populated_vector_store) == 5
        assert populated_vector_store.size() == 5

    def test_contains_after_insert(self, empty_vector_store):
        """An inserted id is reported as present."""
        empty_vector_store.add_vector("X", np.zeros(4, dtype=np.float32))
        assert "X" in empty_vector_store
        assert empty_vector_store.contains("X") is True

    def test_does_not_contain_missing_key(self, populated_vector_store):
        """A missing id is reported as absent."""
        assert "ZZZ" not in populated_vector_store


# ---------------------------------------------------------------------------
# add_vector
# ---------------------------------------------------------------------------


class TestAddVector:
    """`add_vector` inserts new ids and is idempotent for existing ids."""

    def test_inserts_new_id(self, empty_vector_store):
        """A new id is inserted."""
        empty_vector_store.add_vector("A", np.zeros(4, dtype=np.float32))
        assert "A" in empty_vector_store
        assert len(empty_vector_store) == 1

    def test_does_not_overwrite_existing(self, empty_vector_store):
        """Re-adding an existing id keeps the original vector."""
        v1 = np.ones(4, dtype=np.float32)
        v2 = np.full((4,), 99.0, dtype=np.float32)
        empty_vector_store.add_vector("A", v1)
        empty_vector_store.add_vector("A", v2)
        assert torch.allclose(empty_vector_store["A"].cpu(), torch.from_numpy(v1))

    def test_rejects_non_string_id(self, empty_vector_store):
        """A non-string id raises `AssertionError`."""
        with pytest.raises(AssertionError):
            empty_vector_store.add_vector(123, np.zeros(4, dtype=np.float32))

    def test_marks_cache_dirty(self, empty_vector_store):
        """Adding a vector invalidates the similarity cache."""
        empty_vector_store.add_vector("A", np.zeros(4, dtype=np.float32))
        assert getattr(empty_vector_store, "_cache_dirty") is True


# ---------------------------------------------------------------------------
# get_vector / get / get_list / get_vector_tup
# ---------------------------------------------------------------------------


class TestGetters:
    """Retrieval methods and their fallback behavior."""

    def test_get_vector_returns_tensor(self, populated_vector_store):
        """`get_vector` returns a tensor for a present id."""
        v = populated_vector_store.get_vector("Sol Ring")
        assert isinstance(v, torch.Tensor)

    def test_get_with_default_when_missing(self, populated_vector_store):
        """`get` returns the supplied default for a missing id."""
        out = populated_vector_store.get("Missing", default="fallback")
        assert out == "fallback"

    def test_get_with_none_default_returns_none(self, populated_vector_store):
        """An explicit None default is honored, matching dict semantics."""
        assert populated_vector_store.get("Missing", default=None) is None

    def test_get_with_no_default_raises(self, populated_vector_store):
        """`get` with no default raises `KeyError` for a missing id."""
        with pytest.raises(KeyError):
            populated_vector_store.get("Missing")

    def test_get_list_mixes_hits_and_defaults(self, populated_vector_store):
        """`get_list` returns vectors for hits and the default for misses."""
        out = populated_vector_store.get_list(["Sol Ring", "Nope"], default="X")
        assert isinstance(out[0], torch.Tensor)
        assert out[1] == "X"

    def test_get_vector_tup_returns_id_and_tensor(self, populated_vector_store):
        """`get_vector_tup` returns the (id, vector) pair."""
        name, vec = populated_vector_store.get_vector_tup("Atraxa")
        assert name == "Atraxa"
        assert isinstance(vec, torch.Tensor)


# ---------------------------------------------------------------------------
# __getitem__
# ---------------------------------------------------------------------------


class TestGetitem:
    """String key, int index, and invalid types."""

    def test_string_key(self, populated_vector_store):
        """A string key returns the corresponding tensor."""
        v = populated_vector_store["Sol Ring"]
        assert isinstance(v, torch.Tensor)

    def test_int_index_in_range(self, populated_vector_store):
        """An in-range int index returns a tensor."""
        v = populated_vector_store[0]
        assert isinstance(v, torch.Tensor)

    def test_int_index_out_of_range_raises(self, populated_vector_store):
        """An out-of-range int index raises `IndexError`."""
        with pytest.raises(IndexError):
            _ = populated_vector_store[10_000]

    def test_invalid_key_type_raises(self, populated_vector_store):
        """An unsupported key type raises `TypeError`."""
        with pytest.raises(TypeError):
            _ = populated_vector_store[1.5]


# ---------------------------------------------------------------------------
# Iteration, items, keys, values
# ---------------------------------------------------------------------------


class TestIteration:
    """`__iter__`, `items`, `keys`, `values` return matching content."""

    def test_keys_match_items_first_elements(self, populated_vector_store):
        """`keys` matches the first element of each `items` pair."""
        keys = populated_vector_store.keys()
        items_keys = [k for k, _ in populated_vector_store.items()]
        assert keys == items_keys

    def test_values_length_matches_len(self, populated_vector_store):
        """`values` has the same length as the store."""
        assert len(populated_vector_store.values()) == len(populated_vector_store)

    def test_iter_yields_tuples(self, populated_vector_store):
        """Iteration yields (id, vector) tuples."""
        for kv in populated_vector_store:
            assert isinstance(kv, tuple)
            assert len(kv) == 2


# ---------------------------------------------------------------------------
# setdefault
# ---------------------------------------------------------------------------


class TestSetDefault:
    """`setdefault` returns existing on hit, inserts on miss."""

    def test_returns_existing(self, populated_vector_store):
        """`setdefault` returns the existing vector on a hit."""
        existing = populated_vector_store["Atraxa"]
        out = populated_vector_store.setdefault("Atraxa", torch.zeros(4))
        assert torch.equal(out, existing)

    def test_inserts_missing(self, empty_vector_store):
        """`setdefault` inserts and returns the new vector on a miss."""
        new_v = torch.ones(4)
        out = empty_vector_store.setdefault("New", new_v)
        assert "New" in empty_vector_store
        assert torch.equal(out, new_v)


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


class TestClear:
    """`clear` empties the store and invalidates the similarity cache."""

    def test_clear_empties_store(self, populated_vector_store):
        """`clear` removes all entries."""
        populated_vector_store.clear()
        assert len(populated_vector_store) == 0

    def test_clear_marks_cache_dirty(self, populated_vector_store):
        """`clear` invalidates the similarity cache."""
        populated_vector_store.clear()
        assert getattr(populated_vector_store, "_cache_dirty") is True


# ---------------------------------------------------------------------------
# to_ndarray, to_dataframe
# ---------------------------------------------------------------------------


class TestToArrayAndDataframe:
    """Conversion to numpy / pandas with and without predicates."""

    def test_to_ndarray_shape(self, populated_vector_store):
        """`to_ndarray` returns one row per stored vector."""
        arr = populated_vector_store.to_ndarray()
        assert isinstance(arr, np.ndarray)
        assert arr.shape[0] == len(populated_vector_store)

    def test_to_ndarray_with_predicate(self, populated_vector_store):
        """A predicate restricts `to_ndarray` to matching rows."""
        def pred(name, _vec):
            """Match exactly two cards by name."""
            return name in {"Atraxa", "Sol Ring"}
        arr = populated_vector_store.to_ndarray(predicate=pred)
        assert arr.shape[0] == 2

    def test_to_dataframe_columns_match_ids(self, populated_vector_store):
        """`to_dataframe` columns are the stored ids."""
        df = populated_vector_store.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(populated_vector_store.keys())


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------


class TestSimilaritySearch:
    """`get_similar_vectors` returns ranked id+score pairs."""

    def test_returns_id_score_pairs(self, populated_vector_store):
        """Results are (id, score) pairs of the expected types."""
        q = populated_vector_store.get_vector("Atraxa")
        results = populated_vector_store.get_similar_vectors(q, n_results=3)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        assert all(isinstance(r[0], str) and isinstance(r[1], float) for r in results)

    def test_top_result_is_query_itself(self, populated_vector_store):
        """The query vector is its own closest match with similarity 1.0."""
        q = populated_vector_store.get_vector("Atraxa")
        results = populated_vector_store.get_similar_vectors(q, n_results=3)
        assert results[0][0] == "Atraxa"
        assert pytest.approx(results[0][1], rel=1e-5) == 1.0

    def test_results_sorted_descending(self, populated_vector_store):
        """Results are sorted by descending similarity score."""
        q = populated_vector_store.get_vector("Atraxa")
        results = populated_vector_store.get_similar_vectors(q, n_results=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_store_returns_empty(self, empty_vector_store):
        """Similarity search on an empty store returns no results."""
        out = empty_vector_store.get_similar_vectors(torch.zeros(4))
        assert out == []

    def test_n_results_larger_than_store(self, populated_vector_store):
        """Requesting more results than entries returns at most the store size."""
        q = populated_vector_store.get_vector("Atraxa")
        results = populated_vector_store.get_similar_vectors(q, n_results=1000)
        assert len(results) == len(populated_vector_store)

    def test_accepts_numpy_query(self, populated_vector_store):
        """A numpy query returns min(n+1, size) entries, self first."""
        q = populated_vector_store.get_vector("Atraxa").cpu().numpy()
        results = populated_vector_store.get_similar_vectors(q, n_results=2)
        assert len(results) == min(3, len(populated_vector_store))
        assert results[0][0] == "Atraxa"


# ---------------------------------------------------------------------------
# find_vector_pair / find_vector / find_id
# ---------------------------------------------------------------------------


class TestFindMethods:
    """`find_vector_pair` resolves exact and partial (substring) matches."""

    def test_exact_match(self, populated_vector_store):
        """An exact name resolves to itself."""
        name, _ = populated_vector_store.find_vector_pair("Atraxa")
        assert name == "Atraxa"

    def test_partial_case_insensitive_match(self, populated_vector_store):
        """A case-insensitive substring resolves to the full name."""
        name, _ = populated_vector_store.find_vector_pair("sol")
        assert name == "Sol Ring"

    def test_missing_raises(self, populated_vector_store):
        """An unresolvable name raises `KeyError`."""
        with pytest.raises(KeyError):
            populated_vector_store.find_vector_pair("ZZZ-nonexistent")

    def test_find_vector_returns_tensor(self, populated_vector_store):
        """`find_vector` returns the matched tensor."""
        v = populated_vector_store.find_vector("Atraxa")
        assert isinstance(v, torch.Tensor)

    def test_find_id_returns_canonical(self, populated_vector_store):
        """`find_id` returns the canonical id for a partial match."""
        assert populated_vector_store.find_id("counterspell") == "Counterspell"


# ---------------------------------------------------------------------------
# get_random_vector
# ---------------------------------------------------------------------------


class TestRandomVector:
    """`get_random_vector` returns a valid (id, vector) pair from the store."""

    def test_returns_existing_entry(self, populated_vector_store):
        """The random entry exists in the store."""
        name, _ = populated_vector_store.get_random_vector()
        assert name in populated_vector_store

    def test_raises_on_empty_store(self, empty_vector_store):
        """An empty store raises `IndexError`."""
        with pytest.raises(IndexError):
            empty_vector_store.get_random_vector()


# ---------------------------------------------------------------------------
# describe_vector_*
# ---------------------------------------------------------------------------


class TestDescribeVector:
    """Decoder-backed and fallback description paths."""

    def test_describe_uses_decoder_when_available(self, populated_vector_database):
        """The text description includes decoder fields like Name."""
        out = populated_vector_database.get_vector_description("Forest")
        assert "Name" in out

    def test_describe_dict_uses_decoder_when_available(self, populated_vector_database):
        """The dict description includes decoder fields like Name."""
        out = populated_vector_database.get_vector_description_dict("Forest")
        assert "Name" in out


# ---------------------------------------------------------------------------
# filter / filter_iterator
# ---------------------------------------------------------------------------


class TestFilter:
    """`filter` honors the predicate and the names_only/vectors_only flags."""

    def test_filter_default_returns_pairs(self, populated_vector_store):
        """Default `filter` returns (id, vector) pairs."""
        out = populated_vector_store.filter(lambda n, v: n.startswith("S"))
        assert all(isinstance(p, tuple) and len(p) == 2 for p in out)

    def test_filter_names_only(self, populated_vector_store):
        """`names_only` returns matched ids."""
        out = populated_vector_store.filter(
            lambda n, v: n.startswith("S"), names_only=True
        )
        assert all(isinstance(p, str) for p in out)

    def test_filter_vectors_only(self, populated_vector_store):
        """`vectors_only` returns matched tensors."""
        out = populated_vector_store.filter(
            lambda n, v: n.startswith("S"), vectors_only=True
        )
        assert all(isinstance(p, torch.Tensor) for p in out)

    def test_filter_limit_caps_results(self, populated_vector_store):
        """`limit` caps the number of results."""
        out = populated_vector_store.filter(lambda n, v: True, limit=2)
        assert len(out) == 2

    def test_filter_swallows_predicate_errors(self, populated_vector_store):
        """Entries whose predicate raises are silently skipped."""
        def buggy(name, _vec):
            """Raise for one specific entry to exercise error handling."""
            if name == "Sol Ring":
                raise ValueError("boom")
            return True

        out = populated_vector_store.filter(buggy, names_only=True)
        assert "Sol Ring" not in out

    def test_filter_iterator_yields_pairs(self, populated_vector_store):
        """`filter_iterator` lazily yields (id, vector) pairs up to the limit."""
        items = list(populated_vector_store.filter_iterator(lambda n, v: True, limit=3))
        assert len(items) == 3
        assert all(isinstance(i, tuple) for i in items)


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    """Round-trip a populated store through torch.save/torch.load."""

    def test_round_trip(self, tmp_path, populated_vector_store, fake_encoder, fake_decoder):
        """A saved store reloads with identical keys and vectors."""
        path = tmp_path / "vd.pt"
        populated_vector_store.save(str(path))

        new = VectorStore(fake_encoder, fake_decoder)
        new.load(str(path))

        assert set(new.keys()) == set(populated_vector_store.keys())
        for k in populated_vector_store.keys():
            assert torch.allclose(new[k].cpu(), populated_vector_store[k].cpu())

    def test_load_marks_cache_dirty(self, tmp_path, populated_vector_store, fake_encoder, fake_decoder):
        """Loading invalidates the similarity cache."""
        path = tmp_path / "vd.pt"
        populated_vector_store.save(str(path))
        new = VectorStore(fake_encoder, fake_decoder)
        new.load(str(path))
        assert getattr(new, "_cache_dirty") is True


# ---------------------------------------------------------------------------
# Equality and hash
# ---------------------------------------------------------------------------


class TestEqualityAndHash:
    """`__eq__` requires identical contents; `__hash__` is stable."""

    def test_empty_stores_equal(self, fake_encoder, fake_decoder):
        """Two empty stores are equal."""
        a = VectorStore(fake_encoder, fake_decoder)
        b = VectorStore(fake_encoder, fake_decoder)
        assert a == b

    def test_equal_when_tensor_contents_match(self, fake_encoder, fake_decoder):
        """Stores with identical contents are equal."""
        a = VectorStore(fake_encoder, fake_decoder)
        b = VectorStore(fake_encoder, fake_decoder)
        v = np.zeros(4, dtype=np.float32)
        a.add_vector("X", v)
        b.add_vector("X", v)
        assert a == b

    def test_unequal_when_tensor_contents_differ(self, fake_encoder, fake_decoder):
        """Stores whose vectors differ are unequal."""
        a = VectorStore(fake_encoder, fake_decoder)
        b = VectorStore(fake_encoder, fake_decoder)
        a.add_vector("X", np.zeros(4, dtype=np.float32))
        b.add_vector("X", np.ones(4, dtype=np.float32))
        assert a != b

    def test_unequal_when_keys_differ(self, fake_encoder, fake_decoder):
        """Stores with different keys are unequal."""
        a = VectorStore(fake_encoder, fake_decoder)
        b = VectorStore(fake_encoder, fake_decoder)
        a.add_vector("X", np.zeros(4, dtype=np.float32))
        b.add_vector("Y", np.zeros(4, dtype=np.float32))
        assert a != b

    def test_hash_stable_for_equal_stores(self, fake_encoder, fake_decoder):
        """Equal stores hash equally, honoring the eq/hash contract."""
        a = VectorStore(fake_encoder, fake_decoder)
        b = VectorStore(fake_encoder, fake_decoder)
        v = np.zeros(4, dtype=np.float32)
        a.add_vector("X", v)
        b.add_vector("X", v)
        assert hash(a) == hash(b)

    def test_unequal_to_non_store(self, empty_vector_store):
        """A store is unequal to a non-store object."""
        assert (empty_vector_store == 123) is False
