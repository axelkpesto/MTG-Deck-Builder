"""Tests for helpers in `backend.ml.tagging_model`.

We avoid the heavy training pipeline (`build_dataset`, `prepare_dataset`, `train`)
and instead exercise the pure data structures and prediction post-processing
that the API uses at runtime:

- `MLP.forward` shape contract
- `VectorsDataset` indexing
- `predicted_scores_from_probabilities` threshold + sort
- `save_model` / `load_model` round-trip
"""
import numpy as np
import torch

from backend.ml.tagging_model import (
    MLP,
    VectorsDataset,
    load_model,
    predicted_scores_from_probabilities,
    save_model,
)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class TestMLP:
    """`MLP.forward` produces correctly shaped logits."""

    def test_output_shape_matches_args(self):
        """Forward output shape is (batch, output_dim)."""
        m = MLP(input_dim=16, output_dim=4, hidden=32)
        x = torch.zeros(7, 16)
        out = m(x)
        assert out.shape == (7, 4)

    def test_returns_tensor(self):
        """Forward returns a `torch.Tensor`."""
        m = MLP(input_dim=8, output_dim=2)
        out = m(torch.zeros(1, 8))
        assert isinstance(out, torch.Tensor)

    def test_default_hidden_dim(self):
        """The default hidden dimension still produces correctly shaped output."""
        m = MLP(input_dim=5, output_dim=2)
        out = m(torch.zeros(3, 5))
        assert out.shape == (3, 2)

    def test_forward_grad_disabled_under_no_grad(self):
        """Output under `torch.no_grad()` does not require gradients."""
        m = MLP(input_dim=4, output_dim=2)
        x = torch.zeros(1, 4)
        with torch.no_grad():
            out = m(x)
        assert not out.requires_grad


# ---------------------------------------------------------------------------
# VectorsDataset
# ---------------------------------------------------------------------------


class TestVectorsDataset:
    """Dataset wraps numpy arrays as tensors and supports indexing."""

    def test_len_matches_input(self):
        """Dataset length equals the number of input rows."""
        features = np.zeros((5, 4), dtype=np.float32)
        labels = np.zeros((5, 2), dtype=np.float32)
        ds = VectorsDataset(features, labels)
        assert len(ds) == 5

    def test_indexing_returns_pair(self):
        """Indexing returns the (features, labels) tensor pair for that row."""
        features = np.arange(12, dtype=np.float32).reshape(3, 4)
        labels = np.eye(3, dtype=np.float32)
        ds = VectorsDataset(features, labels)
        x, y = ds[1]
        assert x.shape == (4,)
        assert y.shape == (3,)
        # The 1st row of features is [4,5,6,7]
        assert torch.equal(x, torch.tensor([4.0, 5.0, 6.0, 7.0]))

    def test_returns_tensor_types(self):
        """Indexed features and labels are both tensors."""
        features = np.zeros((1, 2), dtype=np.float32)
        labels = np.zeros((1, 2), dtype=np.float32)
        ds = VectorsDataset(features, labels)
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)


# ---------------------------------------------------------------------------
# predicted_scores_from_probabilities
# ---------------------------------------------------------------------------


class TestPredictedScoresFromProbabilities:
    """Threshold-filter probs and return sorted (tag,score) pairs + flat list."""

    def test_filters_below_threshold(self):
        """Tags with probability below the threshold are dropped."""
        probs = np.array([0.9, 0.4, 0.6])
        names = ["a", "b", "c"]
        scores, predicted = predicted_scores_from_probabilities(
            probs, names, threshold=0.5
        )
        assert {s["tag"] for s in scores} == {"a", "c"}
        assert set(predicted) == {"a", "c"}

    def test_results_sorted_descending(self):
        """Scores are sorted descending and `predicted` matches that order."""
        probs = np.array([0.5, 0.9, 0.7])
        names = ["a", "b", "c"]
        scores, predicted = predicted_scores_from_probabilities(
            probs, names, threshold=0.0
        )
        s = [item["score"] for item in scores]
        assert s == sorted(s, reverse=True)
        # `predicted` is ordered to match `scores`
        assert predicted == [scores[0]["tag"], scores[1]["tag"], scores[2]["tag"]]

    def test_empty_when_all_below_threshold(self):
        """When every probability is below the threshold, results are empty."""
        probs = np.array([0.1, 0.2])
        scores, predicted = predicted_scores_from_probabilities(
            probs, ["a", "b"], threshold=0.9
        )
        assert scores == []
        assert predicted == []

    def test_inclusive_threshold(self):
        """A probability exactly at the threshold is included."""
        probs = np.array([0.5])
        _, predicted = predicted_scores_from_probabilities(
            probs, ["x"], threshold=0.5
        )
        assert predicted == ["x"]

    def test_score_field_is_python_float(self):
        """Each score field is a native Python float (not numpy)."""
        probs = np.array([0.8, 0.7], dtype=np.float32)
        scores, _ = predicted_scores_from_probabilities(probs, ["a", "b"], 0.0)
        for s in scores:
            assert isinstance(s["score"], float)


# ---------------------------------------------------------------------------
# save_model / load_model round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadModel:
    """`save_model` writes a checkpoint that `load_model` can reload."""

    def test_round_trip(self, tmp_path):
        """A saved model reloads with identical class names and outputs."""
        model = MLP(input_dim=4, output_dim=3, hidden=8)
        # Set known weights
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
                p.add_(0.5)

        class _FakeMlb:
            classes_ = ["a", "b", "c"]

        path = str(tmp_path / "model.pt")
        save_model(model, _FakeMlb(), path, {"input_dim": 4, "output_dim": 3, "hidden": 8})

        loaded, class_names = load_model(path)
        assert class_names == ["a", "b", "c"]

        # Output should match original on a deterministic input.
        x = torch.ones(1, 4)
        original_out = model(x).detach()
        loaded_out = loaded(x).detach()
        assert torch.allclose(original_out, loaded_out)

    def test_load_returns_eval_mode(self, tmp_path):
        """A reloaded model is returned in eval mode."""
        model = MLP(input_dim=2, output_dim=2)

        class _Mlb:
            classes_ = ["a", "b"]

        path = str(tmp_path / "m.pt")
        save_model(model, _Mlb(), path, {"input_dim": 2, "output_dim": 2, "hidden": 128})
        loaded, _ = load_model(path)
        assert loaded.training is False
