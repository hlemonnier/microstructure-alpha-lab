import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from lob_forge.boundary_neural import (  # noqa: E402
    NeuralForecaster,
    SequenceNormalizer,
    causal_window_indices,
    natural_posteriors,
)
from lob_forge.ml_models import build_torch_sequence_classifier  # noqa: E402


def test_sequence_windows_do_not_cross_session_gaps_or_see_future():
    times = np.array([1000, 2000, 3000, 9000, 10000])
    windows = causal_window_indices(times, 3)
    np.testing.assert_array_equal(windows, [[0, 0, 0], [0, 0, 1], [0, 1, 2], [3, 3, 3], [3, 3, 4]])
    np.testing.assert_array_equal(causal_window_indices(times[:3], 3), windows[:3])


def test_balanced_posterior_correction_and_frozen_neural_checkpoint(tmp_path):
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(1)
    matrix = np.arange(60, dtype=float).reshape(20, 3)
    normalizer = SequenceNormalizer.fit(matrix[:10])
    frozen_mean = normalizer.mean.copy()
    normalizer.transform(matrix * 1000)
    np.testing.assert_array_equal(frozen_mean, normalizer.mean)
    priors = np.array([0.2, 0.6, 0.2])
    np.testing.assert_allclose(natural_posteriors(np.full((1, 3), 1 / 3), priors), priors[None, :])
    network = build_torch_sequence_classifier(window=4, feature_count=3, model_name="sequence_tcn", hidden_size=4)
    model = NeuralForecaster(network, normalizer, ["a", "b", "c"], priors, "sequence_tcn", 4, 4)
    indices = causal_window_indices(np.arange(20) * 1000, 4)
    expected = model.predict_proba(matrix, indices)
    model.save(tmp_path / "model.pt")
    restored = NeuralForecaster.load(tmp_path / "model.pt")
    np.testing.assert_allclose(restored.predict_proba(matrix, indices), expected, atol=0, rtol=0)
    np.testing.assert_allclose(expected.sum(axis=1), 1)
