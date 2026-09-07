import pytest

np = pytest.importorskip("numpy")

from lob_forge.boundary_online import delayed_calibration  # noqa: E402


def test_delayed_online_calibration_cannot_observe_unreleased_labels():
    times = np.arange(20) * 1000
    release = times + 5100
    p = np.full((20, 3), 1 / 3)
    labels = np.ones(20, dtype=int)
    full, state = delayed_calibration(p, labels, times, release, learning_rate=0.1)
    np.testing.assert_allclose(full[:6], p[:6])
    assert full[6, 2] > 1 / 3
    assert state["updates"] == 14
    prefix, _ = delayed_calibration(p[:10], labels[:10], times[:10], release[:10], learning_rate=0.1)
    np.testing.assert_array_equal(prefix, full[:10])
    modified = labels.copy()
    modified[4:] = -1
    changed, _ = delayed_calibration(p, modified, times, release, learning_rate=0.1)
    np.testing.assert_array_equal(changed[:10], full[:10])
    assert changed[10, 0] > full[10, 0]


def test_zero_rate_preserves_exact_forecasts_and_rejects_early_feedback():
    p = np.array([[0.1, 0.5, 0.4], [0.3, 0.4, 0.3]])
    result, _ = delayed_calibration(p, [-1, 1], [0, 1000], [5100, 6100], learning_rate=0)
    np.testing.assert_array_equal(result, p)
    with pytest.raises(ValueError, match="strictly delayed"):
        delayed_calibration(p, [-1, 1], [0, 1000], [0, 1000], learning_rate=0.1)
