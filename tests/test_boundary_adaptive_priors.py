import pytest

np = pytest.importorskip("numpy")

from lob_forge.boundary_adaptive_priors import (  # noqa: E402
    PRIOR_FLOOR, all_policies, forecast_priors, label_priors,
)


def test_label_priors_match_explicit_released_observation_weights():
    origins = np.array([0, 1000, 2000, 3000, 4000])
    release = np.array([2100, 3500, 3500, 12000, 13000])
    labels = np.array([-1, 0, 1, 1, -1])
    queries = np.array([1000, 2100, 3500, 5000, 13000])
    for half_life in (None, 2.0, 300.0):
        actual = label_priors(labels, origins, release, queries, half_life_seconds=half_life)
        expected = []
        for now in queries:
            decay = 1 if half_life is None else np.exp2(-now / (1000 * half_life))
            counts = np.full(3, 20.0 * decay)
            for y, origin, available in zip(labels, origins, release, strict=True):
                if available <= now:
                    counts[y + 1] += 1 if half_life is None else np.exp2(-(now - origin) / (1000 * half_life))
            expected.append(PRIOR_FLOOR + (1 - 3 * PRIOR_FLOOR) * counts / counts.sum())
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)


def test_same_day_estimates_are_prefix_causal_at_actual_release_time():
    origins = np.arange(100) * 1000
    release = origins + 5100
    labels = np.arange(100) % 3 - 1
    queries = np.arange(50, 100) * 1000
    full = label_priors(labels, origins, release, queries, half_life_seconds=300)
    altered = labels.copy()
    altered[54:] = 1
    other = label_priors(altered, origins, release, queries, half_life_seconds=300)
    np.testing.assert_array_equal(full[:10], other[:10])
    np.testing.assert_array_equal(full[:10], label_priors(labels[:60], origins[:60], release[:60], queries[:10], half_life_seconds=300))
    # Query 60s is the first to know the changed label originating at 54s.
    assert not np.array_equal(full[10], other[10])
    np.testing.assert_allclose(full.sum(axis=1), 1)


def test_forecast_marginals_include_only_available_forecasts():
    p = np.tile([0.1, 0.7, 0.2], (10, 1))
    clock = np.arange(10) * 1000
    initial = np.array([0.2, 0.5, 0.3])
    current = forecast_priors(p, clock, initial, half_life_seconds=300)
    np.testing.assert_allclose(current[0], PRIOR_FLOOR + (1 - 3 * PRIOR_FLOOR) * (60 * initial + p[0]) / 61)
    altered = p.copy()
    altered[5:] = [0.8, 0.1, 0.1]
    other = forecast_priors(altered, clock, initial, half_life_seconds=300)
    np.testing.assert_array_equal(current[:5], other[:5])
    np.testing.assert_array_equal(current[:5], forecast_priors(p[:5], clock[:5], initial, half_life_seconds=300))
    assert not np.array_equal(current[5], other[5])
    np.testing.assert_array_equal(p, np.tile([0.1, 0.7, 0.2], (10, 1)))


def test_decision_family_retains_static_controls_and_positive_probabilities():
    p = np.array([[0.1, 0.7, 0.2], [0.2, 0.6, 0.2]])
    clock = [10000, 11000]
    labels = {name: np.tile([0.2, 0.6, 0.2], (2, 1)) for name in ("labels_cumulative", "labels_300", "labels_900", "labels_3600")}
    policies = all_policies(p, clock, [0.3, 0.4, 0.3], [0.25, 0.5, 0.25], labels)
    assert len(policies) == 11
    np.testing.assert_array_equal(policies["training"], [[0.3, 0.4, 0.3]] * 2)
    np.testing.assert_array_equal(policies["registered"], [[0.25, 0.5, 0.25]] * 2)
    for estimated in policies.values():
        np.testing.assert_allclose(estimated.sum(axis=1), 1)
        assert (estimated > 0).all()
    np.testing.assert_allclose(policies["hybrid_300"], (policies["labels_300"] + policies["forecast_300"]) / 2)


def test_rejects_bad_labels_times_and_probabilities_and_handles_empty_history():
    for y, origin, release, query in [
        ([0.5], [0], [5100], [10000]), ([0], [0], [0], [10000]),
        ([0], [0], [5100.5], [10000]), ([0], [0.5], [5100], [10000]),
        ([0], [0], [5100], [10000, 10000]), ([0, 1], [0, 1000], [6000, 5100], [10000]),
    ]:
        with pytest.raises(ValueError):
            label_priors(y, origin, release, query)
    np.testing.assert_allclose(label_priors([], [], [], [0, 1000], half_life_seconds=300), np.full((2, 3), 1 / 3))
    with pytest.raises(ValueError):
        forecast_priors([[0.1, 0.2, 0.3]], [0], [0.2, 0.6, 0.2], half_life_seconds=300)
    with pytest.raises(ValueError):
        label_priors([0], [0], [5100], [10000], half_life_seconds=0)
