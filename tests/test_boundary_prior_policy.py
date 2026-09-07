import pytest

np = pytest.importorskip("numpy")

from lob_forge.boundary_prior_policy import decision_priors  # noqa: E402


def test_prior_estimates_use_only_released_labels_and_preserve_prefixes():
    training, previous = [0.2, 0.6, 0.2], [0.3, 0.4, 0.3]
    clock = np.arange(20) * 1000
    release = clock + 5100
    labels = np.zeros(20, dtype=int)
    full = decision_priors("online_300", training, previous, labels, clock, release)
    np.testing.assert_allclose(full[:6], np.tile(0.01 + 0.97 * np.array(previous), (6, 1)))
    assert full[6, 1] > full[5, 1]
    changed = labels.copy()
    changed[4:] = 1
    alternative = decision_priors("online_300", training, previous, changed, clock, release)
    np.testing.assert_array_equal(full[:10], alternative[:10])
    np.testing.assert_array_equal(
        full[:10], decision_priors("online_300", training, previous, labels[:10], clock[:10], release[:10])
    )
    np.testing.assert_allclose(full.sum(axis=1), 1)


def test_class_prior_policy_changes_decisions_without_changing_probabilities():
    p = np.array([[0.2, 0.5, 0.3]])
    current = decision_priors("training", [0.2, 0.6, 0.2], [0.3, 0.4, 0.3], [0], [0], [5100])
    previous = decision_priors("previous_noon", [0.2, 0.6, 0.2], [0.3, 0.4, 0.3], [0], [0], [5100])
    assert np.argmax(p / current) == 2
    assert np.argmax(p / previous) == 1
    np.testing.assert_array_equal(p, [[0.2, 0.5, 0.3]])
