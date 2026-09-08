import pytest

np = pytest.importorskip("numpy")

from lob_forge.boundary_prior_equivalence import assert_decision_priors_equal  # noqa: E402


def test_constant_and_repeated_priors_have_exactly_the_same_decisions():
    prior = np.array([0.2, 0.6, 0.2])
    repeated = np.tile(prior, (7070, 1))
    assert_decision_priors_equal(repeated, prior, rows=7070)
    assert_decision_priors_equal(prior, repeated, rows=7070)
    probabilities = np.random.default_rng(52).dirichlet([1, 1, 1], size=7070)
    np.testing.assert_array_equal(np.argmax(probabilities / prior, axis=1),
                                  np.argmax(probabilities / repeated, axis=1))


def test_dynamic_priors_are_compared_at_each_query_without_averaging():
    dynamic = np.array([[0.2, 0.6, 0.2], [0.3, 0.4, 0.3]])
    assert_decision_priors_equal(dynamic, dynamic.copy(), rows=2)
    with pytest.raises(AssertionError):
        assert_decision_priors_equal(dynamic, dynamic[::-1], rows=2)
    with pytest.raises(AssertionError):
        assert_decision_priors_equal(dynamic, dynamic.mean(axis=0), rows=2)


def test_one_changed_bit_in_one_query_is_not_equivalent():
    prior = np.array([0.2, 0.6, 0.2])
    changed = np.tile(prior, (7, 1))
    changed[5, 1] = np.nextafter(changed[5, 1], np.inf)
    with pytest.raises(AssertionError):
        assert_decision_priors_equal(changed, prior, rows=7)


def test_shape_and_invalid_value_errors_are_not_hidden_by_broadcasting():
    prior = np.array([0.2, 0.6, 0.2])
    for invalid in (prior[None, :], np.tile(prior, (3, 1)), prior[:, None],
                    np.array([0.0, 0.6, 0.4]), np.array([np.nan, 0.6, 0.2]),
                    np.array([np.inf, 0.6, 0.2]), np.array([-0.1, 0.6, 0.5]),
                    np.array(["0.2", "0.6", "0.2"])):
        with pytest.raises(ValueError):
            assert_decision_priors_equal(invalid, prior, rows=2)
    for rows in (0, -1, True, 2.0):
        with pytest.raises(ValueError):
            assert_decision_priors_equal(prior, prior, rows=rows)
    assert_decision_priors_equal(prior[None, :], prior, rows=1)
