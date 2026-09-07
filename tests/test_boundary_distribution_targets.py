import pytest

np = pytest.importorskip("numpy")

from lob_forge.boundary_distribution_targets import COARSE_CLASSES, coarse_probabilities, exact_move_bins  # noqa: E402
from lob_forge.label_math import price_movement_label  # noqa: E402


def test_move_bins_keep_exact_ties_magnitudes_and_coarse_label_parity():
    # The minimum-tick floor sets a mid-price threshold of 0.1 here.
    moves = np.array([-.41, -.4, -.21, -.2, -.11, -.1, -.01, 0, .01, .1, .11, .2, .21, .4, .41])
    eb, ea = np.full(len(moves), 100.0), np.full(len(moves), 100.1)
    fb, fa = np.round(eb + moves, 2), np.round(ea + moves, 2)
    bins = exact_move_bins(eb, ea, fb, fa, min_tick=.1)
    np.testing.assert_array_equal(bins, [0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 8])
    np.testing.assert_array_equal(COARSE_CLASSES[bins], [price_movement_label(a,b,c,d,min_tick=.1) for a,b,c,d in zip(eb,ea,fb,fa)])
    reflected = exact_move_bins(200-ea, 200-eb, 200-fa, 200-fb, min_tick=.1)
    np.testing.assert_array_equal(reflected, 8-bins)


def test_move_bins_use_spread_floor_and_arbitrary_precision_without_epsilon():
    bins = exact_move_bins([100]*4, [102]*4, [101,102,104,104.0000000000001], [103,104,106,106.0000000000001], min_tick=.1)
    np.testing.assert_array_equal(bins, [5,6,7,8])
    huge = exact_move_bins([1e18], [1e18], [1e18], [1e18], min_tick=1e-18)
    np.testing.assert_array_equal(huge, [4])
    with pytest.raises(ValueError):
        exact_move_bins([100], [99], [100], [101], min_tick=.1)
    with pytest.raises(ValueError):
        exact_move_bins([100], [101], [100], [101], min_tick=0)


def test_coarse_probability_aggregation_and_weighted_posterior_recovery():
    natural = np.array([[.04,.06,.1,.1,.3,.1,.1,.12,.08]])
    priors = np.array([.25,.5,.25])
    weighted = natural / priors[COARSE_CLASSES + 1]
    weighted /= weighted.sum(axis=1, keepdims=True)
    recovered = coarse_probabilities(weighted) * priors
    recovered /= recovered.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(recovered, coarse_probabilities(natural), rtol=0, atol=1e-15)
    np.testing.assert_allclose(recovered.sum(axis=1), 1)
    with pytest.raises(ValueError):
        coarse_probabilities(np.ones((2, 3))/3)
