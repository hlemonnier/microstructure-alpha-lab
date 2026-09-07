import math

import pytest

from lob_forge.label_math import price_movement_label


@pytest.mark.parametrize("quotes,tick", [
    ((26800.1, 26800.2, 26800.2, 26800.3), 0.1),
    ((1800.01, 1800.02, 1800.02, 1800.03), 0.01),
    ((100.1, 100.5, 100.3, 100.7), 0.1),
])
def test_exact_threshold_ties_and_reversed_moves_are_neutral(quotes, tick):
    eb, ea, fb, fa = quotes
    assert price_movement_label(*quotes, min_tick=tick) == 0
    assert price_movement_label(fb, fa, eb, ea, min_tick=tick) == 0


def test_strict_threshold_does_not_round_away_real_decimal_moves():
    assert price_movement_label(100.1, 100.2, 100.2, 100.30000000000001, min_tick=0.1) == 1
    assert price_movement_label(100.2, 100.30000000000001, 100.1, 100.2, min_tick=0.1) == -1
    assert price_movement_label(100, 102, 101, 103, min_tick=0.5) == 0
    assert price_movement_label(100, 102, 101, 103, min_tick=0.5, mode="one_tick") == 1
    assert price_movement_label(100, 102, 100, 102, min_tick=0, mode="zero") == 0
    assert price_movement_label(100, 102, 100, 102.00000000000001, min_tick=0, mode="zero") == 1


def test_invalid_label_inputs_fail_explicitly():
    for tick in [-1, math.nan, math.inf]:
        with pytest.raises(ValueError):
            price_movement_label(100, 101, 100, 101, min_tick=tick)
    for quotes in [(0, 1, 100, 101), (101, 100, 100, 101), (100, 101, math.nan, 102)]:
        with pytest.raises(ValueError):
            price_movement_label(*quotes, min_tick=0.1)
    with pytest.raises(ValueError, match="threshold"):
        price_movement_label(100, 101, 100, 101, min_tick=0.1, mode="invalid")


def test_vector_exact_decimal_labels_match_scalar_with_overflow_fallback():
    np = pytest.importorskip("numpy")
    from lob_forge.label_math import exact_label_array

    # Fine decimal resolution combined with large quotes exceeds int64 after scaling.
    quotes = np.array([
        [26800.1, 26800.2, 26800.2, 26800.3],
        [100.1, 100.2, 100.2, 100.30000000000001],
        [1e20, 1.1e20, 1e20, 1.1000000000000002e20],
        [100.125, 100.2, 100.2, 100.275],
    ])
    for mode in ["half_spread", "one_tick", "zero"]:
        expected = [price_movement_label(*row, min_tick=0.1, mode=mode) for row in quotes]
        np.testing.assert_array_equal(exact_label_array(*quotes.T, min_tick=0.1, mode=mode), expected)
    assert exact_label_array([], [], [], [], min_tick=0.1).shape == (0,)
    with pytest.raises(ValueError, match="Aligned"):
        exact_label_array([100], [101, 102], [100], [101], min_tick=0.1)
    with pytest.raises(ValueError, match="Finite"):
        exact_label_array([100], [101], [np.inf], [101], min_tick=0.1)
