import pytest

np = pytest.importorskip("numpy")

from lob_forge.boundary_multihorizon_targets import HORIZONS_MS, resolve_multihorizon_targets  # noqa: E402
from lob_forge.label_math import price_movement_label  # noqa: E402


def test_exact_multihorizon_resolution_ties_release_times_and_missing_quotes():
    t = np.array([100, 100, 1100, 2100, 5100, 10100, 30100])
    bid = np.array([10, 10.3, 10.1, 9.8, 10.1, 10.5, 9.5])
    ask = bid + .1
    # Explicit decimals avoid introducing unintended floating-point input prices.
    ask[:] = [10.1, 10.4, 10.2, 9.9, 10.2, 10.6, 9.6]
    decisions = np.array([0, 1, 30000, 40000])
    result = resolve_multihorizon_targets(t, bid, ask, decisions, min_tick=.1)
    np.testing.assert_array_equal(result["labels"][0], [0, 0, -1, 1, -1])
    np.testing.assert_array_equal(result["future_times"][0], [5100, 1100, 2100, 10100, 30100])
    assert result["entry_mids"][0] == (bid[0] + ask[0]) / 2
    assert result["entry_times"][1] == 1100
    np.testing.assert_array_equal(result["available"][2:], False)
    np.testing.assert_array_equal(result["labels"][2:], -2)
    assert result["entry_available"][2] and not result["entry_available"][3]
    np.testing.assert_array_equal(result["decision_times"], decisions)
    for row, decision in enumerate(decisions):
        for head, horizon in enumerate(HORIZONS_MS):
            if result["available"][row, head]:
                entry = np.searchsorted(t, decision + 100)
                future = np.searchsorted(t, decision + 100 + horizon)
                assert result["labels"][row, head] == price_movement_label(bid[entry], ask[entry], bid[future], ask[future], min_tick=.1)


def test_future_target_prefix_matches_every_already_resolved_outcome():
    t = np.arange(100, 40001, 100, dtype=np.int64)
    bid = 10 + (np.arange(len(t)) % 7) / 10
    ask = 11 + (np.arange(len(t)) % 7) / 10
    d = np.arange(0, 9000, 1000, dtype=np.int64)
    full = resolve_multihorizon_targets(t, bid, ask, d, min_tick=.1)
    stop = 8000
    truncated = resolve_multihorizon_targets(t[t <= stop], bid[t <= stop], ask[t <= stop], d, min_tick=.1)
    available = full["available"] & (full["future_times"] <= stop)
    np.testing.assert_array_equal(truncated["available"], available)
    for field in ("labels", "future_times", "future_mids"):
        np.testing.assert_array_equal(truncated[field][available], full[field][available])


def test_future_target_resolution_rejects_invalid_clocks_and_price_shapes():
    with pytest.raises(ValueError, match="Integer"):
        resolve_multihorizon_targets(np.array([100.5]), [10], [11], np.array([0]), min_tick=.1)
    with pytest.raises(ValueError, match="Chronological"):
        resolve_multihorizon_targets(np.array([200, 100]), [10, 10], [11, 11], np.array([0]), min_tick=.1)
    with pytest.raises(ValueError, match="uncrossed"):
        resolve_multihorizon_targets(np.array([100]), [11], [10], np.array([0]), min_tick=.1)
    with pytest.raises(ValueError, match="nonoverflowing"):
        resolve_multihorizon_targets(np.array([100]), [10], [11], np.array([np.iinfo(np.int64).max]), min_tick=.1)
    with pytest.raises(ValueError, match="nonoverflowing"):
        resolve_multihorizon_targets(np.array([2**63], dtype=np.uint64), [10], [11], np.array([0]), min_tick=.1)
