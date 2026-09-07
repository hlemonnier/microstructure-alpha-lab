import copy
import math

import pytest

np = pytest.importorskip("numpy")

from lob_forge.boundary_bybit_flow import FLOW_FIELDS, native_bybit_flow_events, sample_native_bybit_flow  # noqa: E402


def _messages():
    def payload(kind, t, u, bids, asks):
        return {"topic": "orderbook.500.BTCUSDT", "type": kind, "ts": t,
                "data": {"s": "BTCUSDT", "u": u, "seq": u + 100, "b": bids, "a": asks}}
    return [payload("snapshot", 1000, 1, [[str(100 - i), "10"] for i in range(25)], [[str(102 + i), "10"] for i in range(25)]),
            payload("delta", 1500, 2, [["100", "13"]], []),
            payload("delta", 1800, 3, [["100", "10"]], []),
            payload("delta", 2200, 4, [["70", "8"]], []),
            payload("snapshot", 2500, 10, [[str(100 - i), "10"] for i in range(25)], [[str(102 + i), "10"] for i in range(25)]),
            payload("delta", 2700, 11, [], [["102", "8"]])]


def test_native_flow_retains_round_trip_activity_and_excludes_unknown_frontier_levels():
    events, checks = native_bybit_flow_events(_messages(), "BTCUSDT")
    assert events["flows"].shape == (6, len(FLOW_FIELDS))
    # A +3/-3 displayed-quantity cycle is absent from endpoint pressure but its
    # gross flow is preserved. Distance is half a spread from the old midpoint.
    np.testing.assert_allclose(events["flows"][1, [0, 4, 8]], [3 * math.exp(-0.5 / w) for w in (1, 5, 20)])
    np.testing.assert_array_equal(events["flows"][1, [0, 4, 8]], events["flows"][2, [1, 5, 9]])
    assert events["flows"][1:3, 12].sum() == 0
    assert events["flows"][1:3, 13].sum() == 6
    assert checks["unresolved_level_updates"] == 1
    assert not events["flows"][[0, 4]].any()
    assert not events["flows"][3, :12].any()


def test_native_flow_sampling_uses_strict_delay_boundaries_and_resets_without_future_dependence():
    messages = _messages()
    events, _ = native_bybit_flow_events(messages, "BTCUSDT")
    clock = np.array([1900, 1901, 2400, 3000], dtype=np.int64)
    actual = sample_native_bybit_flow(events, clock, delays_ms=(100,), windows_ms=(800,))
    assert actual["window_available"][:, 0, 0].tolist() == [True, True, True, False]
    assert actual["flow_totals"][0, 0, 0, 12] == 3
    assert actual["flow_totals"][1, 0, 0, 12] == 0
    np.testing.assert_array_equal(actual["source_publisher_times"][:, 0], [1500, 1800, 2200, 2700])
    prefix, _ = native_bybit_flow_events(messages[:4], "BTCUSDT")
    shorter = sample_native_bybit_flow(prefix, clock[:3], delays_ms=(100,), windows_ms=(800,))
    np.testing.assert_array_equal(shorter["flow_totals"], actual["flow_totals"][:3])
    np.testing.assert_array_equal(shorter["window_available"], actual["window_available"][:3])


def test_native_flow_quantity_price_units_and_atomic_sequence_checks():
    original = _messages()
    transformed = copy.deepcopy(original)
    for message in transformed:
        for side in ("b", "a"):
            message["data"][side] = [[str(float(p) * 7), str(float(q) * 11)] for p, q in message["data"][side]]
    a, _ = native_bybit_flow_events(original, "BTCUSDT")
    b, _ = native_bybit_flow_events(transformed, "BTCUSDT")
    np.testing.assert_allclose(b["flows"][:, :14], a["flows"][:, :14] * 11)
    np.testing.assert_allclose(b["flows"][:, 14:], a["flows"][:, 14:])
    np.testing.assert_allclose(b["top_depth"], a["top_depth"] * 11)
    broken = copy.deepcopy(original)
    broken[2]["data"]["u"] += 1
    with pytest.raises(ValueError, match="consecutive"):
        native_bybit_flow_events(broken, "BTCUSDT")
