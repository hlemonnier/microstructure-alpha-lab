import pytest

np = pytest.importorskip("numpy")

from lob_forge.boundary_bybit_depth import BybitDepthState, sample_bybit_depth  # noqa: E402
from lob_forge.data_sources import _bybit_orderbook_payload_to_rows  # noqa: E402
from lob_forge.l2_replay import AtomicOrderBookReplayer  # noqa: E402


def message(timestamp, update, bids, asks, kind="delta"):
    return {"topic": "orderbook.500.BTCUSDT", "type": kind, "ts": timestamp,
        "data": {"s": "BTCUSDT", "u": update, "seq": 10 * update, "b": bids, "a": asks}}


def test_native_depth_state_matches_existing_atomic_replay_and_fails_closed():
    records = [message(100, 1, [[99, 2], [98, 3]], [[101, 4], [102, 5]], "snapshot"),
        message(200, 2, [[99, 0], [100, 7]], [[101, 3]]),
        message(200, 3, [[100, 6]], [[101, 0], [100.5, 2]])]
    fast, reference = BybitDepthState("BTCUSDT"), AtomicOrderBookReplayer()
    for record in records:
        fast.apply(record)
        ref = reference.apply_event(list(_bybit_orderbook_payload_to_rows(record, default_symbol="BTCUSDT")))
        assert fast.bids == reference.bids and fast.asks == reference.asks
        np.testing.assert_array_equal(fast.snapshot(2), [[ref.asks[i].price, ref.asks[i].size, ref.bids[i].price, ref.bids[i].size] for i in range(2)])
    before = (dict(fast.bids), dict(fast.asks))
    with pytest.raises(ValueError):
        fast.apply(message(300, 5, [[100, 1]], []))
    assert not fast.initialized and (fast.bids, fast.asks) == before
    with pytest.raises(ValueError):
        fast.apply(message(300, 4, [[100, 1]], []))
    fast.apply(message(400, 10, [[99, 2]], [[101, 3]], "snapshot"))
    with pytest.raises(ValueError):
        fast.apply(message(500, 11, [[102, 1]], []))
    assert fast.bids == {99: 2} and not fast.initialized


def test_depth_sampling_delays_exclude_timestamp_ties_and_preserve_prefixes():
    rows = [message(100, 1, [[99, 2]], [[101, 4]], "snapshot"),
        message(200, 2, [[99, 3]], []), message(200, 3, [[99, 5]], []),
        message(400, 4, [[99, 7]], [])]
    clocks = np.array([100, 200, 201, 300, 301, 401])
    result, evidence = sample_bybit_depth(rows, clocks, "BTCUSDT", depth=1, delays_ms=(0, 100))
    assert not result["available"][0].any()
    np.testing.assert_array_equal(result["depth"][:, 0, 0, 3], [0, 2, 5, 5, 5, 7])
    np.testing.assert_array_equal(result["depth"][:, 1, 0, 3], [0, 0, 2, 2, 5, 5])
    shorter, _ = sample_bybit_depth(rows[:3], clocks[:5], "BTCUSDT", depth=1, delays_ms=(0, 100))
    for k in ("depth", "publisher_times", "update_ids", "sequences", "available"):
        np.testing.assert_array_equal(result[k][:5], shorter[k])
    assert evidence["messages"] == 4
    with pytest.raises(ValueError):
        sample_bybit_depth([rows[0], rows[2]], clocks, "BTCUSDT", depth=1)
