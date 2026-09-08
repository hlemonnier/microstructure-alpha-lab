import random

import pytest

from lob_forge.boundary_tardis_depth import BinanceFuturesDepthState, capture_nanoseconds
from lob_forge.boundary_tardis_fast import FastBinanceFuturesDepthState, fast_capture_nanoseconds


def test_fast_replay_matches_reference_for_additions_deletions_snapshots_and_atomic_failure():
    reference, fast = BinanceFuturesDepthState("BTCUSDT"), FastBinanceFuturesDepthState("BTCUSDT")
    initial = {"stream": "btcusdt@depthSnapshot", "generated": True, "data": {"lastUpdateId": 10, "E": 1000, "T": 999,
        "bids": [[str(100 - i), "10"] for i in range(30)], "asks": [[str(102 + i), "10"] for i in range(30)]}}
    for book in (reference, fast):
        book.apply(1500_000_000, initial)
    rng = random.Random(20260907)
    previous = 8
    for i in range(100):
        final = 10 + 2 * i
        data = {"e": "depthUpdate", "s": "BTCUSDT", "U": final - 1, "u": final, "pu": previous,
                "E": 1600 + i, "T": 1599 + i, "b": [[str(100 - rng.randrange(35)), str(rng.choice([0, 2, 7, 11]))]],
                "a": [[str(102 + rng.randrange(35)), str(rng.choice([0, 3, 6, 12]))]]}
        message = {"stream": "btcusdt@depth@0ms", "data": data}
        for book in (reference, fast):
            book.apply((1601 + i) * 1_000_000, message)
        assert fast.bids == reference.bids and fast.asks == reference.asks
        assert fast.last_u == reference.last_u and fast.available_time_ns == reference.available_time_ns
        for levels in (1, 5, 25):
            assert fast.snapshot(levels) == reference.snapshot(levels)
        previous = final
    broken = {"stream": "btcusdt@depth@0ms", "data": {**data, "U": previous + 1, "u": previous + 2, "pu": previous, "b": [["200", "10"]]}}
    for book in (reference, fast):
        before = book.bids.copy(), book.asks.copy()
        with pytest.raises(ValueError, match="crossed"):
            book.apply(2000_000_000, broken)
        assert (book.bids, book.asks) == before and not book.ready
        book.disconnect()
        book.apply(2100_000_000, initial)
    assert fast.snapshot(1) is None and reference.snapshot(1) is None


def test_cached_capture_parser_preserves_reference_validation_and_nanoseconds():
    for text in ("2023-06-01T00:00:00.0088083Z", "2023-06-01T00:00:00.0088084Z", "2023-06-01T00:00:01.9Z", "2024-02-29T23:59:59.123456789Z"):
        assert fast_capture_nanoseconds(text) == capture_nanoseconds(text)
    for text in ("2023-02-29T00:00:00.001Z", "2023-06-01T00:00:00.0000000001Z", "2023-06-01T00:00:00.001+01:00"):
        with pytest.raises(ValueError):
            fast_capture_nanoseconds(text)
