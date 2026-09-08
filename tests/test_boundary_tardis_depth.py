import gzip
import json

import pytest

from lob_forge.boundary_tardis_depth import BinanceFuturesDepthState, capture_nanoseconds, iter_tardis_messages


def snapshot(update=10):
    return {"stream": "btcusdt@depthSnapshot", "generated": True,
            "data": {"lastUpdateId": update, "E": 1000, "T": 990,
                     "bids": [["100", "10"], ["99", "20"]], "asks": [["102", "10"], ["103", "20"]]}}


def delta(first, final, previous, bids=(), asks=()):
    return {"stream": "btcusdt@depth@0ms", "data": {"e": "depthUpdate", "s": "BTCUSDT", "U": first, "u": final,
        "pu": previous, "E": 1100 + final, "T": 1090 + final, "b": list(bids), "a": list(asks)}}


def test_futures_bridge_buffers_then_becomes_available_at_snapshot_capture_time():
    book = BinanceFuturesDepthState("BTCUSDT")
    assert not book.apply(1_110_000_000, delta(8, 9, 7, [["100", "8"]]))
    assert not book.apply(1_120_000_000, delta(10, 12, 9, [["100", "12"]]))
    assert not book.apply(1_130_000_000, delta(13, 15, 12, asks=[["102", "7"]]))
    assert book.snapshot(1) is None
    assert book.apply(1_600_000_000, snapshot())
    assert book.available_time_ns == 1_600_000_000
    assert book.publisher_time_ms == 1115
    assert book.last_u == 15
    assert book.snapshot(1) == [(102.0, 7.0, 100.0, 12.0)]
    assert book.applied_deltas == 2
    book.apply(1_700_000_000, delta(16, 18, 15, [["80", "0"]]))
    before = book.bids.copy(), book.asks.copy()
    with pytest.raises(ValueError, match="continuity"):
        book.apply(1_800_000_000, delta(20, 22, 19, [["100", "100"]]))
    assert (book.bids, book.asks) == before
    assert book.snapshot(1) is None


def test_futures_snapshot_overlap_is_not_spot_plus_one_and_frontier_is_not_inferred():
    book = BinanceFuturesDepthState("BTCUSDT")
    book.apply(1_500_000_000, snapshot())
    with pytest.raises(ValueError, match="overlap"):
        book.apply(1_600_000_000, delta(11, 12, 10))
    book.disconnect()
    book.apply(1_700_000_000, snapshot())
    book.apply(1_800_000_000, delta(9, 10, 8))
    assert book.snapshot(2) is not None
    book.apply(1_900_000_000, delta(11, 13, 10, [["100", "0"], ["99", "0"], ["98", "30"]]))
    assert book.snapshot(1) is None  # A positive update beyond the snapshot frontier cannot establish missing levels.
    book.disconnect()
    assert book.snapshot(1) is None


def test_provider_capture_parser_preserves_submicrosecond_order_and_disconnects(tmp_path):
    first = "2023-06-01T00:00:00.0088083Z"
    second = "2023-06-01T00:00:00.0088084Z"
    assert capture_nanoseconds(second) - capture_nanoseconds(first) == 100
    body = first + " " + json.dumps(snapshot()) + "\n\n" + second + " " + json.dumps(delta(9, 10, 8)) + "\n"
    plain, compressed = tmp_path / "plain.gz", tmp_path / "compressed.ndjson"
    plain.write_text(body)
    compressed.write_bytes(gzip.compress(body.encode()))
    a, b = list(iter_tardis_messages(plain)), list(iter_tardis_messages(compressed))
    assert a == b and a[1] == (None, None)
    plain.write_text(second + " " + json.dumps(snapshot()) + "\n" + first + " " + json.dumps(snapshot()) + "\n")
    with pytest.raises(ValueError, match="capture order"):
        list(iter_tardis_messages(plain))
