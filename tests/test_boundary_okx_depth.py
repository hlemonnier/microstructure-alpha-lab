import gzip
import io
import json
import tarfile
from decimal import Decimal

import pytest

np = pytest.importorskip("numpy")

from lob_forge.boundary_okx_depth import CountedDepthState, counted_archive_messages, sample_counted_depth  # noqa: E402
from lob_forge.boundary_okx_source import parse_counted_depth  # noqa: E402


def _message(now=100, *, action="snapshot", asks=None, bids=None):
    return parse_counted_depth(json.dumps({"instId": "BTC-USDT-SWAP", "action": action, "ts": str(now),
        "asks": asks if asks is not None else [[str(p), "12", "3"] for p in (101, 102, 103)],
        "bids": bids if bids is not None else [[str(p), "10", "2"] for p in (100, 99, 98)]}),
        instrument="BTC-USDT-SWAP")


def _sample(messages, clock, **kwargs):
    return sample_counted_depth(messages, np.array(clock, dtype=np.int64), "BTC-USDT-SWAP",
        depth=3, capacity=3, **kwargs)


def test_counted_updates_are_atomic_absolute_and_keep_decimal_prices():
    book = CountedDepthState("BTC-USDT-SWAP", capacity=3)
    book.apply(_message())
    # A temporary lock occurs if asks are inspected before the matching bid deletion.
    book.apply(_message(110, action="update", asks=[["100", "9", "7"]],
        bids=[["100", "0", "0"], ["99.500000000000000001", "8", "6"]]))
    assert book.initialized and book.asks.top(1)[0].order_count == 7
    assert book.bids.top(1)[0].price == Decimal("99.500000000000000001")
    book.apply(_message(120, action="update", asks=[["100", "3", "1"]], bids=[]))
    assert book.asks.top(1)[0].quantity == 3  # Absolute replacement, not +3.


def test_visible_frontiers_only_shrink_until_a_full_snapshot():
    book = CountedDepthState("BTC-USDT-SWAP", capacity=3)
    book.apply(_message())
    book.apply(_message(110, action="update", asks=[["100.5", "1", "1"]], bids=[["99.5", "1", "1"]]))
    assert book.asks.frontier == 102 and book.bids.frontier == 99
    book.apply(_message(120, action="update", asks=[["100.5", "0", "0"], ["101", "0", "0"]],
        bids=[["100", "0", "0"], ["99.5", "0", "0"]]))
    # Refreshing isolated prices outside the known range cannot certify holes.
    book.apply(_message(130, action="update", asks=[["103", "17", "1"]], bids=[["98", "15", "2"]]))
    assert len(book.asks.prices) == len(book.bids.prices) == 1
    assert book.available(131, 1) and not book.available(131, 2)
    book.apply(_message(140))
    assert book.available(141, 3) and book.asks.frontier == 103 and book.bids.frontier == 98


def test_atomic_update_order_does_not_change_frontier_or_final_levels():
    # Insertion then deletion temporarily exceeds capacity but final range does not.
    changes = [["100.5", "1", "1"], ["101", "0", "0"]]
    books = [CountedDepthState("BTC-USDT-SWAP", capacity=3) for _ in range(2)]
    for book, rows in zip(books, (changes, list(reversed(changes)))):
        book.apply(_message())
        book.apply(_message(110, action="update", asks=rows, bids=[]))
    assert books[0].asks.levels == books[1].asks.levels
    assert books[0].asks.frontier == books[1].asks.frontier == 103


def test_gap_and_crossing_require_snapshot_recovery_with_new_segments():
    book = CountedDepthState("BTC-USDT-SWAP", capacity=3)
    book.apply(_message())
    assert book.available(1100, 3) and not book.available(1101, 3)
    book.apply(_message(1101, action="update", asks=[], bids=[]))
    assert not book.initialized
    book.apply(_message(1102, action="update", asks=[], bids=[]))
    assert not book.available(1103, 1)
    book.apply(_message(1200))
    assert book.segment == 2 and book.available(1201, 3)
    book.apply(_message(1210, action="update", asks=[], bids=[["101", "1", "1"]]))
    assert not book.initialized and book.stats["crossed_book_resets"] == 1
    book.apply(_message(1220))
    assert book.segment == 3


def test_strict_cutoffs_exclude_all_equal_time_messages_and_coarsen_queries():
    messages = [_message(), _message(200, action="update", asks=[["101", "24", "4"]], bids=[]),
                _message(200, action="update", asks=[["101", "36", "9"]], bids=[])]
    values, _ = _sample(messages, [299, 300, 301, 400], delays_ms=(100,), cadence_ms=100)
    assert values["query_cutoffs"][:, 0].tolist() == [100, 200, 200, 300]
    assert values["publisher_times"][:, 0].tolist() == [-1, 100, 100, 200]
    assert values["order_counts"][:, 0, 0, 0].tolist() == [0, 3, 3, 9]
    assert values["depth"][:, 0, 0, 1].tolist() == [0, 12, 12, 36]


def test_initial_incremental_prefix_stays_unavailable_until_its_first_snapshot():
    messages = [_message(100, action="update"),
        _message(150, action="update", asks=[["101", "36", "9"]], bids=[]), _message(200)]
    values, audit = _sample(messages, [200, 300, 400], delays_ms=(100,), cadence_ms=100)
    assert values["publisher_times"][:, 0].tolist() == [-1, -1, 200]
    assert values["known_depths"][:, 0, 0].tolist() == [0, 0, 3]
    assert values["order_counts"][:, 0, 0, 0].tolist() == [0, 0, 3]
    assert audit["initial_updates_ignored"] == 2 and audit["first_snapshot_time"] == 200
    prefix, _ = _sample(messages[:2], [200, 300], delays_ms=(100,), cadence_ms=100)
    for key in values:
        if key != "delays_ms":
            np.testing.assert_array_equal(values[key][:2], prefix[key])


def test_future_prefix_query_partition_and_missing_masks_are_exact():
    messages = [_message(), _message(150, action="update", asks=[["101", "24", "4"]], bids=[]),
                _message(1400), _message(1410, action="update", asks=[["101", "48", "6"]], bids=[])]
    clock = [300, 600, 1200, 1400, 1600]
    full, _ = _sample(messages, clock)
    prefix, _ = _sample(messages[:2], clock[:3])
    partition, _ = _sample(messages, clock[::2])
    for key in full:
        if key == "delays_ms":
            continue
        np.testing.assert_array_equal(full[key][:3], prefix[key])
        np.testing.assert_array_equal(full[key][::2], partition[key])
    assert (full["known_depths"][3, 0] == 0).all()  # Source is stale at cutoff 1300.
    assert (full["depth"][3, 0] == 0).all() and (full["order_counts"][3, 0] == 0).all()
    assert full["segment_ids"][4, 0] == 2


def test_sampler_rejects_bad_clocks_incomplete_snapshots_and_regressions():
    with pytest.raises(ValueError):
        _sample([_message(200), _message(100)], [300])
    with pytest.raises(ValueError):
        _sample([_message(100, asks=[["101", "1", "1"]])], [300])
    with pytest.raises(ValueError):
        _sample([_message()], [300, 300])
    with pytest.raises(ValueError):
        sample_counted_depth([_message()], np.array([300.0]), "BTC-USDT-SWAP")
    with pytest.raises(ValueError):
        _sample([_message()], [300], delays_ms=(100, 100))


def test_archive_reader_checks_member_clock_and_final_crc(tmp_path):
    path = tmp_path / "BTC-USDT-SWAP-L2orderbook-400lv-2023-06-01.tar.gz"
    body = json.dumps({"instId": "BTC-USDT-SWAP", "action": "snapshot", "ts": "100",
        "asks": [["101", "1", "1"]], "bids": [["100", "1", "1"]]}).encode() + b"\n"
    with tarfile.open(path, "w:gz") as package:
        member = tarfile.TarInfo(path.name.removesuffix(".tar.gz") + ".data")
        member.size = len(body)
        package.addfile(member, io.BytesIO(body))
    audit = {}
    messages = list(counted_archive_messages(path, instrument="BTC-USDT-SWAP", start_ms=0, end_ms=200, audit=audit))
    assert len(messages) == 1 and audit["complete_gzip_crc_and_size_trailer_verified"]
    with pytest.raises(ValueError):
        list(counted_archive_messages(path, instrument="BTC-USDT-SWAP", start_ms=101, end_ms=200, audit={}))
    corrupted = bytearray(path.read_bytes())
    corrupted[-8] ^= 1
    path.write_bytes(corrupted)
    with pytest.raises((gzip.BadGzipFile, OSError, tarfile.ReadError)):
        list(counted_archive_messages(path, instrument="BTC-USDT-SWAP", start_ms=0, end_ms=200, audit={}))
