import pytest

from lob_forge.boundary_tardis_certified import CertifiedBinanceFuturesDepthState


def snapshot(uid=10):
    return {"stream": "btcusdt@depthSnapshot", "generated": True, "data": {"lastUpdateId": uid, "E": 1000,
        "bids": [["100", "10"], ["99", "20"]], "asks": [["102", "10"], ["103", "20"], ["104", "30"]]}}


def delta(first, final, previous, bids=(), asks=()):
    return {"stream": "btcusdt@depth@0ms", "data": {"e": "depthUpdate", "s": "BTCUSDT", "U": first,
        "u": final, "pu": previous, "E": 1100 + final, "T": 1090 + final, "b": list(bids), "a": list(asks)}}


def quote(uid, bid=95, bid_q=10, ask=102, ask_q=10):
    return {"s": "BTCUSDT", "u": uid, "E": 1100 + uid, "T": 1090 + uid,
            "b": str(bid), "B": str(bid_q), "a": str(ask), "A": str(ask_q)}


def initialized():
    book = CertifiedBinanceFuturesDepthState("BTCUSDT", min_tick=1)
    book.apply(1500_000_000, snapshot())
    book.apply(1600_000_000, delta(9, 20, 8))
    return book


def test_known_zero_updates_and_quote_certificates_extend_only_proved_ticks():
    book = initialized()
    book.apply(1700_000_000, delta(21, 30, 20, [["100", "0"], ["99", "0"], ["98", "0"],
        ["95", "10"], ["94", "20"], ["93", "30"]]))
    assert book.bid_frontier == 98  # Zero is an observation; 97 and 96 remain unknown.
    assert book.snapshot(3) is None
    book.observe_quote(1800_000_000, quote(30))
    assert book.bid_frontier == 93
    assert book.snapshot(3) == [(102.0, 10.0, 95.0, 10.0), (103.0, 20.0, 94.0, 20.0), (104.0, 30.0, 93.0, 30.0)]
    assert book.available_time_ns == 1700_000_000
    assert book.knowledge_capture_ns == 1800_000_000


def test_quote_waits_for_depth_and_cannot_overwrite_a_later_batch_quantity():
    book = initialized()
    book.observe_quote(1700_000_000, quote(25))
    assert book.bid_frontier == 99 and book.knowledge_capture_ns is None
    book.apply(1800_000_000, delta(21, 30, 20, [["100", "0"], ["99", "0"],
        ["95", "7"], ["94", "20"], ["93", "30"]]))
    assert book.snapshot(3)[0][3] == 7
    assert book.bid_frontier == 93 and book._certificate_u == 25
    before = book.bids.copy(), book.asks.copy(), book.bid_frontier
    book.observe_quote(1900_000_000, quote(24, bid=90))
    assert (book.bids, book.asks, book.bid_frontier) == before
    assert book.certification_counts["ignored_older_quotes"] == 1


def test_quote_can_reveal_an_unchanged_previously_unknown_best_quantity():
    book = initialized()
    book.apply(1700_000_000, delta(21, 30, 20, [["100", "0"], ["99", "0"], ["94", "20"], ["93", "30"]]))
    book.observe_quote(1800_000_000, quote(30))
    assert book.bids[95] == 10 and book.bid_frontier == 93
    assert book.snapshot(3)[0][2:] == (95.0, 10.0)


def test_disagreeing_certificate_fails_before_mutating_either_side_or_frontier():
    book = initialized()
    book.apply(1700_000_000, delta(21, 30, 20, [["100", "0"], ["99", "0"], ["94", "20"], ["93", "30"]]))
    before = book.bids.copy(), book.asks.copy(), book.bid_frontier, book.ask_frontier
    with pytest.raises(ValueError, match="contradicts a known price quantity"):
        book.observe_quote(1800_000_000, quote(30, ask_q=999))
    assert (book.bids, book.asks, book.bid_frontier, book.ask_frontier) == before
    assert not book.ready and book.knowledge_capture_ns is None


def test_certification_epoch_resets_and_off_grid_updates_do_not_mutate_books():
    book = initialized()
    before = book.bids.copy(), book.asks.copy(), book.last_u
    with pytest.raises(ValueError, match="tick grid"):
        book.apply(1700_000_000, delta(21, 30, 20, [["98.5", "10"]]))
    assert (book.bids, book.asks, book.last_u) == before
    book.observe_quote(1800_000_000, quote(25))
    book.disconnect()
    assert not book._pending and not book._level_u[0] and book.knowledge_capture_ns is None
    book.observe_quote(1900_000_000, quote(40, bid=80))
    book.apply(2000_000_000, snapshot(50))
    book.apply(2100_000_000, delta(49, 55, 48))
    assert book.bid_frontier == 99 and book._certificate_u == -1
    assert book.knowledge_capture_ns is None and not book._pending


def test_quote_certificate_rejects_contradictory_known_better_prices():
    book = initialized()
    before = book.bids.copy(), book.asks.copy(), book.bid_frontier
    with pytest.raises(ValueError, match="known better price"):
        book.observe_quote(1700_000_000, quote(20))
    assert (book.bids, book.asks, book.bid_frontier) == before
    assert not book.ready
