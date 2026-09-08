"""Sample replayed native books under strict recorded capture-time delays."""

from __future__ import annotations

from array import array

import numpy as np

from lob_forge.boundary_tardis_depth import BinanceFuturesDepthState, iter_tardis_messages


def sample_tardis_depth(paths, decisions, *, delays_ms=(100, 500), progress=None,
                        state_factory=BinanceFuturesDepthState, message_reader=iter_tardis_messages):
    delays = np.asarray(delays_ms)
    if (delays.ndim != 1 or not len(delays) or not np.issubdtype(delays.dtype, np.integer)
        or (delays < 0).any() or len(np.unique(delays)) != len(delays)):
        raise ValueError("Distinct nonnegative capture delays required")
    books = {s: state_factory(s) for s in decisions}
    samples, queries, quote_arrays = {}, [], {}
    counts = {s: {"depth": 0, "depthSnapshot": 0, "bookTicker": 0} for s in decisions}
    for symbol, clock in decisions.items():
        clock = np.asarray(clock)
        if clock.ndim != 1 or not len(clock) or not np.issubdtype(clock.dtype, np.integer) or (np.diff(clock) <= 0).any():
            raise ValueError("Increasing integer decision clocks required")
        shape = (len(clock), len(delays))
        samples[symbol] = {"decision_times": clock.copy(), "delays_ms": delays.copy(), "depth": np.zeros((*shape, 25, 4)),
            "available": np.zeros(shape, dtype=bool), "frontier_covered": np.zeros(shape, dtype=bool),
            "capture_times_ns": np.full(shape, -1, dtype=np.int64), "publisher_times_ms": np.full(shape, -1, dtype=np.int64),
            "update_ids": np.full(shape, -1, dtype=np.int64)}
        queries.extend((int(t - delay) * 1_000_000, symbol, i, j) for i, t in enumerate(clock) for j, delay in enumerate(delays))
        quote_arrays[symbol] = {"ids": array("q"), "values": array("d"), "capture_times_ns": array("q"),
                                "publisher_times_ms": array("q"), "transaction_times_ms": array("q")}
    queries.sort()
    cursor = 0
    previous_capture = first_capture = None
    disconnects = lines = 0
    pending_disconnect = False

    def sample(query):
        cutoff, symbol, i, j = query
        book, target = books[symbol], samples[symbol]
        knowledge_capture = getattr(book, "knowledge_capture_ns", None)
        knowledge_publisher = getattr(book, "knowledge_publisher_ms", None)
        if book.available_time_ns is not None:
            target["capture_times_ns"][i, j] = max(book.available_time_ns, knowledge_capture or 0)
        if book.publisher_time_ms is not None:
            target["publisher_times_ms"][i, j] = max(book.publisher_time_ms, knowledge_publisher or 0)
        if book.last_u is not None:
            target["update_ids"][i, j] = book.last_u
        snapshot = book.snapshot(25)
        target["frontier_covered"][i, j] = snapshot is not None
        if snapshot is None:
            return
        if (not 0 < cutoff - book.available_time_ns <= 1_000_000_000
            or not 0 < cutoff - book.publisher_time_ms * 1_000_000 <= 1_000_000_000
            or (knowledge_capture is not None and knowledge_capture >= cutoff)
            or (knowledge_publisher is not None and knowledge_publisher * 1_000_000 >= cutoff)):
            return
        target["depth"][i, j] = snapshot
        target["available"][i, j] = True

    for number, path in enumerate(paths, 1):
        for capture, message in message_reader(path):
            lines += 1
            if message is None:
                disconnects += 1
                pending_disconnect = True
                continue
            if previous_capture is not None and capture < previous_capture:
                raise ValueError("Raw slices must preserve global capture order")
            previous_capture = capture
            first_capture = capture if first_capture is None else first_capture
            # Equality deliberately samples before applying this native message.
            while cursor < len(queries) and queries[cursor][0] <= capture:
                sample(queries[cursor])
                cursor += 1
            # Blank disconnect markers have no timestamp. Apply their
            # invalidation only at the next observed capture time, so the
            # availability mask cannot reveal a future disconnect early.
            if pending_disconnect:
                for book in books.values():
                    book.disconnect()
                pending_disconnect = False
            stream, data = message["stream"], message["data"]
            symbol, kind = stream.split("@")[0].upper(), stream.split("@")[1]
            if symbol not in books or kind not in counts[symbol]:
                raise ValueError("Unregistered native source channel or asset")
            counts[symbol][kind] += 1
            if kind == "bookTicker":
                quotes = quote_arrays[symbol]
                value = np.array([float(data[k]) for k in ("a", "A", "b", "B")])
                if (data.get("s") != symbol or not np.isfinite(value).all() or value[2] <= 0 or value[0] <= value[2]
                    or min(value[1], value[3]) < 0):
                    raise ValueError("Invalid native quote")
                quotes["ids"].append(data["u"])
                quotes["values"].extend(value)
                quotes["capture_times_ns"].append(capture)
                quotes["publisher_times_ms"].append(data["E"])
                quotes["transaction_times_ms"].append(data["T"])
                if hasattr(books[symbol], "observe_quote"):
                    books[symbol].observe_quote(capture, data)
            else:
                books[symbol].apply(capture, message)
        if progress is not None:
            progress(number, len(paths), lines, cursor, len(queries))
    while cursor < len(queries):
        sample(queries[cursor])
        cursor += 1
    quotes, checks = {}, {}
    for symbol, buffers in quote_arrays.items():
        native = {k: np.frombuffer(v, dtype=np.float64 if k == "values" else np.int64) for k, v in buffers.items()}
        native["values"] = native["values"].reshape(-1, 4)
        if not len(native["ids"]):
            raise ValueError("A nonempty native quote stream is required for replay comparison")
        order = np.argsort(native["ids"], kind="stable")
        ids, value = native["ids"][order], native["values"][order]
        repeated = np.flatnonzero(np.diff(ids) == 0)
        if len(repeated) and not np.array_equal(value[repeated], value[repeated + 1]):
            raise ValueError("Conflicting native quotes at the same update ID")
        target = samples[symbol]
        loc = np.searchsorted(ids, target["update_ids"], side="right") - 1
        valid = target["available"] & (loc >= 0)
        compared = int(valid.sum())
        if not np.array_equal(target["depth"][valid, 0], value[loc[valid]]):
            raise ValueError("Sampled depth disagrees with the latest quote by native update ID")
        book = books[symbol]
        checks[symbol] = {"messages": counts[symbol], "snapshots": book.snapshots, "replayed_deltas": book.applied_deltas,
            "rolling_buffer_evictions": book.buffer_evictions, "available_rows_by_delay": target["available"].sum(axis=0).tolist(),
            "frontier_covered_rows_by_delay": target["frontier_covered"].sum(axis=0).tolist(),
            "rows": len(target["decision_times"]), "sampled_depth_quote_comparisons": compared, "sampled_depth_quote_mismatches": 0,
            "quote_duplicate_ids": len(repeated), "quote_out_of_order_id_arrivals": int((np.diff(native["ids"]) < 0).sum()),
            "current_epoch_snapshot_frontiers": {"bid": book.bid_frontier, "ask": book.ask_frontier}}
        if hasattr(book, "certification_counts"):
            checks[symbol]["certification"] = book.certification_counts.copy()
            checks[symbol]["historical_tick"] = str(book.tick)
        quotes[symbol] = native
    return samples, quotes, {"lines": lines, "disconnects": disconnects, "first_capture_ns": first_capture,
                             "last_capture_ns": previous_capture, "assets": checks}
