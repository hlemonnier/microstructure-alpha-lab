"""Validate the registered public one-minute Binance raw-data sample."""

from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path
from statistics import median

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_tardis_depth import BinanceFuturesDepthState, iter_tardis_messages
from lob_forge.features import BOOK_TICKER_COLUMNS, iter_zip_dict_rows

ROOT = Path(__file__).resolve().parents[1]


def assess():
    folder = ROOT / "data/research/boundary_tardis_raw_preflight_20260907"
    source = folder / "minute0.ndjson.gz"
    protocol = ROOT / "docs/research/boundary_tardis_raw_preflight_20260907.json"
    transport = ROOT / "docs/research/boundary_tardis_raw_preflight_transport_20260907.json"
    amendment = json.loads(transport.read_text())
    if sha256_file(protocol) != amendment["parent_protocol_sha256"] or sha256_file(source) != amendment["response_sha256"]:
        raise ValueError("Registered response and transport-only amendment must match")
    symbols = ("BTCUSDT", "ETHUSDT")
    books = {s: BinanceFuturesDepthState(s) for s in symbols}
    quotes, depths, lag, counts = ({s: {} for s in symbols} for _ in range(4))
    lines = disconnects = 0
    first = last = None
    for capture, message in iter_tardis_messages(source):
        lines += 1
        if message is None:
            disconnects += 1
            for book in books.values():
                book.disconnect()
            continue
        first = capture if first is None else first
        last = capture
        stream, data = message["stream"], message["data"]
        symbol = stream.split("@")[0].upper()
        kind = stream.split("@")[1]
        counts[symbol][kind] = counts[symbol].get(kind, 0) + 1
        lag[symbol].setdefault(kind, []).append((capture - data["E"] * 1_000_000) / 1_000_000)
        if kind == "bookTicker":
            value = tuple(float(data[k]) for k in ("a", "A", "b", "B"))
            if data["u"] in quotes[symbol] and quotes[symbol][data["u"]][0] != value:
                raise ValueError("Conflicting quote payload for an identical native update ID")
            quotes[symbol][data["u"]] = (value, data["E"], data["T"])
        else:
            book = books[symbol]
            book.apply(capture, message)
            snap = book.snapshot(25)
            if snap is not None:
                depths[symbol][book.last_u] = tuple(snap[0])
    originals = json.loads((ROOT / "data/research/boundary_exact_labels_20260907/dataset_manifest.json").read_text())
    records = []
    for symbol in symbols:
        overlap = sorted(set(depths[symbol]) & set(quotes[symbol]))
        mismatch = [u for u in overlap if depths[symbol][u] != quotes[symbol][u][0]]
        if not overlap or mismatch:
            raise ValueError(f"Reconstructed BBO/native quote mismatch: {symbol}, compared={len(overlap)}, mismatches={mismatch[:5]}")
        quote_ids = sorted(quotes[symbol])
        asof_matches = asof_mismatches = 0
        for uid, value in depths[symbol].items():
            index = bisect_right(quote_ids, uid) - 1
            if index >= 0:
                asof_matches += 1
                asof_mismatches += int(value != quotes[symbol][quote_ids[index]][0])
        if asof_mismatches:
            raise ValueError(f"Reconstructed BBO disagrees with most recent quote by exchange update ID: {symbol}, {asof_mismatches}/{asof_matches}")
        parent = next(r for r in originals["sessions"] if r["symbol"] == symbol and r["session_date"] == "2023-06-01")["sources"]["bookTicker"]
        archive = Path(parent["local_path"])
        if sha256_file(archive) != parent["provider_checksum_sha256"]:
            raise ValueError("Official original quote archive must match its provider checksum")
        official = {}
        for row in iter_zip_dict_rows(archive, BOOK_TICKER_COLUMNS):
            if int(row["event_time"]) > last // 1_000_000 + 5000:
                break
            uid = int(row["update_id"])
            if uid in quotes[symbol]:
                official[uid] = (tuple(float(row[k]) for k in ("best_ask_price", "best_ask_qty", "best_bid_price", "best_bid_qty")),
                                 int(row["event_time"]), int(row["transaction_time"]))
        shared = sorted(set(official) & set(quotes[symbol]))
        # The two recordings have different publisher E timestamps at some
        # identical updates. Preserve and report that observed discrepancy;
        # prices, quantities and transaction T must still match exactly.
        bad = [u for u in shared if (official[u][0], official[u][2]) != (quotes[symbol][u][0], quotes[symbol][u][2])]
        publisher_deltas = [quotes[symbol][u][1] - official[u][1] for u in shared]
        if not shared or bad:
            raise ValueError(f"Provider/original quote mismatch: {symbol}, compared={len(shared)}, mismatches={bad[:5]}")
        book = books[symbol]
        records.append({"symbol": symbol, "native_messages": counts[symbol], "reconstructible_top25_states": len(depths[symbol]),
            "shared_depth_quote_update_ids": len(overlap), "depth_quote_mismatches": len(mismatch),
            "depth_states_compared_to_latest_quote_by_update_id": asof_matches, "latest_quote_mismatches": asof_mismatches,
            "native_quotes": len(quotes[symbol]), "shared_official_quote_update_ids": len(shared), "official_quote_mismatches": len(bad),
            "official_quote_comparison_fields": ["ask", "ask_qty", "bid", "bid_qty", "transaction_time"],
            "provider_minus_official_publisher_ms": {"min": min(publisher_deltas), "median": median(publisher_deltas),
                "max": max(publisher_deltas), "different_count": sum(x != 0 for x in publisher_deltas)},
            "native_quote_ids_absent_from_official_day": len(set(quotes[symbol]) - set(official)),
            "official_archive_path": str(archive.relative_to(ROOT)), "official_archive_sha256": parent["provider_checksum_sha256"],
            "futures_snapshot_bridge_and_pu_continuity_passed": True, "snapshot_capture_not_backdated": True,
            "initial_snapshot_frontier_enforced": True, "snapshots": book.snapshots, "applied_deltas": book.applied_deltas,
            "buffer_evictions": book.buffer_evictions,
            "observed_capture_minus_publisher_ms": {k: {"min": min(v), "median": median(v), "max": max(v), "negative_count": sum(x < 0 for x in v)} for k, v in lag[symbol].items()}})
    evidence = {"protocol_sha256": sha256_file(protocol), "transport_amendment_sha256": sha256_file(transport),
        "source_path": str(source.relative_to(ROOT)), "source_sha256": sha256_file(source), "source_bytes": source.stat().st_size,
        "format": "plain_ndjson_despite_gz_suffix", "http_status": 200, "headers_sha256": sha256_file(folder / "minute0.headers"),
        "first_capture_ns": first, "last_capture_ns": last, "lines": lines, "disconnects": disconnects,
        "model_fits": 0, "assessment_labels_read": False, "substantial_gain_confirmed": False,
        "scope": "One-minute reconstructibility and official quote parity only. No whole-day availability or predictive performance claim.",
        "publisher_clock_discrepancy": "docs/research/boundary_tardis_raw_clock_findings_20260907.json",
        "code_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in (Path(__file__), ROOT / "src/lob_forge/boundary_tardis_depth.py", ROOT / "tests/test_boundary_tardis_depth.py")},
        "records": records}
    output = ROOT / "docs/research/boundary_tardis_raw_preflight_evidence_20260907.json"
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(output)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    assess()
