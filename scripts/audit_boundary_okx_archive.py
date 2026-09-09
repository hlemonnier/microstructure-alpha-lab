"""Inventory one frozen public OKX archive without constructing model features."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
import time
from collections import Counter
from pathlib import Path

from lob_forge.boundary_okx_source import parse_counted_depth

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def audit(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if output.exists():
        raise ValueError("Preserve every existing full-source audit attempt")
    for path, checksum in protocol["input_hashes"].items():
        if digest(ROOT / path) != checksum:
            raise ValueError(f"Frozen source-audit input changed: {path}")
    output.mkdir(parents=True)
    started = time.monotonic()
    messages, actions, gaps, hours = 0, Counter(), Counter(), Counter()
    level_entries = zero_entries = positive_count_entries = multi_order_entries = 0
    previous = first = last = None
    member_records, source_bytes, maximum_line = [], 0, 0
    snapshots = []
    with gzip.open(ROOT / protocol["archive"], "rb") as decoded:
        with tarfile.open(fileobj=decoded, mode="r|") as package:
            for member in package:
                if (not member.isfile() or member.name != protocol["expected_member"]
                    or member.size != protocol["expected_member_bytes"] or member_records):
                    raise ValueError("The frozen one-member archive inventory changed")
                member_records.append({"name": member.name, "size": member.size})
                with package.extractfile(member) as source:
                    while line := source.readline(protocol["maximum_line_bytes"] + 1):
                        source_bytes += len(line)
                        maximum_line = max(maximum_line, len(line))
                        if len(line) > protocol["maximum_line_bytes"] or source_bytes > protocol["expected_member_bytes"]:
                            raise ValueError("Source decompression exceeded a frozen byte bound")
                        if not line.strip():
                            raise ValueError("Unexpected blank source record")
                        message = parse_counted_depth(line, instrument=protocol["instrument"])
                        clock = message.timestamp_ms
                        if not protocol["start_ms"] <= clock < protocol["end_ms"]:
                            raise ValueError("Source publisher clock lies outside the declared UTC date")
                        if previous is not None:
                            if clock < previous:
                                raise ValueError("Publisher clocks regressed; do not silently reorder")
                            gaps[clock - previous] += 1
                        elif message.action != "snapshot":
                            raise ValueError("The source must begin with its own snapshot")
                        first = clock if first is None else first
                        previous = last = clock
                        messages += 1
                        actions[message.action] += 1
                        hours[(clock - protocol["start_ms"]) // 3600000] += 1
                        for side in (message.asks, message.bids):
                            level_entries += len(side)
                            zero_entries += sum(level.quantity == 0 for level in side)
                            positive_count_entries += sum(level.order_count > 0 for level in side)
                            multi_order_entries += sum(level.order_count > 1 for level in side)
                        if message.action == "snapshot":
                            snapshots.append({"timestamp_ms": clock, "ask_levels": len(message.asks), "bid_levels": len(message.bids)})
                        if messages % 100000 == 0:
                            elapsed = time.monotonic() - started
                            write(output / "progress.json", {"messages": messages, "member_bytes_read": source_bytes,
                                "last_timestamp_ms": last, "seconds": elapsed, "market_model_fits": 0})
                            print(f"okx_source_records={messages} seconds={elapsed:.1f}", flush=True)
                            if elapsed > protocol["maximum_seconds"]:
                                raise ValueError("Full-source inventory exceeded its declared runtime budget")
        # Tar iteration stops at its end marker; consuming the gzip remainder
        # explicitly verifies the compressed stream's final CRC/length trailer.
        tail_bytes = 0
        while block := decoded.read(1024 * 1024):
            tail_bytes += len(block)
            if tail_bytes > protocol["maximum_tar_tail_bytes"] or any(block):
                raise ValueError("Unexpected trailing data after the tar end marker")
    if source_bytes != protocol["expected_member_bytes"] or not messages:
        raise ValueError("Incomplete source member")
    elapsed = time.monotonic() - started
    if elapsed > protocol["maximum_seconds"]:
        raise ValueError("Full-source inventory exceeded its declared runtime budget")
    result = {"protocol_sha256": digest(protocol_path), "archive_sha256": protocol["input_hashes"][protocol["archive"]],
        "members": member_records, "uncompressed_member_bytes_verified": source_bytes,
        "complete_gzip_crc_and_size_trailer_verified": True, "tar_tail_bytes": tail_bytes,
        "messages": messages, "actions": dict(actions), "snapshot_records": snapshots,
        "first_timestamp_ms": first, "last_timestamp_ms": last, "source_records_by_utc_hour": dict(hours),
        "publisher_clock_gap_histogram_ms": dict(sorted(gaps.items())), "maximum_publisher_gap_ms": max(gaps, default=0),
        "duplicate_publisher_timestamps": gaps.get(0, 0), "maximum_record_bytes": maximum_line,
        "published_level_entries": level_entries, "zero_quantity_count_entries": zero_entries,
        "positive_order_count_entries": positive_count_entries, "multi_order_entries": multi_order_entries,
        "all_original_decimal_prices_quantities_and_counts_valid": True,
        "market_model_fits": 0, "assessment_metrics_computed": False, "independent_confirmation_data_opened": False,
        "seconds": elapsed,
        "limitations": ["A publisher-clock gap is not proof of dropped messages or of a quiet market; sequence IDs and local arrival clocks are absent.",
            "This inventory does not certify reconstructed top-depth coverage or infer individual cancellations/executions.",
            "Absolute contract units, cross-venue availability delays and economic comparability still require a separate feature contract.",
            "Source integrity and order-count variation do not establish any predictive gain."]}
    write(output / "summary.json", result)
    print(f"okx_source_audit_complete {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.protocol.resolve(), args.output.resolve())
