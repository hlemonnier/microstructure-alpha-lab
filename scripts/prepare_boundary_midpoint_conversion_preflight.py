"""Verify raw-plus-side and midpoint-proxy conversions without predictive fits."""

from __future__ import annotations

import argparse
import gc
import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_conversion_midpoint import side_aware_converted_spot_frame
from lob_forge.boundary_spot_data import read_spot_trades
from benchmark_boundary_tabicl_feature_budget import memory_guard

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def prepare(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if output.exists():
        raise ValueError("Preserve every prior side-aware conversion feature attempt")
    for path, digest in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError("A fixed side-aware conversion prerequisite changed")
    output.mkdir(parents=True)
    write(output / "frozen_preparation.json", {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"]})
    start = time.monotonic()
    readings = {"maximum_sampled_rss_bytes": 0, "maximum_sampled_gpu_driver_bytes": 0, "maximum_sampled_simultaneous_sum_bytes": 0}
    stop = threading.Event()
    guard = threading.Thread(target=memory_guard, args=("cpu", output, protocol["limits"], stop, readings), daemon=True)
    guard.start()
    records, audits, sources = [], {}, {}
    try:
        for symbol in ("BTCTUSD", "TUSDUSDT"):
            record = protocol["trade_sources"][symbol]
            path = ROOT / record["path"]
            if sha256_file(path) != record["official_sha256"] or path.stat().st_size != record["bytes"]:
                raise ValueError("Each original trade archive requires its official SHA and exact length")
            frame = read_spot_trades(path, day_start_ms=protocol["day_start_ms"])
            clocks, ids = frame.transact_time.to_numpy(), frame.agg_trade_id.to_numpy()
            if ((np.diff(ids) != 1).any() or clocks[0]-protocol["day_start_ms"] > 300000
                or protocol["day_start_ms"]+86400000-clocks[-1] > 300000):
                raise ValueError("Original source ID and day-boundary completeness must remain intact")
            audits[symbol] = {"rows": len(frame), "aggregate_id_gaps": 0,
                "first_publisher_time": int(clocks[0]), "last_publisher_time": int(clocks[-1])}
            sources[symbol] = frame
        canonical = pd.read_parquet(ROOT / protocol["canonical"], columns=["decision_time", "bid", "ask", "bid_qty", "ask_qty"])
        native, conversion = sources["BTCTUSD"], sources["TUSDUSDT"]
        for delay in protocol["delays_ms"]:
            original = pd.read_parquet(ROOT / protocol["original_feature_groups"][str(delay)]["features_path"])
            for variant, half in protocol["variants"].items():
                frame = side_aware_converted_spot_frame(canonical, native, conversion, half_spacing=half,
                    information_delay_ms=delay, conversion_delay_ms=delay,
                    maximum_conversion_age_ms=protocol["maximum_conversion_age_ms"])
                unchanged = [c for c in original if "basis" not in c and c != "conversion_log_target_quote_per_native_quote"]
                pd.testing.assert_frame_equal(frame[unchanged], original[unchanged], check_exact=True)
                if half == 0:
                    pd.testing.assert_frame_equal(frame[original.columns], original, check_exact=True)
                prefix = canonical.iloc[:protocol["prefix_rows"]]
                last = int(prefix.decision_time.iloc[-1])
                shorter = side_aware_converted_spot_frame(prefix, native.loc[native.transact_time + delay < last],
                    conversion.loc[conversion.transact_time + delay < last], half_spacing=half,
                    information_delay_ms=delay, conversion_delay_ms=delay,
                    maximum_conversion_age_ms=protocol["maximum_conversion_age_ms"])
                pd.testing.assert_frame_equal(frame.iloc[:len(prefix)], shorter, check_exact=True)
                destination = output / f"BTCUSDT_{variant}_{delay}.parquet"
                frame.to_parquet(destination, index=False, compression="zstd")
                pd.testing.assert_frame_equal(frame, pd.read_parquet(destination), check_exact=True)
                records.append({"variant": variant, "delay_ms": delay, "half_spacing_usdt_per_tusd": half,
                    "rows": len(frame), "columns": list(frame.columns), "native_fields_and_availability_exact": True,
                    "raw_plus_side_replays_original_exactly": True if half == 0 else None,
                    "truncated_source_prefix_exact": True, "saved_features_exact": True,
                    "conversion_available_rows": int(frame.conversion_available.sum()),
                    "features_path": str(destination.relative_to(ROOT)), "sha256": sha256_file(destination)})
                del frame, shorter
                gc.collect()
            raw = pd.read_parquet(output / f"BTCUSDT_raw_side_{delay}.parquet")
            midpoint = pd.read_parquet(output / f"BTCUSDT_midpoint_side_{delay}.parquet")
            matched = [c for c in raw if "basis" not in c and c != "conversion_log_target_quote_per_native_quote"]
            pd.testing.assert_frame_equal(raw[matched], midpoint[matched], check_exact=True)
            del raw, midpoint, original
            gc.collect()
    finally:
        stop.set()
        guard.join()
    if readings["maximum_sampled_rss_bytes"] <= 0:
        raise ValueError("Side-aware conversion preparation requires a working memory monitor")
    for path, digest in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError("Side-aware conversion input changed during preparation")
    write(output / "summary.json", {"protocol_sha256": sha256_file(protocol_path), "complete": True, "records": records,
        "source_audits": audits, "memory": readings, "seconds": time.monotonic()-start,
        "both_variants_share_exact_conversion_sides": True, "original_read_columns": ["decision_time", "bid", "ask", "bid_qty", "ask_qty"],
        "model_fits": 0, "target_labels_decoded": False, "predictive_metrics_computed": False,
        "independent_confirmation_data_opened": False,
        "artifact_hashes": {str(p.relative_to(output)): sha256_file(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({"summary_sha256": sha256_file(output / "summary.json"), "groups": len(records),
        "maximum_sampled_rss_bytes": readings["maximum_sampled_rss_bytes"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.protocol.resolve(), args.output.resolve())
