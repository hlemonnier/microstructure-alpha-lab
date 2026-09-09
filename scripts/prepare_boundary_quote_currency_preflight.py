"""Verify alternative-quote features on one historical training day, without fits."""

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
from lob_forge.boundary_quote_currency_features import converted_spot_feature_frame
from lob_forge.boundary_spot_data import read_spot_trades
from lob_forge.boundary_spot_features import SPOT_FEATURES, spot_feature_frame
from benchmark_boundary_tabicl_feature_budget import memory_guard

ROOT = Path(__file__).resolve().parents[1]


def write(path, record):
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def prepare(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if output.exists():
        raise ValueError("Preserve any previous quote-feature preflight")
    for path, digest in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError("Frozen quote-feature preflight input changed")
    output.mkdir(parents=True)
    write(output / "frozen_preflight.json", {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"]})
    started = time.monotonic()
    readings = {"maximum_sampled_rss_bytes": 0, "maximum_sampled_gpu_driver_bytes": 0, "maximum_sampled_simultaneous_sum_bytes": 0}
    stop = threading.Event()
    guard = threading.Thread(target=memory_guard, args=("cpu", output, protocol["limits"], stop, readings), daemon=True)
    guard.start()
    sources, audits, records = {}, {}, []
    try:
        for symbol, record in protocol["trade_sources"].items():
            path = ROOT / record["path"]
            if path.stat().st_size != record["bytes"] or sha256_file(path) != record["official_sha256"]:
                raise ValueError("Each archive must retain its official checksum and byte length")
            frame = read_spot_trades(path, day_start_ms=protocol["day_start_ms"])
            times, ids = frame.transact_time.to_numpy(), frame.agg_trade_id.to_numpy()
            gaps = int((np.diff(ids) != 1).sum())
            if gaps or times[0] - protocol["day_start_ms"] > 300000 or protocol["day_start_ms"] + 86400000 - times[-1] > 300000:
                raise ValueError("Source ID gaps or incomplete day boundaries require a separately specified treatment")
            sources[symbol] = frame
            audits[symbol] = {"rows": len(frame), "aggregate_id_gaps": gaps,
                "first_publisher_time": int(times[0]), "last_publisher_time": int(times[-1]),
                "maximum_intertrade_gap_ms": int(np.diff(times).max(initial=0))}
        for symbol, native_symbol, conversion_symbol in protocol["pairs"]:
            canonical = pd.read_parquet(ROOT / protocol["canonical_sources"][symbol],
                columns=["decision_time", "bid", "ask", "bid_qty", "ask_qty"])
            trades = sources[native_symbol]
            conversion = sources[conversion_symbol][["transact_time", "price"]].rename(columns={"transact_time": "publisher_time"})
            for delay in protocol["delays_ms"]:
                frame = converted_spot_feature_frame(canonical, trades, conversion, information_delay_ms=delay,
                    conversion_delay_ms=delay, maximum_conversion_age_ms=protocol["maximum_conversion_age_ms"])
                plain = spot_feature_frame(canonical, trades, information_delay_ms=delay)
                native = [c for c in plain if "basis" not in c]
                pd.testing.assert_frame_equal(frame[native], plain[native], check_exact=True)
                constant_conversion = pd.DataFrame({"publisher_time": canonical.decision_time.to_numpy() - 1, "price": 1.})
                identity = converted_spot_feature_frame(canonical, trades, constant_conversion, information_delay_ms=delay,
                    conversion_delay_ms=0, maximum_conversion_age_ms=protocol["maximum_conversion_age_ms"])
                pd.testing.assert_frame_equal(identity[["decision_time", *SPOT_FEATURES]], plain, check_exact=True)
                prefix = canonical.iloc[:protocol["prefix_rows"]]
                last_clock = int(prefix.decision_time.iloc[-1])
                shorter = converted_spot_feature_frame(prefix, trades.loc[trades.transact_time + delay < last_clock],
                    conversion.loc[conversion.publisher_time + delay < last_clock], information_delay_ms=delay,
                    conversion_delay_ms=delay, maximum_conversion_age_ms=protocol["maximum_conversion_age_ms"])
                pd.testing.assert_frame_equal(frame.iloc[:len(prefix)], shorter, check_exact=True)
                path = output / f"{symbol}_{delay}.parquet"
                frame.to_parquet(path, index=False, compression="zstd")
                pd.testing.assert_frame_equal(frame, pd.read_parquet(path), check_exact=True)
                row = {"symbol": symbol, "native_symbol": native_symbol, "conversion_symbol": conversion_symbol,
                    "delay_ms": delay, "rows": len(frame), "columns": list(frame.columns),
                    "conversion_available_rows": int(frame.conversion_available.sum()),
                    "converted_basis_available_rows": int(frame.converted_basis_available.sum()),
                    "original_native_fields_exact": True, "identity_conversion_replay_exact": True,
                    "truncated_source_prefix_exact": True, "saved_features_exact": True,
                    "feature_path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
                records.append(row)
                print(f"quote_features_verified={symbol}/{delay} rows={len(frame)}", flush=True)
                del frame, plain, identity, shorter
                gc.collect()
    finally:
        stop.set()
        guard.join()
    if readings["maximum_sampled_rss_bytes"] <= 0:
        raise ValueError("The source preflight requires working resource monitoring")
    for path, digest in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError("Quote-feature preflight input changed during execution")
    write(output / "summary.json", {"protocol_sha256": sha256_file(protocol_path), "complete": True,
        "source_audits": audits, "records": records, "memory": readings, "seconds": time.monotonic() - started,
        "original_read_columns": ["decision_time", "bid", "ask", "bid_qty", "ask_qty"],
        "model_fits": 0, "target_labels_decoded": False, "predictive_metrics_computed": False,
        "independent_confirmation_data_opened": False,
        "artifact_hashes": {str(p.relative_to(output)): sha256_file(p) for p in output.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.protocol.resolve(), args.output.resolve())
