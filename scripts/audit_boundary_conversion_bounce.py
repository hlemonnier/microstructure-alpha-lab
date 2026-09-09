"""Test fixed half-spacing trade-side correction on training-only FX observations."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_spot_data import read_spot_trades

ROOT = Path(__file__).resolve().parents[1]


def audit(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if output.exists():
        raise ValueError("Preserve each original trade-side conversion diagnostic")
    for path, digest in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError("A frozen conversion-bounce prerequisite changed")
    output.mkdir(parents=True)
    records = []
    for source in protocol["sources"]:
        start = int(datetime.fromisoformat(source["date"]).replace(tzinfo=timezone.utc).timestamp() * 1000)
        trades = read_spot_trades(ROOT / source["conversion_archive"], day_start_ms=start)
        prices, times = trades.price.to_numpy(), trades.transact_time.to_numpy()
        signs = np.where(trades.is_buyer_maker.to_numpy(), -1., 1.)
        half = protocol["fixed_half_spacing_usdt_per_tusd"]
        corrected = prices - half * signs
        if not np.isfinite(corrected).all() or (corrected <= 0).any():
            raise ValueError("Fixed-side adjusted observations must remain finite and positive")
        opposite = (signs[1:] != signs[:-1]) & (np.diff(times) <= protocol["adjacent_opposite_maximum_gap_ms"])
        apparent_spread = np.diff(prices)[opposite] * signs[1:][opposite]
        if not len(apparent_spread):
            raise ValueError("Opposite-side diagnostic requires eligible adjacent trade pairs")
        match = np.abs(apparent_spread - 2 * half) <= protocol["descriptive_float_match_tolerance"]
        clock = pd.read_parquet(ROOT / source["canonical"], columns=["decision_time"]).decision_time.to_numpy(dtype=np.int64)
        last = np.searchsorted(times + protocol["delay_ms"], clock, side="left") - 1
        safe = np.maximum(last, 0)
        available = (last >= 0) & (clock - times[safe] <= protocol["maximum_age_ms"])
        sampled, adjusted = prices[safe], corrected[safe]
        pair = available[1:] & available[:-1]
        raw_changes, corrected_changes = np.diff(sampled)[pair], np.diff(adjusted)[pair]
        raw_variance, corrected_variance = np.var(raw_changes), np.var(corrected_changes)
        if raw_variance <= 0:
            raise ValueError("Variation is required to describe the fixed adjustment")
        records.append({"date": source["date"], "source_trades": len(trades),
            "adjacent_opposite_side_pairs_within_one_second": len(apparent_spread),
            "one_observed_spacing_separation_fraction": float(np.mean(match)),
            "zero_price_separation_fraction": float(np.mean(np.abs(apparent_spread) <= protocol["descriptive_float_match_tolerance"])),
            "median_buyer_minus_seller_price_usdt_per_tusd": float(np.median(apparent_spread)),
            "sampled_raw_increment_variance": float(raw_variance), "sampled_corrected_increment_variance": float(corrected_variance),
            "corrected_to_raw_increment_variance_ratio": float(corrected_variance / raw_variance),
            "raw_rate_range": [float(sampled[available].min()), float(sampled[available].max())],
            "corrected_rate_range": [float(adjusted[available].min()), float(adjusted[available].max())],
            "available_decision_rows": int(available.sum()), "available_adjacent_query_pairs": int(pair.sum())})
    for path, digest in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError("A conversion-bounce input changed during execution")
    result = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"]
        , "complete": True, "records": records, "model_fits": 0, "target_labels_decoded": False,
        "predictive_metrics_computed": False, "independent_confirmation_data_opened": False,
        "interpretation": "A fixed trade-side adjustment is a hypothesized midpoint proxy. Lower observed increment variance does not prove lower latent-rate error or better predictive accuracy. No true FX midpoint or executable FX spread was observed."}
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"summary_sha256": sha256_file(output / "summary.json"), "records": records}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.protocol.resolve(), args.output.resolve())
