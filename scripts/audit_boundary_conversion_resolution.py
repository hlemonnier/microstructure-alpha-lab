"""Measure observed conversion-price resolution on original training dates only."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_spot_data import read_spot_trades

ROOT = Path(__file__).resolve().parents[1]


def describe(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Resolution diagnostics require finite nonempty samples")
    return {name: float(value) for name, value in zip(("minimum", "p05", "median", "p95", "maximum"),
        np.quantile(values, [0, .05, .5, .95, 1]), strict=True)}


def audit(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if output.exists():
        raise ValueError("Preserve previous conversion-resolution diagnostics")
    for path, digest in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError("Frozen conversion diagnostic input changed")
    if [r["date"] for r in protocol["sources"]] != ["2023-05-29", "2023-05-30", "2023-05-31", "2023-06-01"]:
        raise ValueError("Only the original first-fold four training dates are allowed")
    output.mkdir(parents=True)
    rows = []
    for source in protocol["sources"]:
        start = int(datetime.fromisoformat(source["date"]).replace(tzinfo=timezone.utc).timestamp() * 1000)
        trades = read_spot_trades(ROOT / source["conversion_archive"], day_start_ms=start)
        prices = trades.price.to_numpy(dtype=float)
        unique = sorted({Decimal(str(p)) for p in prices})
        if len(unique) < 2:
            raise ValueError("Observed spacing cannot be identified from one traded price")
        spacing = min(b - a for a, b in zip(unique[:-1], unique[1:], strict=True))
        canonical = pd.read_parquet(ROOT / source["canonical"], columns=["decision_time", "bid", "ask"])
        times = canonical.decision_time.to_numpy(dtype=np.int64)
        published = trades.transact_time.to_numpy(dtype=np.int64)
        last = np.searchsorted(published + protocol["delay_ms"], times, side="left") - 1
        present, safe = last >= 0, np.maximum(last, 0)
        available = present & (times - published[safe] <= protocol["maximum_age_ms"])
        chosen = prices[safe]
        mid = (canonical.bid.to_numpy() + canonical.ask.to_numpy()) / 2
        spread = canonical.ask.to_numpy() - canonical.bid.to_numpy()
        # This is a decision-time scale diagnostic. The actual target uses the
        # future entry quote, which is deliberately not read by this audit.
        known_barrier = np.maximum(spread / 2, protocol["btc_minimum_tick_usdt"])
        spacing_bps = float(spacing) / chosen * 10000
        known_barrier_bps = known_barrier / mid * 10000
        both = available[1:] & available[:-1]
        changes = np.diff(chosen)
        eligible_changes = changes[both]
        triplets = available[2:] & available[1:-1] & available[:-2]
        left, right = changes[:-1][triplets], changes[1:][triplets]
        lag_one = float(np.corrcoef(left, right)[0, 1]) if np.std(left) > 0 and np.std(right) > 0 else None
        row = {"date": source["date"], "conversion_trade_rows": len(trades), "distinct_traded_prices": [str(p) for p in unique],
            "minimum_observed_spacing_usdt_per_tusd": str(spacing), "decision_rows": len(canonical),
            "conversion_available_rows": int(available.sum()), "observed_spacing_bps": describe(spacing_bps[available]),
            "known_current_mid_barrier_bps": describe(known_barrier_bps[available]),
            "spacing_to_known_current_mid_barrier_ratio": describe((spacing_bps / known_barrier_bps)[available]),
            "adjacent_available_query_pairs": int(both.sum()), "sampled_rate_changes": int(np.count_nonzero(eligible_changes)),
            "sampled_nonzero_rate_change_fraction": float(np.mean(eligible_changes != 0)),
            "sampled_rate_increment_lag_one_correlation": lag_one}
        rows.append(row)
    for path, digest in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError("Conversion diagnostic input changed during execution")
    summary = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"], "complete": True,
        "scope": "Four original training dates only; no target, future entry quote, assessment or predictive score reads.",
        "records": rows, "model_fits": 0, "target_labels_decoded": False, "predictive_metrics_computed": False,
        "independent_confirmation_data_opened": False,
        "interpretation": "Observed minimum price spacing is not an exchange tick-rule certification, estimation-error variance or predictive-gain estimate. The denominator is a current-quote scale analogue, not the actual future-entry target threshold."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"summary_sha256": sha256_file(output / "summary.json"), "records": rows}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.protocol.resolve(), args.output.resolve())
