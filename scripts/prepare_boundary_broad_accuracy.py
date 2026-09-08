"""Cache unchanged observation context for the twenty exposed-date anchor audit."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_confirm_model import SYMBOLS
from lob_forge.boundary_event_inputs import utc_ms
from lob_forge.boundary_long_context_inputs import context_columns, load_context_inputs
from lob_forge.boundary_prequential import decoded_release_clock
from lob_forge.boundary_regime_coverage import calendar_stride_mask
from run_boundary_confirmation import noon, write_json

ROOT = Path(__file__).resolve().parents[1]


def prepare(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if protocol["evidence_status"] != "development_only_on_exposed_dates" or protocol["market_model_fits"] != 0:
        raise ValueError("Only the frozen exposed-data preparation is authorized by this protocol")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen broad-context input changed: {path}")
    if output.exists():
        raise ValueError("Preserve any existing broad-context preparation attempt")
    output.mkdir(parents=True)
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"],
        "python": platform.python_version(), "dependencies": {n: version(n) for n in ("numpy", "pandas", "pyarrow")}}
    write_json(output / "frozen_preparation.json", identity)
    sources = [json.loads((ROOT / path).read_text()) for path in protocol["source_manifests"]]
    sessions = [r for source in sources for r in source["sessions"]]
    expected = {(s, d) for d in protocol["feature_dates"] for s in SYMBOLS}
    if len(sessions) != 50 or len({(r["symbol"], r["session_date"]) for r in sessions}) != 50 or {(r["symbol"], r["session_date"]) for r in sessions} != expected:
        raise ValueError("Exactly fifty unique registered feature sessions required")
    combined = {"sessions": sessions, "evidence_status": protocol["evidence_status"], "forecast_only": True,
        "parent_manifest_hashes": {p: sha256_file(ROOT / p) for p in protocol["source_manifests"]}}
    write_json(output / "dataset_manifest.json", combined)
    records, schema = [], None
    started = time.monotonic()
    for day in protocol["feature_dates"]:
        selected = [r for r in sessions if r["session_date"] == day]
        partition = output / "inputs" / f"{day}.json"
        write_json(partition, {"sessions": selected})
        x, y, clocks = load_context_inputs(ROOT, partition)
        for symbol in SYMBOLS:
            key = symbol, day
            columns = context_columns(x[key], "observations")
            if len(columns) != 220 or (schema is not None and columns != schema):
                raise ValueError("The original 220 observation-context feature schema must remain identical")
            schema = columns
            frame, target, clock = x[key][columns].copy(), y[key].copy(), clocks[key].copy()
            if (not np.isfinite(frame.to_numpy()).all() or not np.issubdtype(clock.dtype, np.integer)
                or not np.issubdtype(target.dtype, np.integer) or not np.isin(target, [-1, 0, 1]).all()
                or (np.diff(clock) <= 0).any()):
                raise ValueError("Finite original features and aligned chronological integer outcomes required")
            original = next(r for r in selected if r["symbol"] == symbol)
            raw = pd.read_parquet(ROOT / original["features_path"], columns=["decision_time", "label", "future_event_time"]).set_index("decision_time").loc[clock]
            np.testing.assert_array_equal(raw.label.to_numpy(), target)
            released = decoded_release_clock(raw.future_event_time.to_numpy())
            if (released < clock + 5100).any():
                raise ValueError("Actual target-release clock must respect the original five-second horizon and entry delay")
            mask = noon(clock, day)
            if mask.sum() != 7070:
                raise ValueError(f"Every original noon row must remain: {symbol}/{day} has {mask.sum()}")
            reference_parity = None
            if day in protocol["assessment_dates"]:
                baseline = ROOT / protocol["reference_run"] / "dates" / day / "predictions" / f"{symbol}_original_reference.npz"
                with np.load(baseline, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["labels"], target[mask])
                    np.testing.assert_array_equal(saved["decision_times"], clock[mask])
                with np.load(ROOT / protocol["prior_run"] / "priors" / day / f"{symbol}_original_reference.npz", allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["decision_times"], clock[mask])
                reference_parity = True
            destination = output / "contexts" / day
            destination.mkdir(parents=True, exist_ok=True)
            fp, yp = destination / f"{symbol}_features.parquet", destination / f"{symbol}_outcomes.npz"
            frame.to_parquet(fp, index=False)
            np.savez_compressed(yp, labels=target, decision_times=clock, release_times=released)
            pd.testing.assert_frame_equal(frame.reset_index(drop=True), pd.read_parquet(fp), check_exact=True)
            with np.load(yp, allow_pickle=False) as saved:
                for name, values in (("labels", target), ("decision_times", clock), ("release_times", released)):
                    np.testing.assert_array_equal(saved[name], values)
            stride = calendar_stride_mask(clock, utc_ms(day), stride_seconds=4)
            records.append({"symbol": symbol, "date": day, "observed_rows": len(frame), "stride_four_rows": int(stride.sum()),
                "noon_rows": int(mask.sum()), "source_sha256": original["sha256"], "features": str(fp.relative_to(ROOT)),
                "features_sha256": sha256_file(fp), "outcomes": str(yp.relative_to(ROOT)), "outcomes_sha256": sha256_file(yp),
                "original_assessment_labels_clocks_exact": reference_parity, "native_storage_roundtrip_exact": True,
                "minimum_target_release_delay_ms": int((released - clock).min()), "maximum_release_time": int(released.max()),
                "historical_class_counts_at_stride_four": [int(np.sum(target[stride] == c)) for c in (-1, 0, 1)]})
            del frame, target, clock, raw, released
        del x, y, clocks
        gc.collect()
        write_json(output / "progress.json", {"completed_feature_dates": len(records) // 2, "planned_feature_dates": 25,
            "market_model_fits": 0, "assessment_metrics_computed": False})
        print(f"broad_context_prepared={len(records)//2}/25 date={day}", flush=True)
    write_json(output / "context_manifest.json", {"identity": identity, "columns": schema, "sessions": records,
        "feature_dates": protocol["feature_dates"], "assessment_dates": protocol["assessment_dates"],
        "evidence_status": protocol["evidence_status"], "market_model_fits": 0, "assessment_metrics_computed": False,
        "seconds": time.monotonic() - started,
        "artifact_hashes": {str(p.relative_to(output)): sha256_file(p) for p in output.rglob("*") if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.protocol.resolve(), args.output.resolve())
