"""Prepare native queue-flow observations without reading target columns."""

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
from lob_forge.boundary_bybit_depth import bybit_archive_messages
from lob_forge.boundary_bybit_flow import FLOW_FIELDS, native_bybit_flow_events, sample_native_bybit_flow
from run_boundary_confirmation import write_json

ROOT = Path(__file__).resolve().parents[1]


def prepare(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    for name, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / name) != checksum:
            raise ValueError(f"Registered native-flow input changed: {name}")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"],
                "python": platform.python_version(), "dependencies": {k: version(k) for k in ("numpy", "pandas", "pyarrow")}}
    frozen = output / "frozen_preparation.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve existing flow preparation after input changes")
    write_json(frozen, identity)
    parents = json.loads((ROOT / protocol["source_manifest"]).read_text())
    sources = {(r["symbol"], r["session_date"]): r for r in parents["sessions"]}
    records = []
    for count, selected in enumerate(protocol["sessions"], 1):
        symbol, day = selected["symbol"], selected["session_date"]
        source = sources[symbol, day]
        folder = output / symbol / day
        completed = folder / "completed.json"
        if completed.exists():
            record = json.loads(completed.read_text())
            if record["identity"] != identity or any(sha256_file(ROOT / p) != h for p, h in record["artifact_hashes"].items()):
                raise ValueError("Completed flow artifacts changed")
            records.append(record)
            continue
        begin = time.monotonic()
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / f"attempt_{time.time_ns()}.json", {"source": source["archive_path"], "model_fits": 0})
        archive, features, depth_file = [ROOT / source[k] for k in ("archive_path", "original_features_path", "observation_path")]
        for file, expected in ((archive, source["archive_sha256"]), (features, source["original_features_sha256"]), (depth_file, source["observation_sha256"])):
            if sha256_file(file) != expected:
                raise ValueError("Frozen archive, canonical observations and original depth required")
        clock = pd.read_parquet(features, columns=["decision_time"]).decision_time.to_numpy()
        events, checks = native_bybit_flow_events(bybit_archive_messages(archive), symbol)
        observations = sample_native_bybit_flow(events, clock, delays_ms=tuple(protocol["delays_ms"]), windows_ms=tuple(protocol["windows_ms"]))
        with np.load(depth_file, allow_pickle=False) as original:
            np.testing.assert_array_equal(clock, original["decision_times"])
            np.testing.assert_array_equal(observations["delays_ms"], original["delays_ms"])
            np.testing.assert_array_equal(observations["source_publisher_times"], original["publisher_times"])
            ages = clock[:, None] - observations["delays_ms"][None, :] - original["publisher_times"]
            valid = original["available"] & (ages <= 1000)
            np.testing.assert_array_equal(observations["top_depth"], np.where(valid, original["depth"][:, :, 0, 1] + original["depth"][:, :, 0, 3], 0))
        artifacts = {}
        for name, values in (("native_events", events), ("observations", observations)):
            file = folder / f"{name}.npz"
            np.savez_compressed(file, **values)
            with np.load(file, allow_pickle=False) as restored:
                for key, expected in values.items():
                    np.testing.assert_array_equal(restored[key], expected)
            artifacts[str(file.relative_to(ROOT))] = sha256_file(file)
        if sha256_file(features) != source["original_features_sha256"]:
            raise ValueError("Canonical source observations changed during preparation")
        record = {"identity": identity, "symbol": symbol, "session_date": day,
                  "original_features_path": source["original_features_path"], "original_features_sha256": source["original_features_sha256"],
                  "observation_path": str((folder / "observations.npz").relative_to(ROOT)),
                  "observation_sha256": artifacts[str((folder / "observations.npz").relative_to(ROOT))],
                  "fields": list(FLOW_FIELDS), "rows": len(clock), "checks": checks,
                  "covered_window_rows": observations["window_available"].sum(axis=0).tolist(),
                  "only_decision_time_column_read": True, "original_publisher_and_depth_parity": True,
                  "save_reload_parity": True, "artifact_hashes": artifacts, "model_fits": 0,
                  "seconds": time.monotonic() - begin}
        write_json(completed, record)
        records.append(record)
        print(f"native_flow_prepared={count}/{len(protocol['sessions'])} {symbol}/{day} seconds={record['seconds']:.1f}", flush=True)
        del events, observations
        gc.collect()
    write_json(output / "flow_manifest.json", {"identity": identity, "evidence_status": protocol["evidence_status"], "sessions": records, "model_fits": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_bybit_flow_preflight_20260907.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/research/boundary_bybit_flow_preflight_20260907")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.protocol.resolve(), args.output.resolve())
    else:
        print("Pass --prepare for the registered native queue-flow preparation.")
