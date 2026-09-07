"""Build and verify full-day forecast data from the already cached development archives."""

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
from lob_forge.boundary_events import build_event_dataset
from lob_forge.boundary_forecasts import OBSERVED_FIELDS

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(output):
    protocol = ROOT / "docs/research/boundary_event_data_20260907.json"
    source_manifest = ROOT / "data/research/performance_pilot_20260907_window_revision/dataset_manifest.json"
    source = json.loads(source_manifest.read_text())
    identity = {
        "protocol_sha256": sha256_file(protocol),
        "source_manifest_sha256": sha256_file(source_manifest),
        "code_hashes": {
            name: sha256_file(ROOT / name)
            for name in ["scripts/prepare_boundary_event_data.py", "src/lob_forge/boundary_events.py"]
        },
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ["numpy", "pandas", "pyarrow"]},
    }
    frozen = output / "frozen_preparation.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve the old preparation and use a new output directory after source changes")
    write_json(frozen, identity)
    sessions = []
    for session in source["sessions"]:
        started = time.monotonic()
        symbol, day = session["symbol"], session["session_date"]
        if day < "2023-05-16" or day > "2023-05-23" or symbol not in ["BTCUSDT", "ETHUSDT"]:
            raise ValueError("Only the registered exposed dates and assets may be opened")
        metadata = output / symbol / f"{day}.json"
        if metadata.exists():
            cached = json.loads(metadata.read_text())
            if cached["identity"] != identity or sha256_file(ROOT / cached["features_path"]) != cached["sha256"]:
                raise ValueError("Cached feature identity mismatch")
            sessions.append(cached)
            continue
        for raw in session["sources"].values():
            path = Path(raw["local_path"])
            if (
                path.stat().st_size != raw["bytes"]
                or sha256_file(path) != raw["sha256"]
                or raw["sha256"] != raw["provider_checksum_sha256"]
            ):
                raise ValueError("Cached raw archive failed its official checksum/size verification")
        frame = build_event_dataset(
            Path(session["sources"]["bookTicker"]["local_path"]),
            Path(session["sources"]["aggTrades"]["local_path"]),
            min_tick=0.1 if symbol == "BTCUSDT" else 0.01,
        )
        canonical_path = Path(session["features_path"])
        if sha256_file(canonical_path) != session["feature"]["sha256"]:
            raise ValueError("Canonical parity reference changed")
        canonical = pd.read_csv(canonical_path)
        canonical = canonical[canonical["decision_time"] >= canonical["decision_time"].min() + 120000].set_index(
            "decision_time"
        )
        actual = frame.set_index("decision_time").loc[canonical.index]
        columns = [name for name in OBSERVED_FIELDS if name != "decision_time"]
        np.testing.assert_allclose(actual[columns].to_numpy(), canonical[columns].to_numpy(), rtol=1e-8, atol=1e-10)
        for name in ["label", "entry_event_time", "future_event_time"]:
            np.testing.assert_array_equal(actual[name], canonical[name])
        deviation = np.max(
            np.abs(actual[columns].to_numpy() - canonical[columns].to_numpy())
            / np.maximum(1, np.abs(canonical[columns].to_numpy()))
        )
        path = output / symbol / f"{day}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
        record = {
            "symbol": symbol,
            "session_date": day,
            "identity": identity,
            "features_path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "rows": len(frame),
            "first_decision_time_ms": int(frame["decision_time"].min()),
            "last_decision_time_ms": int(frame["decision_time"].max()),
            "sources": session["sources"],
            "canonical_parity": {
                "reference_sha256": session["feature"]["sha256"],
                "rows": len(canonical),
                "exact_labels_and_quote_timestamps": True,
                "max_scaled_observation_difference": float(deviation),
            },
            "seconds": time.monotonic() - started,
        }
        write_json(metadata, record)
        sessions.append(record)
        write_json(output / "progress.json", {"completed_sessions": len(sessions), "sessions": sessions})
        print(
            f"prepared={len(sessions)}/16 {symbol} {day} rows={len(frame)} parity_rows={len(canonical)} seconds={record['seconds']:.1f}",
            flush=True,
        )
        del frame, canonical, actual
        gc.collect()
    if len(sessions) != 16:
        raise ValueError("Incomplete registered preparation")
    write_json(
        output / "dataset_manifest.json",
        {
            "identity": identity,
            "sessions": sessions,
            "evidence_status": "development_only_previously_exposed_dates",
            "forecast_only": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/research/boundary_event_data_20260907")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.output)
    else:
        print("Pass --prepare to build the registered development dataset from cached sources.")
