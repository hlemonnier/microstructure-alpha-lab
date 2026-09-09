"""Build frozen quantity/count feature caches as their source sidecars complete."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from lob_forge.boundary_okx_features import counted_depth_features

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ((25, 100), (100, 100), (100, 500))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def prepare(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if output.exists():
        raise ValueError("Preserve existing feature preparation attempts")
    for path, checksum in protocol["input_hashes"].items():
        if digest(ROOT / path) != checksum:
            raise ValueError(f"Frozen counted-feature input changed: {path}")
    if len(protocol["sources"]) != 28 or len({(r["symbol"], r["date"]) for r in protocol["sources"]}) != 28:
        raise ValueError("The complete fixed 28-source inventory is required")
    output.mkdir(parents=True)
    identity = {"protocol_sha256": digest(protocol_path), "input_hashes": protocol["input_hashes"],
        "python": platform.python_version(), "dependencies": {k: version(k) for k in ("numpy", "pandas", "pyarrow")}}
    write(output / "frozen_preparation.json", identity)
    begun, sessions = time.monotonic(), []
    for dependency in protocol["sources"]:
        source_path = ROOT / dependency["record"]
        if "supervision" in dependency:
            supervision = ROOT / dependency["supervision"]
            while not supervision.exists():
                if time.monotonic() - begun > protocol["maximum_total_wall_seconds"]:
                    raise ValueError("Source availability exceeded the registered total wall budget")
                time.sleep(1)
            for attempt in range(5):
                try:
                    status = json.loads(supervision.read_text())
                    break
                except json.JSONDecodeError:
                    if attempt == 4:
                        raise
                    time.sleep(0.1)  # The writer may just have created the file.
            if status["status"] != "complete" or status["record"] != dependency["record"]:
                raise ValueError("A registered source failed preparation; do not substitute dates")
            source_sha = status["record_sha256"]
        else:
            source_sha = dependency["record_sha256"]
        if digest(source_path) != source_sha:
            raise ValueError("Completed source record changed")
        record = json.loads(source_path.read_text())
        for key in ("symbol", "date", "archive_sha256", "features_sha256", "protocol_sha256"):
            if record[key] != dependency[key]:
                raise ValueError(f"Source lineage mismatch for {key}")
        if (record["decision_rows"] > protocol["maximum_rows"] or record["uncompressed_array_bytes"] > 1024**3
            or not record["save_reload_exact"] or record["assessment_labels_decoded"]):
            raise ValueError("The counted source allocation or data-access contract changed")
        path = ROOT / record["observation_path"]
        if digest(path) != record["observation_sha256"] or digest(ROOT / record["features_path"]) != record["features_sha256"]:
            raise ValueError("Source observations or original quote grid changed")
        with np.load(path, allow_pickle=False) as source:
            observations = {k: source[k] for k in source.files}
        canonical = pd.read_parquet(ROOT / record["features_path"], columns=["decision_time", "bid", "ask"])
        folder = output / record["symbol"] / record["date"]
        folder.mkdir(parents=True)
        started, groups = time.monotonic(), {}
        for levels, delay in GROUPS:
            frame = counted_depth_features(canonical, observations, delay_ms=delay, levels=levels, include_counts=True)
            columns = list(frame.columns)
            if frame.memory_usage(index=False, deep=True).sum() > protocol["maximum_feature_matrix_bytes"]:
                raise ValueError("Counted feature matrix exceeds its registered size bound")
            availability = int(frame.venue_available.sum())
            frame.insert(0, "decision_time", canonical.decision_time.to_numpy())
            name = f"top{levels}_{delay}"
            destination = folder / (name + ".parquet")
            frame.to_parquet(destination, index=False, compression="zstd")
            restored = pd.read_parquet(destination)
            pd.testing.assert_frame_equal(frame, restored, check_exact=True)
            groups[name] = {"features_path": str(destination.relative_to(ROOT)), "sha256": digest(destination),
                "columns": columns, "rows": len(frame), "available_rows": availability, "save_reload_exact": True}
            del frame, restored
            gc.collect()
        result = {"symbol": record["symbol"], "date": record["date"], "decision_rows": len(canonical),
            "original_features_path": record["features_path"], "original_features_sha256": record["features_sha256"],
            "source_record": dependency["record"], "source_record_sha256": source_sha,
            "source_observation_sha256": record["observation_sha256"], "groups": groups,
            "original_read_columns": ["decision_time", "bid", "ask"], "assessment_labels_decoded": False,
            "market_model_fits": 0, "seconds": time.monotonic() - started}
        write(folder / "record.json", result)
        sessions.append(result)
        write(output / "progress.json", {"prepared_sources": len(sessions), "planned_sources": 28,
            "latest": [record["symbol"], record["date"]], "market_model_fits": 0})
        print(f"counted_features={len(sessions)}/28 {record['symbol']}/{record['date']} seconds={result['seconds']:.1f}", flush=True)
        del canonical, observations
        gc.collect()
        if time.monotonic() - begun > protocol["maximum_total_wall_seconds"]:
            raise ValueError("Feature preparation exceeded its registered total wall budget")
    for path, checksum in protocol["input_hashes"].items():
        if digest(ROOT / path) != checksum:
            raise ValueError(f"Input changed during counted-feature preparation: {path}")
    write(output / "feature_manifest.json", {"identity": identity, "complete": True, "sessions": sessions,
        "market_model_fits": 0, "assessment_labels_decoded": False, "independent_confirmation_data_opened": False,
        "seconds": time.monotonic() - begun, "evidence_status": "exposed_development_counted_features"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.protocol.resolve(), args.output.resolve())
