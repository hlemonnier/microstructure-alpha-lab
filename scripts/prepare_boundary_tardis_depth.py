"""Replay a frozen public native interval onto existing Binance decisions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_tardis_observations import sample_tardis_depth
from run_boundary_confirmation import write_json

ROOT = Path(__file__).resolve().parents[1]


def prepare(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    for name, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / name) != checksum:
            raise ValueError(f"Frozen native-depth preparation input changed: {name}")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"]}
    frozen = output / "frozen_preparation.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve existing native-depth preparation after source changes")
    write_json(frozen, identity)
    if (output / "depth_manifest.json").exists():
        existing = json.loads((output / "depth_manifest.json").read_text())
        if existing["identity"] != identity:
            raise ValueError("Existing native-depth manifest identity changed")
        for record in existing["sessions"]:
            for name, checksum in record["artifact_hashes"].items():
                if sha256_file(ROOT / name) != checksum:
                    raise ValueError("Existing prepared native source bytes changed")
        return
    begin = time.monotonic()
    write_json(output / f"attempt_{time.time_ns()}.json", {"model_fits": 0, "target_columns_read": False})
    raw = json.loads((ROOT / protocol["raw_manifest"]).read_text())
    for record in raw["records"]:
        for name, checksum in record["artifact_hashes"].items():
            if sha256_file(ROOT / name) != checksum:
                raise ValueError("Original raw response bytes changed")
    original = json.loads((ROOT / protocol["original_manifest"]).read_text())
    parents = {r["symbol"]: r for r in original["sessions"] if r["session_date"] == protocol["date"]}
    if set(parents) != set(protocol["symbols"]):
        raise ValueError("The original session must contain both registered assets")
    clocks = {}
    for symbol, record in parents.items():
        file = ROOT / record["features_path"]
        if sha256_file(file) != record["sha256"]:
            raise ValueError("Original decision source changed")
        clock = pd.read_parquet(file, columns=["decision_time"]).decision_time.to_numpy()
        clocks[symbol] = clock[(clock >= protocol["start_ms"]) & (clock < protocol["end_ms"])]
    implementation = {}
    if protocol.get("implementation") == "equivalence_checked_fast_replay":
        from lob_forge.boundary_tardis_fast import FastBinanceFuturesDepthState, iter_tardis_messages_fast

        implementation = {"state_factory": FastBinanceFuturesDepthState, "message_reader": iter_tardis_messages_fast}
    elif protocol.get("implementation") == "captured_quote_and_explicit_tick_frontiers_v1":
        from lob_forge.boundary_tardis_certified import CertifiedBinanceFuturesDepthState
        from lob_forge.boundary_tardis_fast import iter_tardis_messages_fast

        implementation = {"state_factory": CertifiedBinanceFuturesDepthState, "message_reader": iter_tardis_messages_fast}
    elif protocol.get("implementation") is not None:
        raise ValueError("Unrecognized registered depth-replay implementation")
    samples, quotes, checks = sample_tardis_depth([ROOT / r["payload_path"] for r in raw["records"]], clocks,
        delays_ms=tuple(protocol["delays_ms"]), **implementation,
        progress=lambda n, total, lines, done, all_queries: print(f"native_depth_slices={n}/{total} lines={lines} queries={done}/{all_queries}", flush=True))
    sessions = []
    for symbol in protocol["symbols"]:
        folder = output / symbol
        folder.mkdir(exist_ok=True)
        artifacts = {}
        for name, values in (("observations", samples[symbol]), ("native_quotes", quotes[symbol])):
            path = folder / f"{name}.npz"
            np.savez_compressed(path, **values)
            with np.load(path, allow_pickle=False) as restored:
                for key, expected in values.items():
                    np.testing.assert_array_equal(restored[key], expected)
            artifacts[str(path.relative_to(ROOT))] = sha256_file(path)
        parent = parents[symbol]
        if sha256_file(ROOT / parent["features_path"]) != parent["sha256"]:
            raise ValueError("Original source changed during native preparation")
        sessions.append({"symbol": symbol, "session_date": protocol["date"], "rows": len(clocks[symbol]),
            "original_features_path": parent["features_path"], "original_features_sha256": parent["sha256"],
            "observation_path": str((folder / "observations.npz").relative_to(ROOT)),
            "observation_sha256": artifacts[str((folder / "observations.npz").relative_to(ROOT))],
            "artifact_hashes": artifacts, "save_reload_parity": True, "only_original_decision_time_read": True,
            "checks": checks["assets"][symbol]})
    write_json(output / "depth_manifest.json", {"identity": identity, "evidence_status": protocol["evidence_status"],
        "sessions": sessions, "checks": checks, "model_fits": 0, "seconds": time.monotonic() - begin})
    print(f"native_depth_prepared {output / 'depth_manifest.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.protocol.resolve(), args.output.resolve())
    else:
        print("Pass --prepare to replay the frozen public native interval.")
