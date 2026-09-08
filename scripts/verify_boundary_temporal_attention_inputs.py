"""Replay original temporal inputs and forecasts without training a model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lob_forge.binance_vision import sha256_file
from run_boundary_confirmation import write_json
from run_boundary_tabicl_screen import verify_artifacts
from run_boundary_temporal_attention_screen import prepare

ROOT = Path(__file__).resolve().parents[1]


def run(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if (protocol["purpose"] != "temporal_input_and_original_forecast_replay_only" or protocol["market_model_fits"] != 0
        or protocol["dates"] != ["2023-06-03", "2023-06-07", "2023-06-11"]):
        raise ValueError("Only the fixed score-free input replay may run")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen temporal-input replay source changed: {path}")
    if output.exists():
        raise ValueError("Preserve every previous attention input-replay attempt")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"]}
    write_json(output / "frozen_replay.json", identity)
    begun, records = time.monotonic(), []
    for day in protocol["dates"]:
        folder = output / "dates" / day
        prepare(day, protocol, folder, identity)
        prepared = json.loads((folder / "prepared.json").read_text())
        verify_artifacts(folder, prepared)
        for source, checksum in prepared["input_artifacts"].items():
            if sha256_file(ROOT / source) != checksum:
                raise ValueError("Verified original input changed during its replay")
        records.append({"date": day, "preparation_sha256": sha256_file(folder / "prepared.json"),
            "seconds": prepared["seconds"], "metadata": json.loads((folder / "preparation_metadata.json").read_text()),
            "verified_source_artifacts": len(prepared["input_artifacts"]), "new_model_fits": 0})
        write_json(output / "progress.json", {"dates_verified": len(records), "records": records, "market_model_fits": 0})
        print(f"temporal_attention_original_inputs_verified={day} seconds={prepared['seconds']:.1f}", flush=True)
    write_json(output / "summary.json", {"identity": identity, "records": records, "market_model_fits": 0,
        "new_assessment_metrics_computed": False, "seconds": time.monotonic() - begun,
        "scope": "Original model inference and input/label/release/weight identity only. No attention fit, runtime preflight, new market forecast, model selection or fresh confirmation content.",
        "artifact_hashes": {str(p.relative_to(output)): sha256_file(p) for p in output.rglob("*") if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())
