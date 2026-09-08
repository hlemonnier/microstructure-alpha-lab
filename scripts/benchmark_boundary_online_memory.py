"""Score-free production-dimension checks for online and recurrent procedures."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_event_inputs import utc_ms
from lob_forge.boundary_fading_memory import FadingMemoryMap, observe_selected
from lob_forge.boundary_pooled import PooledForecaster, build_member_network
from lob_forge.boundary_prequential import SYMBOLS, run_prequential
from run_boundary_confirmation import write_json

ROOT = Path(__file__).resolve().parents[1]


def original(dimensions):
    import torch
    from sklearn.preprocessing import QuantileTransformer

    torch.manual_seed(20260908)
    rng = np.random.default_rng(20260908)
    columns = [f"synthetic_{i:03}" for i in range(dimensions)]
    raw = rng.normal(size=(4096, dimensions))
    normalizer = QuantileTransformer(n_quantiles=1024, output_distribution="normal", random_state=20260908).fit(raw)
    network = build_member_network(dimensions + 1, members=1, hidden_size=64)
    network.eval()
    return PooledForecaster(network, normalizer, np.ones(dimensions, dtype=bool), columns,
        {0: np.array([.2, .6, .2]), 1: np.array([.25, .5, .25])}, 1, 64)


def prequential(dimensions, output):
    rng, base = np.random.default_rng(20260908), original(dimensions)
    day = "2023-06-03"
    times = np.arange(utc_ms(day, "10:30:00"), utc_ms(day, "14:00:00"), 1000, dtype=np.int64)
    keep = (times >= utc_ms(day, "12:02:00")) & (times < utc_ms(day, "13:59:50"))
    stream = {s: pd.DataFrame(rng.normal(size=(len(times), dimensions)), columns=base.columns) for s in SYMBOLS}
    labels = {s: rng.integers(-1, 2, len(times)) for s in SYMBOLS}
    query = {s: stream[s].loc[keep].copy() for s in SYMBOLS}
    replay_assets = np.repeat([0, 1], 3072)
    replay_y = np.tile(np.repeat([-1, 0, 1], 1024), 2)
    replay_x = np.column_stack([rng.normal(size=(6144, dimensions)), 2 * replay_assets - 1]).astype(np.float32)
    anchors = np.arange(utc_ms(day, "11:00:00"), utc_ms(day, "14:00:00"), 60000, dtype=np.int64)
    begin = time.monotonic()
    predictions, state, history = run_prequential(base, "full_replay_fast", stream, labels,
        {s: times for s in SYMBOLS}, {s: times + 5100 for s in SYMBOLS}, query, {s: times[keep] for s in SYMBOLS},
        (replay_x, replay_y, replay_assets), anchors)
    elapsed = time.monotonic() - begin
    if history["optimizer_steps"] != 180 or any(p.shape != (7070, 3) for p in predictions.values()):
        raise ValueError("Production-size synthetic trajectory changed")
    write_json(output / f"prequential_{dimensions}_history.json", history)
    return {"kind": "prequential", "dimensions": dimensions, "seconds": elapsed,
        "optimizer_steps": state.updates, "query_rows": 14140, "replay_rows": 6144,
        "maximum_update_seconds": history["max_update_seconds"], "publication_allowance_seconds": 10,
        "finite_normalized_forecasts": True, "assessment_metrics_computed": False}


def fading_memory(dimensions, output):
    rng, base = np.random.default_rng(20260908), original(dimensions)
    generator = FadingMemoryMap.from_checkpoint(base)
    x = pd.DataFrame(rng.normal(size=(16384, dimensions)), columns=base.columns)
    times = np.arange(len(x), dtype=np.int64) * 1000
    keep = np.arange(len(x)) % 4 == 0
    begin = time.monotonic()
    values, record = observe_selected(generator, x, times, 0, keep)
    elapsed = time.monotonic() - begin
    changed = x.copy()
    changed.iloc[8192:] = 1e8
    future, _ = observe_selected(generator, changed, times, 0, keep)
    shorter, _ = observe_selected(generator, x.iloc[:8192], times[:8192], 0, keep[:8192])
    rechunked, _ = observe_selected(generator, x, times, 0, keep, chunk_rows=1024)
    errors = {}
    for name in values:
        np.testing.assert_array_equal(future[name][:2048], values[name][:2048])
        np.testing.assert_array_equal(shorter[name], values[name][:2048])
        np.testing.assert_allclose(rechunked[name], values[name], rtol=0, atol=1e-7)
        errors[name] = float(np.max(np.abs(rechunked[name] - values[name])))
    generator.save(output / f"generator_{dimensions}.joblib")
    return {"kind": "fading_memory", "dimensions": dimensions, "seconds": elapsed, **record,
        "observations_per_second": len(x) / elapsed, "future_prefix_max_error": 0,
        "shorter_stream_max_error": 0, "chunking_max_errors": errors, "finite_bounded_states": True}


def run(protocol_path, output):
    import torch

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads(protocol_path.read_text())
    if protocol["cases"] != ["prequential", "fading_memory"] or protocol["dimensions"] != [220, 399]:
        raise ValueError("Exactly the four registered synthetic cases are required")
    if output.exists():
        raise ValueError("Preserve any previous benchmark attempt")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen synthetic input changed: {path}")
    output.mkdir(parents=True)
    write_json(output / "attempt.json", {"protocol_sha256": sha256_file(protocol_path), "market_model_fits": 0})
    records = []
    for kind in protocol["cases"]:
        for dimensions in protocol["dimensions"]:
            record = (prequential if kind == "prequential" else fading_memory)(dimensions, output)
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            record["process_high_water_rss_bytes"] = int(peak if platform.system() == "Darwin" else 1024 * peak)
            records.append(record)
            write_json(output / f"{kind}_{dimensions}.json", record)
            print(f"synthetic={kind}/{dimensions} seconds={record['seconds']:.3f}", flush=True)
    write_json(output / "summary.json", {"protocol_sha256": sha256_file(protocol_path), "records": records,
        "market_model_fits": 0, "assessment_metrics_computed": False,
        "artifact_hashes": {str(p.relative_to(output)): sha256_file(p) for p in output.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())
