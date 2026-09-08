"""Bounded synthetic native-runtime preflight for stochastic boosting."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_stochastic_boost import VARIANTS, StochasticBoostForecaster, fit_stochastic_boost
from benchmark_boundary_tabicl_feature_budget import memory_guard
from run_boundary_confirmation import write_json

ROOT = Path(__file__).resolve().parents[1]


def synthetic_history(dimensions):
    rng = np.random.default_rng(20260908)
    columns = [f"synthetic_{i:03}" for i in range(dimensions - 1)]
    features, labels, times = {}, {}, {}
    for asset, symbol in enumerate(("BTCUSDT", "ETHUSDT")):
        counts = [4320, 12960, 4320] if asset == 0 else [5400, 10800, 5400]
        y = np.concatenate([rng.permutation(np.repeat([-1, 0, 1], counts)) for _ in range(4)])
        x = rng.normal(size=(86400, dimensions - 1)).astype(np.float32)
        x[:, 0] += y
        x[:, 1] += np.abs(y)
        features[symbol], labels[symbol] = pd.DataFrame(x, columns=columns), y
        times[symbol] = 1700000000000 + 4000 * np.arange(86400, dtype=np.int64)
    query = pd.DataFrame(rng.normal(size=(512, dimensions - 1)), columns=columns)
    return features, labels, times, query


def worker(protocol_path, output, variant, dimensions):
    import psutil

    protocol = json.loads(protocol_path.read_text())
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen native boosting preflight input changed: {path}")
    psutil.Process().memory_info()
    folder = output / f"{variant}_{dimensions}"
    folder.mkdir()
    readings = {"maximum_sampled_rss_bytes": 0, "maximum_sampled_gpu_driver_bytes": 0, "maximum_sampled_simultaneous_sum_bytes": 0}
    stop = threading.Event()
    guard = threading.Thread(target=memory_guard, args=("cpu", folder, protocol["limits"], stop, readings), daemon=True)
    guard.start()
    started = time.monotonic()
    features, labels, times, query = synthetic_history(dimensions)
    begin = time.monotonic()
    model = fit_stochastic_boost(features, labels, times, variant=variant, iterations=protocol["benchmark_iterations"])
    fit_seconds = time.monotonic() - begin
    begin = time.monotonic()
    model.save(folder / "model")
    restored = StochasticBoostForecaster.load(folder / "model")
    checkpoint_seconds = time.monotonic() - begin
    p = model.predict_proba(query, "BTCUSDT")
    np.testing.assert_array_equal(p, restored.predict_proba(query, "BTCUSDT"))
    changed = query.copy()
    changed.iloc[37:] = 1e100
    future = model.predict_proba(changed, "BTCUSDT")
    shorter = model.predict_proba(query.iloc[:37], "BTCUSDT")
    individual = np.concatenate([model.predict_proba(query.iloc[i:i + 1], "BTCUSDT") for i in range(7)])
    errors = {"future_prefix": float(np.max(np.abs(p[:37] - future[:37]))), "shorter_prefix": float(np.max(np.abs(p[:37] - shorter))),
        "single_query": float(np.max(np.abs(p[:7] - individual)))}
    if max(errors.values()) > protocol["query_probability_absolute_tolerance"]:
        raise ValueError(f"Frozen native boosting query-independence check failed: {errors}")
    for asset, symbol in enumerate(("BTCUSDT", "ETHUSDT")):
        np.testing.assert_array_equal(model.priors[asset], [(labels[symbol] == c).mean() for c in (-1, 0, 1)])
    elapsed = []
    for _ in range(protocol["warmup_query_batches"]):
        model.predict_proba(query.iloc[:64], "BTCUSDT")
    for _ in range(protocol["timed_query_repetitions"]):
        begin = time.monotonic()
        np.testing.assert_array_equal(p, model.predict_proba(query, "BTCUSDT"))
        elapsed.append(time.monotonic() - begin)
    del restored
    gc.collect()
    projected = protocol["market_iterations"] / protocol["benchmark_iterations"] * fit_seconds
    projected += checkpoint_seconds + 2 * 14140 / 512 * float(np.median(elapsed)) * protocol["market_iterations"] / protocol["benchmark_iterations"]
    if projected > protocol["maximum_projected_procedure_seconds"]:
        raise ValueError("Native boosting projection exceeded its frozen procedure ceiling")
    stop.set()
    guard.join()
    np.savez_compressed(folder / "forecasts.npz", probabilities=p)
    write_json(folder / "result.json", {"variant": variant, "dimensions": dimensions, "status": "passed", "backend": "cpu",
        "benchmark_iterations": protocol["benchmark_iterations"], "training_rows": model.training_rows,
        "fit_seconds_including_packing": fit_seconds, "checkpoint_save_reload_seconds": checkpoint_seconds,
        "query_512_repetition_seconds": elapsed, "projected_procedure_seconds": projected,
        "effective_parameters": model.effective_parameters, "training_priors_exact": True, "checkpoint_forecasts_exact": True,
        "query_max_absolute_errors": errors, "memory": readings, "seconds": time.monotonic() - started,
        "market_model_fits": 0, "assessment_metrics_computed": False})


def run(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if protocol["variants"] != list(VARIANTS) or protocol["dimensions"] != [221, 400] or protocol["backend"] != "cpu":
        raise ValueError("Exactly the eight frozen native boosting cases required")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen native boosting preflight input changed: {path}")
    if output.exists():
        raise ValueError("Preserve any previous native boosting preflight attempt")
    output.mkdir(parents=True)
    write_json(output / "attempt.json", {"protocol_sha256": sha256_file(protocol_path), "market_model_fits": 0})
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    records = []
    for variant in protocol["variants"]:
        for dimensions in protocol["dimensions"]:
            name = f"{variant}_{dimensions}"
            command = [sys.executable, str(Path(__file__).resolve()), "--protocol", str(protocol_path), "--output", str(output), "--worker", variant, "--dimensions", str(dimensions)]
            failure, returncode = None, None
            with (output / f"{name}.log").open("w") as log:
                try:
                    result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=protocol["limits"]["worker_seconds"])
                    returncode = result.returncode
                    if returncode:
                        failure = f"Worker exited {returncode}; preserve its complete log"
                except subprocess.TimeoutExpired:
                    failure = "Frozen native boosting worker deadline exceeded"
            record = ({"variant": variant, "dimensions": dimensions, "status": "failed", "failure": failure}
                if failure else json.loads((output / name / "result.json").read_text()))
            record.update(returncode=returncode, log_sha256=sha256_file(output / f"{name}.log"))
            records.append(record)
            write_json(output / "progress.json", {"records": records, "market_model_fits": 0})
            print(f"stochastic_boost_preflight={name} status={record['status']} projected_seconds={record.get('projected_procedure_seconds')}", flush=True)
    passed = len(records) == 8 and all(r["status"] == "passed" for r in records)
    write_json(output / "summary.json", {"protocol_sha256": sha256_file(protocol_path), "records": records, "all_cases_passed": passed,
        "market_model_fits": 0, "assessment_metrics_computed": False,
        "artifact_hashes": {str(p.relative_to(output)): sha256_file(p) for p in output.rglob("*") if p.is_file()}})
    if not passed:
        raise ValueError("The complete native boosting family did not pass its frozen preflight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=list(VARIANTS))
    parser.add_argument("--dimensions", type=int)
    args = parser.parse_args()
    if args.worker:
        worker(args.protocol.resolve(), args.output.resolve(), args.worker, args.dimensions)
    else:
        run(args.protocol.resolve(), args.output.resolve())
