"""Full-size synthetic CPU preflight for bounded proper neural losses."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_pooled import PooledForecaster, balanced_asset_weights, build_member_network
from lob_forge.boundary_proper_score import VARIANTS, fit_proper_score, predict_matrix
from benchmark_boundary_tabicl_feature_budget import memory_guard
from run_boundary_confirmation import write_json

ROOT = Path(__file__).resolve().parents[1]


def synthetic_history(dimensions):
    rng = np.random.default_rng(20260908)
    labels = np.concatenate([np.concatenate([rng.permutation(np.repeat([-1, 0, 1], counts)) for _ in range(4)])
        for counts in ([4320, 12960, 4320], [5400, 10800, 5400])])
    assets = np.repeat([0, 1], 86400)
    matrix = rng.normal(size=(len(labels), dimensions)).astype(np.float32)
    matrix[:, 0] += labels
    matrix[:, 1] += np.abs(labels)
    matrix[:, -1] = 2 * assets - 1
    priors, _ = balanced_asset_weights(labels, assets)
    validation, validation_y = {}, {}
    for a in (0, 1):
        y = rng.choice([-1, 0, 1], 7070, p=priors[a])
        x = rng.normal(size=(7070, dimensions)).astype(np.float32)
        x[:, 0] += y
        x[:, 1] += np.abs(y)
        x[:, -1] = 2 * a - 1
        validation[a], validation_y[a] = x, y
    # The input is already a synthetic normalized matrix. No raw-feature
    # transformer is fitted here; the real QT persistence has its own unit test.
    original = PooledForecaster(build_member_network(dimensions, hidden_size=64, members=1), None,
        np.ones(dimensions - 1, dtype=bool), [f"synthetic_{i:03}" for i in range(dimensions - 1)], priors, 1, 64)
    return original, matrix, labels, assets, validation, validation_y


def worker(protocol_path, output, variant, dimensions):
    import psutil
    import torch

    protocol = json.loads(protocol_path.read_text())
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen proper-score preflight input changed: {path}")
    psutil.Process().memory_info()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    folder = output / f"{variant}_{dimensions}"
    folder.mkdir()
    readings = {"maximum_sampled_rss_bytes": 0, "maximum_sampled_gpu_driver_bytes": 0, "maximum_sampled_simultaneous_sum_bytes": 0}
    stop = threading.Event()
    guard = threading.Thread(target=memory_guard, args=("cpu", folder, protocol["limits"], stop, readings), daemon=True)
    guard.start()
    begin = time.monotonic()
    original, matrix, labels, assets, validation, validation_y = synthetic_history(dimensions)
    tick = time.monotonic()
    model, history = fit_proper_score(original, matrix, labels, assets, validation, validation_y, variant=variant)
    fit_seconds = time.monotonic() - tick
    if any(not torch.isfinite(p).all() or p.grad is None or not torch.isfinite(p.grad).all() for p in model.network.parameters()):
        raise ValueError("Finite final synthetic neural parameters and gradients required")
    tick = time.monotonic()
    model.save(folder / "model")
    restored = PooledForecaster.load(folder / "model")
    checkpoint_seconds = time.monotonic() - tick
    write_json(folder / "synthetic_training_history.json", history)
    forecasts, checks, query_times = {}, {}, {}
    for a in (0, 1):
        p = predict_matrix(model, validation[a], a)
        np.testing.assert_array_equal(p, predict_matrix(restored, validation[a], a))
        np.testing.assert_array_equal(model.priors[a], original.priors[a])
        prefix = validation[a][:128]
        before = predict_matrix(model, prefix, a)
        changed = prefix.copy()
        changed[37:, :-1] = 1e4
        future = predict_matrix(model, changed, a)
        shorter = predict_matrix(model, prefix[:37], a)
        single = predict_matrix(model, prefix[:7], a, batch_size=1)
        errors = {"future_prefix": float(np.max(np.abs(before[:37] - future[:37]))),
            "full_query_prefix": float(np.max(np.abs(p[:128] - before))),
            "shorter_prefix": float(np.max(np.abs(p[:37] - shorter))), "single_query": float(np.max(np.abs(p[:7] - single)))}
        if max(errors.values()) > protocol["query_probability_absolute_tolerance"]:
            raise ValueError(f"Registered proper-score query independence failed: {errors}")
        timed = []
        for _ in range(protocol["timed_full_query_repetitions"]):
            tick = time.monotonic()
            np.testing.assert_array_equal(p, predict_matrix(model, validation[a], a))
            timed.append(time.monotonic() - tick)
        forecasts[str(a)], checks[str(a)], query_times[str(a)] = p, errors, timed
    stop.set()
    guard.join()
    np.savez_compressed(folder / "forecasts.npz", **forecasts)
    write_json(folder / "result.json", {"variant": variant, "dimensions": dimensions, "status": "passed", "backend": "cpu",
        "training_rows": len(labels), "validation_rows_per_asset": 7070, "complete_training_epochs": 12,
        "optimizer_steps": history["optimizer_steps"], "selected_epoch": history["best_epoch"], "fit_seconds": fit_seconds,
        "checkpoint_seconds": checkpoint_seconds, "query_7070_seconds": query_times, "query_max_absolute_errors": checks,
        "checkpoint_and_repeated_forecasts_exact": True, "training_priors_exact": True, "selected_parameters_and_last_optimizer_gradients_finite": True,
        "memory": readings, "seconds": time.monotonic() - begin, "market_model_fits": 0, "market_assessment_metrics_computed": False,
        "normalization_scope": "Synthetic normalized inputs; no raw-feature transform or transform timing is included."})


def run(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if protocol["variants"] != list(VARIANTS) or protocol["dimensions"] != [221, 400] or protocol["backend"] != "cpu":
        raise ValueError("Exactly the eight frozen proper-score CPU cases required")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen proper-score preflight input changed: {path}")
    if output.exists():
        raise ValueError("Preserve every previous proper-score preflight attempt")
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
                    failure = "Frozen proper-score worker deadline exceeded"
            record = ({"variant": variant, "dimensions": dimensions, "status": "failed", "failure": failure}
                if failure else json.loads((output / name / "result.json").read_text()))
            record.update(returncode=returncode, log_sha256=sha256_file(output / f"{name}.log"))
            records.append(record)
            write_json(output / "progress.json", {"records": records, "market_model_fits": 0})
            print(f"proper_score_preflight={name} status={record['status']} seconds={record.get('seconds')}", flush=True)
    passed = len(records) == 8 and all(r["status"] == "passed" for r in records)
    write_json(output / "summary.json", {"protocol_sha256": sha256_file(protocol_path), "records": records, "all_cases_passed": passed,
        "market_model_fits": 0, "market_assessment_metrics_computed": False,
        "artifact_hashes": {str(p.relative_to(output)): sha256_file(p) for p in output.rglob("*") if p.is_file()}})
    if not passed:
        raise ValueError("The complete proper-score family did not pass its frozen synthetic preflight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=VARIANTS)
    parser.add_argument("--dimensions", type=int)
    args = parser.parse_args()
    if args.worker:
        worker(args.protocol.resolve(), args.output.resolve(), args.worker, args.dimensions)
    else:
        run(args.protocol.resolve(), args.output.resolve())
