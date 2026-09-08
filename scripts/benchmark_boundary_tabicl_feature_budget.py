"""Synthetic fixed-feature transformer context scaling on CPU and Mac GPU."""

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
from lob_forge.boundary_tabicl import SYMBOLS
from lob_forge.boundary_tabicl_backend import fit_backend_context, load_backend_context
from run_boundary_confirmation import write_json

ROOT = Path(__file__).resolve().parents[1]


def memory_guard(backend, folder, limits, stop, readings):
    import psutil
    import torch

    process = psutil.Process()
    while not stop.is_set():
        rss = int(process.memory_info().rss)
        driver = int(torch.mps.driver_allocated_memory()) if backend == "mps" else 0
        readings["maximum_sampled_rss_bytes"] = max(readings["maximum_sampled_rss_bytes"], rss)
        readings["maximum_sampled_gpu_driver_bytes"] = max(readings["maximum_sampled_gpu_driver_bytes"], driver)
        readings["maximum_sampled_simultaneous_sum_bytes"] = max(readings["maximum_sampled_simultaneous_sum_bytes"], rss + driver)
        if rss > limits["worker_rss_bytes"] or rss + driver > limits["rss_plus_driver_bytes"]:
            write_json(folder / "resource_failure.json", {"reason": "registered_sampled_memory_ceiling", "rss_bytes": rss, "gpu_driver_bytes": driver, "readings": dict(readings)})
            os._exit(3)
        stop.wait(limits["memory_sample_seconds"])


def synchronize(backend):
    if backend == "mps":
        import torch
        torch.mps.synchronize()


def worker(protocol_path, output, backend, rows):
    import torch

    protocol = json.loads(protocol_path.read_text())
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if backend == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Mac GPU unavailable in this execution context")
    folder = output / f"context{rows}_{backend}"
    folder.mkdir()
    readings = {"maximum_sampled_rss_bytes": 0, "maximum_sampled_gpu_driver_bytes": 0, "maximum_sampled_simultaneous_sum_bytes": 0}
    stop = threading.Event()
    guard = threading.Thread(target=memory_guard, args=(backend, folder, protocol["limits"], stop, readings), daemon=True)
    guard.start()
    begin = time.monotonic()
    rng = np.random.default_rng(20260908)
    columns = [f"synthetic_{i:03}" for i in range(96)]
    features, labels = {}, {}
    for symbol in SYMBOLS:
        y = np.tile([-1, 0, 1], rows // 6)
        x = rng.normal(size=(rows // 2, 96))
        x[:, 0] += y
        x[:, 1] += np.abs(y)
        features[symbol], labels[symbol] = pd.DataFrame(x, columns=columns), y
    priors = {SYMBOLS[0]: np.array([.2, .6, .2]), SYMBOLS[1]: np.array([.25, .5, .25])}
    query = pd.DataFrame(rng.normal(size=(512, 96)), columns=columns)
    checkpoint = ROOT / protocol["checkpoint"]
    started = time.monotonic()
    model = fit_backend_context(features, labels, priors, checkpoint=checkpoint,
        checkpoint_sha256=protocol["checkpoint_sha256"], backend=backend)
    synchronize(backend)
    fit_seconds = time.monotonic() - started
    model.save(folder / "model")
    restored = load_backend_context(folder / "model", backend=backend)
    synchronize(backend)
    ready_seconds = time.monotonic() - started
    symbol = SYMBOLS[0]
    p = model.predict_proba(query, symbol)
    np.testing.assert_array_equal(p, restored.predict_proba(query, symbol))
    prefix = query.iloc[:64]
    initial = p[:64]
    later = prefix.copy()
    later.iloc[17:] = 1e100
    changed = model.predict_proba(later, symbol)
    shorter = model.predict_proba(prefix.iloc[:17], symbol)
    single = model.predict_proba(prefix.iloc[:7], symbol, batch_size=1)
    errors = {"future_prefix": float(np.max(np.abs(initial[:17] - changed[:17]))),
        "shorter_prefix": float(np.max(np.abs(initial[:17] - shorter))),
        "single_query": float(np.max(np.abs(initial[:7] - single)))}
    if errors["future_prefix"] > protocol["future_prefix_absolute_tolerance"] or max(errors["shorter_prefix"], errors["single_query"]) > protocol["query_partition_absolute_tolerance"]:
        raise ValueError(f"Registered synthetic transformer causality tolerance exceeded: {errors}")
    disagreement = 0.
    if backend == "mps":
        with np.load(output / f"context{rows}_cpu" / "forecasts.npz", allow_pickle=False) as saved:
            disagreement = float(np.max(np.abs(saved["probabilities"] - p)))
        if disagreement > protocol["backend_probability_absolute_tolerance"]:
            raise ValueError(f"Registered CPU/MPS probability tolerance exceeded: {disagreement}")
    np.savez_compressed(folder / "forecasts.npz", probabilities=p)
    del restored
    gc.collect()
    for _ in range(protocol["warmup_query_batches"]):
        model.predict_proba(prefix, symbol)
    elapsed = []
    for _ in range(protocol["timed_query_repetitions"]):
        synchronize(backend)
        started = time.monotonic()
        again = model.predict_proba(query, symbol)
        synchronize(backend)
        elapsed.append(time.monotonic() - started)
        np.testing.assert_array_equal(p, again)
    if sha256_file(checkpoint) != protocol["checkpoint_sha256"]:
        raise ValueError("Pretrained checkpoint changed")
    score = ready_seconds + protocol["projected_assessment_rows"] / len(query) * float(np.median(elapsed))
    stop.set()
    guard.join()
    result = {"backend": backend, "context_rows": rows, "features_including_asset": 97, "status": "passed",
        "fit_seconds": fit_seconds, "fit_save_reload_ready_seconds": ready_seconds, "query_rows": len(query),
        "query_repetition_seconds": elapsed, "projected_ready_plus_full_query_seconds": score,
        "cpu_mps_probability_max_absolute_error": disagreement, "causality_max_absolute_errors": errors,
        "checkpoint_forecasts_exact": True, "repeated_query_forecasts_exact": True, "pretrained_weights_unchanged": True,
        "memory": dict(readings), "memory_scope": "Sampled process RSS and MPS driver allocation; their simultaneous sum may double-count unified memory and is not an instantaneous hard peak.",
        "seconds": time.monotonic() - begin, "market_model_fits": 0, "assessment_metrics_computed": False}
    write_json(folder / "result.json", result)


def run(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if protocol["context_rows"] != [3072, 12288] or protocol["backends"] != ["cpu", "mps"] or protocol["features_including_asset"] != 97:
        raise ValueError("Exactly the four registered synthetic transformer cases required")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen context benchmark input changed: {path}")
    if output.exists():
        raise ValueError("Preserve every prior synthetic benchmark attempt")
    output.mkdir(parents=True)
    write_json(output / "attempt.json", {"protocol_sha256": sha256_file(protocol_path), "market_model_fits": 0})
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0", "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"}
    records = []
    for rows in protocol["context_rows"]:
        for backend in protocol["backends"]:
            name = f"context{rows}_{backend}"
            failure = None
            command = [sys.executable, str(Path(__file__).resolve()), "--protocol", str(protocol_path), "--output", str(output), "--worker", backend, "--context-rows", str(rows)]
            with (output / f"{name}.log").open("w") as stream:
                try:
                    result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=protocol["limits"]["worker_seconds"])
                    if result.returncode:
                        failure = f"Worker exited {result.returncode}; preserve complete log and artifacts."
                except subprocess.TimeoutExpired:
                    failure = "Registered worker timeout exceeded; worker terminated."
            record = {"backend": backend, "context_rows": rows, "status": "failed", "failure": failure} if failure else json.loads((output / name / "result.json").read_text())
            records.append(record)
            write_json(output / "progress.json", {"records": records, "market_model_fits": 0})
            print(f"tabicl_feature_preflight={name} status={record['status']} projected_seconds={record.get('projected_ready_plus_full_query_seconds')}", flush=True)
    feasible = {b: sum(r["projected_ready_plus_full_query_seconds"] for r in records if r["backend"] == b)
        for b in protocol["backends"] if all(r["status"] == "passed" for r in records if r["backend"] == b)}
    chosen = min(feasible, key=feasible.get) if feasible else None
    write_json(output / "summary.json", {"protocol_sha256": sha256_file(protocol_path), "records": records,
        "feasible_backend_summed_projected_seconds": feasible, "selected_backend": chosen, "market_model_fits": 0,
        "assessment_metrics_computed": False, "artifact_hashes": {str(p.relative_to(output)): sha256_file(p) for p in output.rglob("*") if p.is_file()}})
    if chosen is None:
        raise ValueError("No common backend passed the registered feature/context benchmark")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=("cpu", "mps"))
    parser.add_argument("--context-rows", type=int)
    args = parser.parse_args()
    if args.worker:
        worker(args.protocol.resolve(), args.output.resolve(), args.worker, args.context_rows)
    else:
        run(args.protocol.resolve(), args.output.resolve())
