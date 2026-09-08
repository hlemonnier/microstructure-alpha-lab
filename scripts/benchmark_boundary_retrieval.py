"""Synthetic compute and numerical checks for the frozen retrieval procedure."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_retrieval import ARCHITECTURES, HistoricalBankSampler, RetrievalForecaster, build_retrieval_network, retrieval_training_loss
from benchmark_boundary_tabicl_feature_budget import memory_guard, synchronize
from run_boundary_confirmation import write_json

ROOT = Path(__file__).resolve().parents[1]


def synthetic_history(dimensions):
    rng = np.random.default_rng(20260908)
    labels = np.tile(np.repeat([-1, 0, 1], 7200), 8)
    assets = np.repeat(np.tile([0, 1], 4), 21600)
    dates = np.repeat(np.arange(4), 43200)
    matrix = rng.normal(size=(172800, dimensions)).astype(np.float32)
    matrix[:, 0] += labels
    matrix[:, 1] += np.abs(labels)
    matrix[:, -1] = 2 * assets - 1
    queries = rng.normal(size=(512, dimensions)).astype(np.float32)
    queries[:, 0] += np.resize([-1, 0, 1], 512)
    queries[:, -1] = -1
    return matrix, labels, assets, dates, queries


def worker(protocol_path, output, backend, dimensions, architecture):
    import psutil
    import torch

    protocol = json.loads(protocol_path.read_text())
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen retrieval preflight source changed: {path}")
    psutil.Process().memory_info()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if backend == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Registered Mac GPU is unavailable in this execution context")
    folder = output / f"{architecture}_{dimensions}_{backend}"
    folder.mkdir()
    readings = {"maximum_sampled_rss_bytes": 0, "maximum_sampled_gpu_driver_bytes": 0, "maximum_sampled_simultaneous_sum_bytes": 0}
    stop = threading.Event()
    guard = threading.Thread(target=memory_guard, args=(backend, folder, protocol["limits"], stop, readings), daemon=True)
    guard.start()
    begin = time.monotonic()
    matrix, labels, assets, dates, queries = synthetic_history(dimensions)
    sampler = HistoricalBankSampler(labels, assets, dates)
    model = build_retrieval_network(dimensions, architecture).to(backend)
    initial = copy.deepcopy(model.state_dict())
    rng = np.random.default_rng(20260909)
    gradients = {}
    if architecture != "fixed_metric":
        query_rows, key_rows = rng.choice(len(matrix), 1024, replace=False), sampler.sample(rng)
        loss = retrieval_training_loss(model, matrix, labels, sampler, query_rows, key_rows)
        loss.backward()
        gradients = {name: value.grad.detach().cpu().numpy().copy() for name, value in model.named_parameters()}
        if any(not np.isfinite(value).all() for value in gradients.values()):
            raise ValueError("Every retrieval encoder/head gradient must be finite")
    np.savez_compressed(folder / "gradients.npz", **gradients)
    # Unequal natural priors stress the probability-recovery operation. The
    # synthetic bank itself remains uniform by asset/class, by construction.
    priors = {0: np.array([.2, .6, .2]), 1: np.array([.25, .5, .25])}
    learner = RetrievalForecaster(model, None, np.ones(dimensions - 1, dtype=bool), [], priors, {})
    synchronize(backend)
    tick = time.monotonic()
    learner.refresh_bank(matrix, labels, assets)
    synchronize(backend)
    bank_seconds = time.monotonic() - tick
    tick = time.monotonic()
    learner.save(folder / "model")
    restored = RetrievalForecaster.load(folder / "model", device=backend)
    synchronize(backend)
    checkpoint_seconds = time.monotonic() - tick
    p = learner.predict_matrix(queries, 0)
    np.testing.assert_array_equal(p, restored.predict_matrix(queries, 0))
    changed = queries[:128].copy()
    changed[37:] = 1e4
    future = learner.predict_matrix(changed, 0)
    short = learner.predict_matrix(queries[:37], 0)
    single = learner.predict_matrix(queries[:7], 0, batch_size=1)
    rechunk = learner.predict_matrix(queries[:128], 0, batch_size=31, key_chunk_rows=2048)
    errors = {"future_prefix": float(np.max(np.abs(p[:37] - future[:37]))),
        "shorter_prefix": float(np.max(np.abs(p[:37] - short))),
        "single_query": float(np.max(np.abs(p[:7] - single))),
        "query_key_partition": float(np.max(np.abs(p[:128] - rechunk)))}
    if max(errors.values()) > protocol["partition_probability_absolute_tolerance"]:
        raise ValueError(f"Registered retrieval partition tolerance exceeded: {errors}")
    np.savez_compressed(folder / "forecasts.npz", probabilities=p)
    agreement = {"reference_backend": "cpu", "measured": False, "probability_max_absolute_error": None,
        "gradient_relative_l2_error": None, "gradient_max_absolute_error": None}
    if backend == "mps":
        cpu = output / f"{architecture}_{dimensions}_cpu"
        with np.load(cpu / "forecasts.npz", allow_pickle=False) as saved:
            agreement.update(measured=True, probability_max_absolute_error=float(np.max(np.abs(saved["probabilities"] - p))))
        if gradients:
            with np.load(cpu / "gradients.npz", allow_pickle=False) as saved:
                left = np.concatenate([saved[name].ravel().astype(float) for name in gradients])
            right = np.concatenate([gradients[name].ravel().astype(float) for name in gradients])
            agreement.update(gradient_relative_l2_error=float(np.linalg.norm(left - right) / max(np.linalg.norm(left), 1e-12)),
                gradient_max_absolute_error=float(np.max(np.abs(left - right))))
        if agreement["probability_max_absolute_error"] > protocol["backend_probability_absolute_tolerance"] or (gradients and (
            agreement["gradient_relative_l2_error"] > protocol["backend_gradient_relative_l2_tolerance"] or
            agreement["gradient_max_absolute_error"] > protocol["backend_gradient_absolute_tolerance"])):
            raise ValueError(f"Registered CPU/MPS retrieval agreement failed: {agreement}")
    del restored
    gc.collect()
    query_seconds = []
    for _ in range(protocol["warmup_query_batches"]):
        learner.predict_matrix(queries[:64], 0)
    for _ in range(protocol["timed_repetitions"]):
        synchronize(backend)
        tick = time.monotonic()
        again = learner.predict_matrix(queries, 0)
        synchronize(backend)
        query_seconds.append(time.monotonic() - tick)
        np.testing.assert_array_equal(p, again)
    # Timed optimization restarts from the exact original initialization.
    model.load_state_dict(initial)
    training_seconds = []
    if architecture != "fixed_metric":
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        rng = np.random.default_rng(20260909)
        for iteration in range(protocol["warmup_training_steps"] + protocol["timed_repetitions"]):
            synchronize(backend)
            tick = time.monotonic()
            query_rows, key_rows = rng.choice(len(matrix), 1024, replace=False), sampler.sample(rng)
            optimizer.zero_grad()
            loss = retrieval_training_loss(model, matrix, labels, sampler, query_rows, key_rows)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            synchronize(backend)
            if iteration >= protocol["warmup_training_steps"]:
                training_seconds.append(time.monotonic() - tick)
    if any(not torch.isfinite(value).all() for value in model.parameters()):
        raise ValueError("Timed optimizer produced nonfinite parameters")
    median_train = float(np.median(training_seconds)) if training_seconds else 0.
    median_query = float(np.median(query_seconds))
    epochs = 0 if architecture == "fixed_metric" else 6
    projected = epochs * (np.ceil(172800 / 1024) * median_train + bank_seconds + 14140 / 512 * median_query)
    projected += bank_seconds + checkpoint_seconds + 2 * 14140 / 512 * median_query
    if projected > protocol["maximum_projected_procedure_seconds"]:
        raise ValueError("Projected complete retrieval procedure exceeds its frozen timing ceiling")
    stop.set()
    guard.join()
    write_json(folder / "result.json", {"backend": backend, "dimensions": dimensions, "architecture": architecture, "status": "passed",
        "training_step_seconds": training_seconds, "median_training_step_seconds": median_train,
        "query_512_seconds": query_seconds, "median_query_512_seconds": median_query,
        "historical_bank_encoding_seconds": bank_seconds, "checkpoint_save_reload_seconds": checkpoint_seconds,
        "projected_procedure_seconds": float(projected), "backend_agreement": agreement,
        "partition_max_absolute_errors": errors, "checkpoint_forecasts_exact": True, "repeated_forecasts_exact": True,
        "memory": readings, "seconds": time.monotonic() - begin, "market_model_fits": 0, "assessment_metrics_computed": False})


def run(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if protocol["architectures"] != list(ARCHITECTURES) or protocol["dimensions"] != [221, 400] or protocol["backends"] != ["cpu", "mps"]:
        raise ValueError("Exactly sixteen frozen synthetic retrieval cases required")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen retrieval benchmark input changed: {path}")
    if output.exists():
        raise ValueError("Preserve every previous synthetic retrieval attempt")
    output.mkdir(parents=True)
    write_json(output / "attempt.json", {"protocol_sha256": sha256_file(protocol_path), "market_model_fits": 0})
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTORCH_ENABLE_MPS_FALLBACK": "0"}
    records = []
    for architecture in protocol["architectures"]:
        for dimensions in protocol["dimensions"]:
            for backend in protocol["backends"]:
                name = f"{architecture}_{dimensions}_{backend}"
                command = [sys.executable, str(Path(__file__).resolve()), "--protocol", str(protocol_path), "--output", str(output),
                    "--worker", backend, "--dimensions", str(dimensions), "--architecture", architecture]
                failure, returncode = None, None
                with (output / f"{name}.log").open("w") as stream:
                    try:
                        result = subprocess.run(command, env=environment, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                            timeout=protocol["limits"]["worker_seconds"])
                        returncode = result.returncode
                        if returncode:
                            failure = f"Worker exited {returncode}; preserve its log and artifacts"
                    except subprocess.TimeoutExpired:
                        failure = "Registered synthetic worker deadline exceeded"
                record = ({"backend": backend, "dimensions": dimensions, "architecture": architecture, "status": "failed", "failure": failure}
                    if failure else json.loads((output / name / "result.json").read_text()))
                record.update(returncode=returncode, log_sha256=sha256_file(output / f"{name}.log"))
                records.append(record)
                write_json(output / "progress.json", {"records": records, "market_model_fits": 0})
                print(f"retrieval_preflight={name} status={record['status']} projected_seconds={record.get('projected_procedure_seconds')}", flush=True)
    feasible = {backend: sum(r["projected_procedure_seconds"] for r in records if r["backend"] == backend)
        for backend in protocol["backends"] if len([r for r in records if r["backend"] == backend]) == 8
        and all(r["status"] == "passed" for r in records if r["backend"] == backend)}
    chosen = min(feasible, key=feasible.get) if feasible else None
    write_json(output / "summary.json", {"protocol_sha256": sha256_file(protocol_path), "records": records,
        "feasible_backend_summed_projected_seconds": feasible, "selected_backend": chosen,
        "market_model_fits": 0, "assessment_metrics_computed": False,
        "memory_scope": "Sampled RSS and GPU driver allocation; their simultaneous sum can double-count unified memory and is not an instantaneous hard peak.",
        "artifact_hashes": {str(p.relative_to(output)): sha256_file(p) for p in output.rglob("*") if p.is_file()}})
    if chosen is None:
        raise ValueError("No common execution backend passed all frozen retrieval checks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=("cpu", "mps"))
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--architecture", choices=ARCHITECTURES)
    args = parser.parse_args()
    if args.worker:
        worker(args.protocol.resolve(), args.output.resolve(), args.worker, args.dimensions, args.architecture)
    else:
        run(args.protocol.resolve(), args.output.resolve())
