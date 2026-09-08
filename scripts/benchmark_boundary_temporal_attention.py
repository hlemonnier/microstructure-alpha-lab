"""Frozen, score-free numerical and block-runtime checks for temporal attention."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import subprocess
import sys
import threading
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_learned_memory import corrected_posterior
from lob_forge.boundary_temporal_attention import ARCHITECTURES, CONFIG, accumulate_attention_block, build_attention_network, contiguous_segments, predict_deltas
from benchmark_boundary_tabicl_feature_budget import memory_guard, synchronize
from run_boundary_confirmation import write_json

ROOT = Path(__file__).resolve().parents[1]
PREREQUISITES = [f"results/boundary_{name}_screen_20260908/summary.json"
    for name in ("stochastic_boost", "proper_score", "broad_accuracy")]


def synthetic_block(dimensions):
    rng = np.random.default_rng(20260908)
    # A middle block has 127 prefix rows followed by 512 new elapsed seconds.
    matrix = rng.normal(size=(8, 639, dimensions)).astype(np.float32)
    assets = np.repeat([0, 1], 4)
    matrix[:, :, -1] = 2 * assets[:, None] - 1
    available = np.ones((8, 639), dtype=bool)
    available[:, :20] = False
    available[0, 231:242] = False
    available[4, 401:418] = False
    matrix[~available] = 1e25
    positions = np.arange(127, 639, 4)
    keep = available[:, positions].copy()
    logits = rng.normal(0, .4, size=(8, len(positions), 3)).astype(np.float32)
    labels = rng.choice([-1, 0, 1], size=keep.shape, p=[.2, .6, .2])
    weights = np.zeros(keep.shape, dtype=np.float32)
    # Synthetic cell counts are frozen once for the whole benchmark trajectory.
    # They are not actual four-day market frequencies or recomputed per update.
    for asset in (0, 1):
        for label in (-1, 0, 1):
            selected = keep & (assets[:, None] == asset) & (labels == label)
            if not selected.any():
                raise ValueError("Every synthetic asset/class cell must be represented")
            weights[selected] = keep.sum() / (6 * selected.sum())
    return matrix, contiguous_segments(available), positions, logits, labels, weights, keep


def synthetic_query(dimensions):
    rng = np.random.default_rng(20260909)
    matrix = rng.normal(size=(7190, dimensions)).astype(np.float32)
    matrix[:, -1] = -1
    clock = np.arange(len(matrix), dtype=np.int64) * 1000
    clock[3700:] += 2000
    selected = np.arange(len(matrix)) >= 120
    logits = rng.normal(0, .4, size=(int(selected.sum()), 3))
    prior = np.array([.2, .6, .2])
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True)) * prior
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return matrix, clock, selected, logits, prior, probabilities


def checked_protocol(path):
    protocol = json.loads(path.read_text())
    if (protocol["architectures"] != list(ARCHITECTURES) or protocol["dimensions"] != [221, 400]
        or protocol["backends"] != ["cpu", "mps"] or protocol["config"] != CONFIG
        or protocol["warmup_updates"] != 3 or protocol["timed_blocks"] != 8
        or protocol["timed_full_query_repetitions"] != 3):
        raise ValueError("Only the sixteen fixed architecture/width/backend cases may run")
    if (protocol["completed_prerequisite_summaries"] != PREREQUISITES
        or any(source not in protocol["input_hashes"] for source in PREREQUISITES)):
        raise ValueError("The existing boosting, proper-score and broad studies must complete and be pinned first")
    for source, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / source) != checksum:
            raise ValueError(f"Frozen temporal-attention preflight input changed: {source}")
    return protocol


def choose_backend(records):
    expected = {(architecture, dimensions) for architecture in ARCHITECTURES for dimensions in (221, 400)}
    if len(records) != 16:
        raise ValueError("All sixteen preflight cases, including failures, must be recorded")
    feasible = {}
    for backend in ("cpu", "mps"):
        subset = [row for row in records if row["backend"] == backend]
        if len(subset) != 8 or {(r["architecture"], r["dimensions"]) for r in subset} != expected:
            raise ValueError("Exactly one result per registered case is required")
        if all(row["status"] == "passed" for row in subset):
            values = [float(row["median_block_seconds"]) for row in subset]
            if not np.isfinite(values).all() or min(values) <= 0:
                raise ValueError("Passing backend timings must be finite and positive")
            feasible[backend] = float(sum(values))
    # Insertion order intentionally makes CPU win an exact tie.
    return (min(feasible, key=feasible.get) if feasible else None), feasible


def worker(protocol_path, output, backend, dimensions, architecture):
    import psutil
    import torch

    protocol = checked_protocol(protocol_path)
    if backend not in protocol["backends"] or dimensions not in protocol["dimensions"] or architecture not in ARCHITECTURES:
        raise ValueError("Unregistered temporal-attention worker")
    if backend == "mps" and not torch.backends.mps.is_available():
        raise ValueError("The registered Mac GPU is unavailable")
    psutil.Process().memory_info()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    folder = output / f"{architecture}_{dimensions}_{backend}"
    folder.mkdir()
    readings = {"maximum_sampled_rss_bytes": 0, "maximum_sampled_gpu_driver_bytes": 0, "maximum_sampled_simultaneous_sum_bytes": 0}
    stop = threading.Event()
    guard = threading.Thread(target=memory_guard, args=(backend, folder, protocol["limits"], stop, readings), daemon=True)
    guard.start()
    begun = time.monotonic()
    inputs = synthetic_block(dimensions)
    model = build_attention_network(dimensions, architecture).to(backend)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

    def update():
        model.train()
        synchronize(backend)
        tick = time.monotonic()
        optimizer.zero_grad()
        record = accumulate_attention_block(model, *inputs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_norm_limit"], error_if_nonfinite=True)
        optimizer.step()
        synchronize(backend)
        seconds = time.monotonic() - tick
        if seconds > protocol["limits"]["block_seconds"]:
            raise ValueError("A training block exceeded its frozen wall-time ceiling")
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Nonfinite post-update temporal parameters")
        return seconds, record

    warmup = [update()[0] for _ in range(protocol["warmup_updates"])]
    warm_state = copy.deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()})
    if not torch.count_nonzero(warm_state["head.weight"]):
        raise ValueError("Numerical checks require genuinely nonzero learned corrections")
    torch.save(warm_state, folder / "independent_warmup.pt")
    matrix, clock, selected, logits, prior, original = synthetic_query(dimensions)
    own_delta, _ = predict_deltas(model, matrix, clock, selected=selected)
    own_probability = corrected_posterior(logits, own_delta, prior, original)
    np.savez_compressed(folder / "independent_warmup_forecasts.npz", probabilities=own_probability)
    checked = build_attention_network(dimensions, architecture).to(backend)
    comparison = output / f"{architecture}_{dimensions}_cpu"
    # Hold weights exactly fixed for the arithmetic comparison. Independent
    # training trajectories are checked separately below, with their own scope.
    fixed_state = warm_state if backend == "cpu" else torch.load(comparison / "independent_warmup.pt", weights_only=True, map_location="cpu")
    checked.load_state_dict(fixed_state)
    calculation = accumulate_attention_block(checked, *inputs)
    gradients = {name: value.grad.detach().cpu().numpy().copy() for name, value in checked.named_parameters()}
    if any(not np.isfinite(value).all() for value in gradients.values()):
        raise ValueError("Every fixed-weight gradient must be finite")
    np.savez_compressed(folder / "fixed_weight_gradients.npz", **gradients)
    delta, query_record = predict_deltas(checked, matrix, clock, selected=selected)
    probability = corrected_posterior(logits, delta, prior, original)
    changed = matrix.copy()
    changed[777:, :-1] = 1e4
    future, _ = predict_deltas(checked, changed, clock, selected=selected)
    prefix, _ = predict_deltas(checked, matrix[:777], clock[:777], selected=selected[:777])
    chunks, _ = predict_deltas(checked, matrix, clock, selected=selected, chunk_rows=97)
    fresh, _ = predict_deltas(checked, matrix[3700:], clock[3700:])
    errors = {"future_prefix": float(np.max(np.abs(delta[:657] - future[:657]))),
        "shorter_prefix": float(np.max(np.abs(delta[:657] - prefix))),
        "chunking": float(np.max(np.abs(delta - chunks))),
        "gap_reset": float(np.max(np.abs(delta[3580:] - fresh)))}
    if max(errors.values()) > protocol["causal_delta_absolute_tolerance"]:
        raise ValueError(f"Causal/window numerical tolerance exceeded: {errors}")
    torch.save({name: value.detach().cpu() for name, value in checked.state_dict().items()}, folder / "checked_adapter.pt")
    restored = build_attention_network(dimensions, architecture).to(backend)
    restored.load_state_dict(torch.load(folder / "checked_adapter.pt", weights_only=True, map_location="cpu"))
    replay, _ = predict_deltas(restored, matrix, clock, selected=selected)
    np.testing.assert_array_equal(replay, delta)
    zero = build_attention_network(dimensions, architecture).to(backend)
    zero_delta, _ = predict_deltas(zero, matrix, clock, selected=selected)
    np.testing.assert_array_equal(corrected_posterior(logits, zero_delta, prior, original), original)
    np.savez_compressed(folder / "fixed_weight_forecasts.npz", probabilities=probability, delta=delta)
    agreement = {"fixed_weight_probability_absolute_error": 0., "fixed_weight_gradient_relative_l2_error": 0.,
        "fixed_weight_gradient_absolute_error": 0., "independent_warmup_probability_absolute_error": 0.}
    if backend == "mps":
        with np.load(comparison / "fixed_weight_forecasts.npz", allow_pickle=False) as saved:
            agreement["fixed_weight_probability_absolute_error"] = float(np.max(np.abs(saved["probabilities"] - probability)))
        with np.load(comparison / "independent_warmup_forecasts.npz", allow_pickle=False) as saved:
            agreement["independent_warmup_probability_absolute_error"] = float(np.max(np.abs(saved["probabilities"] - own_probability)))
        with np.load(comparison / "fixed_weight_gradients.npz", allow_pickle=False) as saved:
            left = np.concatenate([saved[name].ravel().astype(float) for name in sorted(gradients)])
        right = np.concatenate([gradients[name].ravel().astype(float) for name in sorted(gradients)])
        agreement["fixed_weight_gradient_relative_l2_error"] = float(np.linalg.norm(left - right) / max(np.linalg.norm(left), 1e-12))
        agreement["fixed_weight_gradient_absolute_error"] = float(np.max(np.abs(left - right)))
        if (max(agreement["fixed_weight_probability_absolute_error"], agreement["independent_warmup_probability_absolute_error"]) > protocol["cross_backend_probability_absolute_tolerance"]
            or agreement["fixed_weight_gradient_relative_l2_error"] > protocol["cross_backend_gradient_relative_l2_tolerance"]
            or agreement["fixed_weight_gradient_absolute_error"] > protocol["cross_backend_gradient_absolute_tolerance"]):
            raise ValueError(f"Frozen backend agreement tolerance exceeded: {agreement}")
    del restored, zero
    query_times = []
    for _ in range(protocol["timed_full_query_repetitions"]):
        synchronize(backend)
        tick = time.monotonic()
        repeated, _ = predict_deltas(checked, matrix, clock, selected=selected)
        repeated_p = corrected_posterior(logits, repeated, prior, original)
        synchronize(backend)
        query_times.append(time.monotonic() - tick)
        np.testing.assert_array_equal(repeated_p, probability)
    del checked
    timed_blocks = [update()[0] for _ in range(protocol["timed_blocks"])]
    stop.set()
    guard.join()
    median_block, median_query = float(np.median(timed_blocks)), float(np.median(query_times))
    result = {"architecture": architecture, "dimensions": dimensions, "backend": backend, "status": "passed",
        "python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
        "dependencies": {name: version(name) for name in ("numpy", "torch", "scikit-learn")},
        "torch_threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "mps_fallback_environment": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "config": dict(CONFIG), "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "warmup_updates": len(warmup), "warmup_update_seconds": warmup, "timed_block_seconds": timed_blocks,
        "median_block_seconds": median_block, "query_7070_seconds": query_times, "median_query_seconds": median_query,
        "arithmetic_probe": calculation, "backend_agreement": agreement, "causal_delta_max_absolute_errors": errors,
        "same_backend_checkpoint_replay_exact": True, "zero_correction_exact": True, "query_record": query_record,
        "projected_six_epoch_updates_and_validation_seconds": 6 * 169 * median_block + 12 * median_query,
        "projection_scope": "Block-time extrapolation only: 169 updates/day-grid times six epochs plus two validation queries/epoch. Excludes actual full-grid allocation, normalization, preparation, cache paging, checkpointing, assessment and replay. Neither measured full market runtime nor a memory guarantee.",
        "memory": readings, "seconds": time.monotonic() - begun, "market_model_fits": 0, "market_assessment_metrics_computed": False}
    write_json(folder / "result.json", result)


def run(protocol_path, output):
    protocol = checked_protocol(protocol_path)
    if output.exists():
        raise ValueError("Preserve every previous temporal-attention preflight attempt")
    output.mkdir(parents=True)
    write_json(output / "attempt.json", {"protocol_sha256": sha256_file(protocol_path), "market_model_fits": 0})
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTORCH_ENABLE_MPS_FALLBACK": "0"}
    records = []
    for architecture in protocol["architectures"]:
        for dimensions in protocol["dimensions"]:
            for backend in protocol["backends"]:
                name = f"{architecture}_{dimensions}_{backend}"
                command = [sys.executable, str(Path(__file__).resolve()), "--protocol", str(protocol_path), "--output", str(output),
                    "--worker", backend, "--dimensions", str(dimensions), "--architecture", architecture]
                failure, returncode = None, None
                with (output / f"{name}.log").open("w") as log:
                    try:
                        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=protocol["limits"]["worker_seconds"])
                        returncode = completed.returncode
                        if returncode:
                            failure = f"Worker exited {returncode}; preserve complete log and partial artifacts"
                    except subprocess.TimeoutExpired:
                        failure = "Worker exceeded its frozen deadline and was terminated"
                record = ({"architecture": architecture, "dimensions": dimensions, "backend": backend, "status": "failed", "failure": failure}
                    if failure else json.loads((output / name / "result.json").read_text()))
                record.update(returncode=returncode, log_sha256=sha256_file(output / f"{name}.log"))
                records.append(record)
                write_json(output / "progress.json", {"records": records, "market_model_fits": 0})
                print(f"temporal_attention_preflight={name} status={record['status']} seconds={record.get('seconds')}", flush=True)
    chosen, feasible = choose_backend(records)
    write_json(output / "summary.json", {"protocol_sha256": sha256_file(protocol_path), "records": records,
        "selected_backend": chosen, "feasible_backend_summed_median_block_seconds": feasible,
        "market_model_fits": 0, "market_assessment_metrics_computed": False,
        "artifact_hashes": {str(p.relative_to(output)): sha256_file(p) for p in output.rglob("*") if p.is_file()}})
    if chosen is None:
        raise ValueError("No common backend passed every frozen temporal-attention case")


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
