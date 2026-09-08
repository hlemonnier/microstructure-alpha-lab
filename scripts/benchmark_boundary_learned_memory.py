"""Score-free CPU/MPS checks of the registered learned-memory training block."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_learned_memory import ARCHITECTURES, accumulate_temporal_block, build_residual_network, corrected_posterior, predict_deltas
from run_boundary_confirmation import write_json

ROOT = Path(__file__).resolve().parents[1]


def synchronize(device):
    if device == "mps":
        import torch
        torch.mps.synchronize()


def synthetic_block(dimensions):
    rng = np.random.default_rng(20260908)
    x = rng.normal(size=(8, 512, dimensions)).astype(np.float32)
    z = rng.normal(0, .4, size=(8, 512, 3)).astype(np.float32)
    y = rng.choice([-1, 0, 1], size=(8, 512), p=[.2, .6, .2])
    valid = np.ones((8, 512), dtype=bool)
    valid[:, :120] = False
    valid[0, 231:242] = False
    valid[4, 401:418] = False
    selected = valid & (np.arange(512)[None] % 4 == 0)
    weights = np.zeros((8, 512), dtype=np.float32)
    assets = np.repeat([0, 1], 4)
    for asset in (0, 1):
        for label in (-1, 0, 1):
            mask = selected & (assets[:, None] == asset) & (y == label)
            weights[mask] = selected.sum() / (6 * mask.sum())
    x[~valid] = np.nan
    return x, z, y, weights, valid, selected


def worker(protocol_path, output, backend, dimensions, architecture):
    import torch

    protocol = json.loads(protocol_path.read_text())
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if backend == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Registered Mac GPU is unavailable in this execution context")
    begin = time.monotonic()
    folder = output / f"{architecture}_{dimensions}_{backend}"
    folder.mkdir()
    model = build_residual_network(dimensions, architecture)
    with torch.no_grad():
        model.head.weight.normal_(std=.2)
    initial = copy.deepcopy(model.state_dict())
    model.to(backend)
    inputs = synthetic_block(dimensions)
    _, calculation = accumulate_temporal_block(model, *inputs, None)
    gradients = {name: value.grad.detach().cpu().numpy().copy() for name, value in model.named_parameters()}
    if any(not np.isfinite(v).all() for v in gradients.values()):
        raise ValueError("All production-dimension gradients must be finite")
    np.savez_compressed(folder / "gradients.npz", **gradients)

    rng = np.random.default_rng(20260909)
    query = rng.normal(size=(2053, dimensions)).astype(np.float32)
    clock = np.arange(2053, dtype=np.int64) * 1000
    clock[1029:] += 1000
    logits = rng.normal(0, .4, size=(2053, 3))
    prior = np.array([.2, .6, .2])
    original = np.exp(logits) * prior
    original /= original.sum(axis=1, keepdims=True)
    delta, record = predict_deltas(model, query, clock)
    probabilities = corrected_posterior(logits, delta, prior, original)
    changed = query.copy()
    changed[777:] = 1e4
    future, _ = predict_deltas(model, changed, clock)
    shorter, _ = predict_deltas(model, query[:777], clock[:777])
    chunks, _ = predict_deltas(model, query, clock, chunk_rows=97)
    fresh, _ = predict_deltas(model, query[1029:], clock[1029:])
    errors = {"future_prefix": float(np.max(np.abs(delta[:777] - future[:777]))),
        "shorter_prefix": float(np.max(np.abs(delta[:777] - shorter))),
        "chunking": float(np.max(np.abs(delta - chunks))),
        "gap_reset": float(np.max(np.abs(delta[1029:] - fresh)))}
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, folder / "adapter.pt")
    restored = build_residual_network(dimensions, architecture).to(backend)
    restored.load_state_dict(torch.load(folder / "adapter.pt", weights_only=True, map_location="cpu"))
    reloaded, _ = predict_deltas(restored, query, clock)
    np.testing.assert_array_equal(delta, reloaded)
    zero = build_residual_network(dimensions, architecture).to(backend)
    zero_delta, _ = predict_deltas(zero, query, clock)
    np.testing.assert_array_equal(corrected_posterior(logits, zero_delta, prior, original), original)
    for error in errors.values():
        if error > protocol["causality_and_chunk_absolute_tolerance"]:
            raise ValueError(f"Registered numerical causality tolerance exceeded: {errors}")
    np.savez_compressed(folder / "forecasts.npz", probabilities=probabilities, delta=delta)
    agreement = {"probability_max_absolute_error": 0., "gradient_global_relative_l2_error": 0., "gradient_max_absolute_error": 0.}
    if backend == "mps":
        cpu = output / f"{architecture}_{dimensions}_cpu"
        with np.load(cpu / "forecasts.npz", allow_pickle=False) as saved:
            agreement["probability_max_absolute_error"] = float(np.max(np.abs(saved["probabilities"] - probabilities)))
        with np.load(cpu / "gradients.npz", allow_pickle=False) as saved:
            a = np.concatenate([saved[name].ravel().astype(float) for name in gradients])
        b = np.concatenate([gradients[name].ravel().astype(float) for name in gradients])
        agreement["gradient_global_relative_l2_error"] = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))
        agreement["gradient_max_absolute_error"] = float(np.max(np.abs(a - b)))
        if agreement["probability_max_absolute_error"] > protocol["cross_backend_probability_absolute_tolerance"] or agreement["gradient_global_relative_l2_error"] > protocol["cross_backend_gradient_relative_l2_tolerance"] or agreement["gradient_max_absolute_error"] > protocol["cross_backend_gradient_absolute_tolerance"]:
            raise ValueError(f"Registered backend agreement tolerance exceeded: {agreement}")
    del zero, restored
    model.load_state_dict(initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
    elapsed, state = [], None
    for iteration in range(protocol["warmup_blocks"] + protocol["timed_blocks"]):
        model.train()
        synchronize(backend)
        tick = time.monotonic()
        optimizer.zero_grad()
        state, _ = accumulate_temporal_block(model, *inputs, state, offset=iteration * 512)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        synchronize(backend)
        seconds = time.monotonic() - tick
        if seconds > protocol["maximum_block_seconds"]:
            raise ValueError("Production optimization block exceeded its registered resource bound")
        if iteration >= protocol["warmup_blocks"]:
            elapsed.append(seconds)
    if any(not torch.isfinite(v).all() for v in model.parameters()):
        raise ValueError("All warmed optimizer parameters must remain finite")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = int(peak if platform.system() == "Darwin" else 1024 * peak)
    gpu = int(torch.mps.driver_allocated_memory()) if backend == "mps" else 0
    if rss + gpu > protocol["maximum_rss_plus_driver_bytes"]:
        raise ValueError("Registered conservative process/GPU memory bound exceeded")
    result = {"backend": backend, "dimensions": dimensions, "architecture": architecture, "status": "passed",
        "trainable_parameters": sum(p.numel() for p in model.parameters()), "gradient_calculation": calculation,
        "causality_max_absolute_errors": errors, "zero_correction_exact": True, "checkpoint_reload_exact": True,
        "backend_agreement": agreement, "forward": record, "timed_block_seconds": elapsed,
        "median_block_seconds": float(np.median(elapsed)), "seconds": time.monotonic() - begin,
        "process_high_water_rss_bytes": rss, "end_gpu_driver_allocated_bytes": gpu,
        "memory_limit_scope": "Process high-water RSS plus end-of-case GPU driver allocation; conservative sum may double-count unified allocations and is not a sampled GPU peak.",
        "market_model_fits": 0, "assessment_metrics_computed": False}
    write_json(folder / "result.json", result)


def run(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if protocol["architectures"] != list(ARCHITECTURES) or protocol["dimensions"] != [221, 400] or protocol["backends"] != ["cpu", "mps"]:
        raise ValueError("Only the registered twelve production-dimension cases may run")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen synthetic benchmark input changed: {path}")
    if output.exists():
        raise ValueError("Preserve any previous learned-memory benchmark attempt")
    output.mkdir(parents=True)
    write_json(output / "attempt.json", {"protocol_sha256": sha256_file(protocol_path), "market_model_fits": 0})
    records = []
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTORCH_ENABLE_MPS_FALLBACK": "0"}
    for architecture in protocol["architectures"]:
        for dimensions in protocol["dimensions"]:
            for backend in protocol["backends"]:
                name = f"{architecture}_{dimensions}_{backend}"
                command = [sys.executable, str(Path(__file__).resolve()), "--protocol", str(protocol_path), "--output", str(output),
                    "--worker", backend, "--dimensions", str(dimensions), "--architecture", architecture]
                failure = None
                with (output / f"{name}.log").open("w") as stream:
                    try:
                        result = subprocess.run(command, env=environment, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=protocol["maximum_case_seconds"])
                        if result.returncode:
                            failure = f"Worker exited with code {result.returncode}; preserve its complete log."
                    except subprocess.TimeoutExpired:
                        failure = "Worker exceeded the registered timeout and was terminated."
                record = {"backend": backend, "dimensions": dimensions, "architecture": architecture, "status": "failed", "failure": failure} if failure else json.loads((output / name / "result.json").read_text())
                records.append(record)
                write_json(output / "progress.json", {"records": records, "market_model_fits": 0})
                print(f"learned_memory_preflight={name} status={record['status']} median_block_seconds={record.get('median_block_seconds')}", flush=True)
    feasible = {backend: sum(r["median_block_seconds"] for r in records if r["backend"] == backend)
        for backend in protocol["backends"] if all(r["status"] == "passed" for r in records if r["backend"] == backend)}
    chosen = min(feasible, key=feasible.get) if feasible else None
    write_json(output / "summary.json", {"protocol_sha256": sha256_file(protocol_path), "records": records,
        "feasible_backend_summed_median_block_seconds": feasible, "selected_backend": chosen,
        "market_model_fits": 0, "assessment_metrics_computed": False,
        "artifact_hashes": {str(p.relative_to(output)): sha256_file(p) for p in output.rglob("*") if p.is_file()}})
    if chosen is None:
        raise ValueError("No common backend passed all registered checks")


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
