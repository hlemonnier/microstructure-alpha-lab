"""Bounded synthetic TabICL compatibility, causality and CPU measurement."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from lob_forge.binance_vision import sha256_file
from run_boundary_confirmation import write_json

ROOT = Path(__file__).resolve().parents[1]


def worker(protocol, case, output):
    import numpy as np
    import torch
    from tabicl import TabICLClassifier

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(20260908)
    rows, features = case["context_rows"], case["features"]
    x = rng.normal(size=(rows, features))
    y = np.arange(rows) % 3
    x[:, 0] += y - 1
    query = rng.normal(size=(64, features))
    settings = {"n_estimators": 1, "norm_methods": ["none"], "batch_size": 1,
        "model_path": str(ROOT / protocol["checkpoint"]), "allow_auto_download": False,
        "checkpoint_version": "tabicl-classifier-v2-20260212.ckpt", "device": "cpu", "use_amp": False,
        "use_fa3": False, "offload_mode": "cpu", "random_state": 20260908, "n_jobs": 2, "kv_cache": True}
    start = time.monotonic()
    model = TabICLClassifier(**settings).fit(x, y)
    fit_seconds = time.monotonic() - start
    start = time.monotonic()
    p = model.predict_proba(query)
    predict_seconds = time.monotonic() - start
    if p.shape != (64, 3) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("Finite normalized three-class forecasts required")
    changed = query.copy()
    changed[16:] = 100 * rng.normal(size=changed[16:].shape)
    p_changed = model.predict_proba(changed)
    # Keeping query shape identical isolates future-query dependence from
    # numerical differences caused by changing matrix multiplication shapes.
    np.testing.assert_allclose(p_changed[:16], p[:16], rtol=0, atol=2e-6)
    prefix = model.predict_proba(query[:16])
    np.testing.assert_allclose(prefix, p[:16], rtol=0, atol=2e-5)
    if case.get("verify_cache"):
        uncached = TabICLClassifier(**{**settings, "kv_cache": False}).fit(x, y)
        original = uncached.predict_proba(query)
        np.testing.assert_allclose(original, p, rtol=0, atol=2e-5)
        cache_error = float(np.max(np.abs(original - p)))
    else:
        cache_error = None
    write_json(output, {"case": case, "fit_seconds": fit_seconds, "predict_64_seconds": predict_seconds,
        "later_query_change_max_absolute_error": float(np.max(np.abs(p_changed[:16] - p[:16]))),
        "shorter_query_batch_max_absolute_error": float(np.max(np.abs(prefix - p[:16]))),
        "cached_uncached_max_absolute_error": cache_error, "probability_checks_passed": True,
        "synthetic_only": True, "market_model_procedures": 0})


def run(protocol_path, output):
    import psutil

    protocol = json.loads(protocol_path.read_text())
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen TabICL preflight input changed: {path}")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, case in enumerate(protocol["cases"]):
        destination = output / f"case_{index}.json"
        log = output / f"case_{index}.log"
        if destination.exists() or log.exists():
            raise ValueError("Preserve all earlier synthetic preflight attempts")
        command = [sys.executable, str(Path(__file__).resolve()), "--protocol", str(protocol_path),
            "--worker", str(index), "--output", str(destination)]
        start, peak, reason = time.monotonic(), 0, None
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"}
        with log.open("w") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
            monitored = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    rss = monitored.memory_info().rss + sum(child.memory_info().rss for child in monitored.children(recursive=True))
                    peak = max(peak, rss)
                except psutil.NoSuchProcess:
                    pass
                if peak > protocol["per_case_rss_limit_bytes"]:
                    reason = "registered_rss_limit"
                elif time.monotonic() - start > protocol["per_case_seconds_limit"]:
                    reason = "registered_time_limit"
                if reason:
                    process.kill()
                    break
                time.sleep(.2)
            code = process.wait()
        record = {"case": case, "returncode": code, "stop_reason": reason, "seconds": time.monotonic() - start,
            "peak_observed_process_tree_rss_bytes": peak, "log_sha256": sha256_file(log)}
        if destination.exists():
            record["result"] = json.loads(destination.read_text())
            record["result_sha256"] = sha256_file(destination)
        records.append(record)
        write_json(output / "progress.json", {"completed_cases": len(records), "record": record})
        print(f"tabicl_synthetic_case={index + 1}/{len(protocol['cases'])} status={code} seconds={record['seconds']:.1f}", flush=True)
    write_json(output / "summary.json", {"protocol_sha256": sha256_file(protocol_path), "records": records,
        "evidence_status": "synthetic_operational_and_causality_checks_only", "market_model_procedures": 0,
        "interpretation": "Synthetic CPU feasibility and conditional query invariance only. No market observations, targets or accuracy scores were read. RSS is polled and may miss shorter peaks; time and memory stops are operational limits, not empirical accuracy failures."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    if args.worker is None:
        run(args.protocol.resolve(), args.output.resolve())
    else:
        protocol = json.loads(args.protocol.read_text())
        worker(protocol, protocol["cases"][args.worker], args.output.resolve())
