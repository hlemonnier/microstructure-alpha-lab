"""Exercise the restart-verification CLI on synthetic complete and corrupt attempts.

No market data, target labels or learners are used. Written before the verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DAYS = ("2023-06-03", "2023-06-07", "2023-06-11")
SYMBOLS = ("BTCUSDT", "ETHUSDT")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def rehash_fold(folder):
    marker = folder / "completed.json"
    record = json.loads(marker.read_text())
    record["artifact_hashes"] = {
        str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob("*")) if p.is_file() and p != marker
    }
    write(marker, record)


def fixture(root):
    original = root / "results/old"
    snapshot = root / "snapshot"
    fresh = snapshot / "results/fresh"
    models = {f"worker_{i:02}": {"kind": "synthetic", "index": i} for i in range(16)}
    baselines = [f"baseline_{i:02}" for i in range(8)]
    blends = {f"blend_{i:02}": ["worker_00", "worker_01"] for i in range(24)}
    protocol = {
        "dates": list(DAYS),
        "symbols": list(SYMBOLS),
        "model_specs": models,
        "baselines": baselines,
        "blends": blends,
        "input_hashes": {"synthetic": "fixture-only"},
        "trainable_model_fits": 48,
        "post_fit_panels": 576,
        "limits": {"worker_rss_bytes": 8589934592, "rss_plus_driver_bytes": 8589934592},
        "future_prefix_absolute_tolerance": 0,
        "query_partition_absolute_tolerance": 2e-6,
    }
    protocol_path = snapshot / "docs/protocol.json"
    write(protocol_path, protocol)
    identity = {
        "protocol_sha256": sha(protocol_path),
        "input_hashes": protocol["input_hashes"],
        "python": "synthetic",
        "dependencies": {"numpy": np.__version__},
    }
    write(original / "frozen_screen.json", identity)
    write(fresh / "frozen_screen.json", identity)
    old_workers = []
    for day in DAYS:
        folder = fresh / "dates" / day
        for index, (name, specification) in enumerate(models.items()):
            worker = folder / "workers" / name
            result = {
                "model": name,
                "specification": specification,
                "assessment_labels_decoded": False,
                "columns": ["synthetic_feature"],
                "train_rows": {s: 12 for s in SYMBOLS},
                "memory": {
                    "maximum_sampled_rss_bytes": 64000,
                    "maximum_sampled_gpu_driver_bytes": 0,
                    "maximum_sampled_simultaneous_sum_bytes": 64000,
                },
                "checks": {
                    s: {
                        "checkpoint_probabilities_exact": True,
                        "training_priors_exact": True,
                        "errors": {
                            "full_query_prefix": 0,
                            "future_prefix": 0,
                            "individual_query": 0,
                            "shorter_prefix": 0,
                        },
                    }
                    for s in SYMBOLS
                },
            }
            write(worker / "worker.json", result)
            for symbol in SYMBOLS:
                np.savez_compressed(
                    worker / f"{symbol}_probabilities.npz",
                    probabilities=np.full((7070, 3), 1 / 3),
                    train_priors=np.full(3, 1 / 3),
                )
            log = folder / f"{name}.log"
            log.write_text("Synthetic worker completed.\n")
            supervision = {
                "date": day,
                "model": name,
                "returncode": 0,
                "failure": None,
                "seconds": 1,
                "log_sha256": sha(log),
            }
            write(folder / f"{name}.supervision.json", supervision)
            if day == DAYS[0] and index < 8:
                old_worker = original / "dates" / day / "workers" / name
                shutil.copytree(worker, old_worker)
                old_workers.append(
                    {"date": day, "model": name, "worker_path": str((old_worker / "worker.json").relative_to(root))}
                )
        predictions = folder / "predictions"
        predictions.mkdir()
        for symbol in SYMBOLS:
            for name in (*baselines, *models, *blends):
                for policy in ("registered", "forecast_3600"):
                    (predictions / f"{symbol}_{name}_{policy}.npz").write_bytes(b"Opaque synthetic panel.\n")
        write(folder / "completed.json", {"identity": identity, "date": day, "model_fits": 16})
        rehash_fold(folder)
    # Intentionally unparsable: the compatibility verifier must not inspect scores.
    (fresh / "summary.json").write_bytes(b"Opaque synthetic summary; not JSON.\n")
    old_files = [p for p in original.rglob("*") if p.is_file()]
    for i in range(173 - len(old_files)):
        (original / f"preserved_{i:03}.txt").write_text(f"Preserved synthetic artifact {i}.\n")
    pause = {
        "completed_workers": old_workers,
        "completed_fits": 8,
        "planned_fits": 48,
        "local_artifacts": {
            str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": sha(p)}
            for p in original.rglob("*")
            if p.is_file()
        },
    }
    write(root / "docs/pause.json", pause)
    registration = {
        "protocol": "docs/protocol.json",
        "protocol_sha256": sha(protocol_path),
        "execution_root_relative_to_checkout": "snapshot",
        "original_attempt": "results/old",
        "output_relative_to_execution_root": "results/fresh",
        "pause_evidence": "docs/pause.json",
        "pause_evidence_sha256": sha(root / "docs/pause.json"),
        "input_hashes_verified": 1,
        "fresh_fits": 48,
        "prior_completed_fits": 8,
        "prior_interrupted_fits": 1,
        "preserved_artifacts_verified": 173,
        "runtime": {k: identity[k] for k in ("python", "dependencies")},
    }
    write(root / "docs/registration.json", registration)


def mutate(root, case):
    fresh = root / "snapshot/results/fresh"
    old = root / "results/old"
    folder = fresh / "dates" / DAYS[0]
    worker = folder / "workers/worker_00"
    record_path = worker / "worker.json"
    if case == "missing_summary":
        (fresh / "summary.json").unlink()
    elif case == "wrong_identity":
        path = fresh / "frozen_screen.json"
        d = json.loads(path.read_text())
        d["python"] = "drift"
        write(path, d)
    elif case in {"protocol_drift", "pause_drift"}:
        path = root / ("snapshot/docs/protocol.json" if case == "protocol_drift" else "docs/pause.json")
        path.write_text(path.read_text() + " ")
    elif case == "aliased_attempts":
        shutil.rmtree(fresh)
        fresh.symlink_to(old, target_is_directory=True)
    elif case == "nested_attempts":
        d = json.loads((root / "docs/registration.json").read_text())
        d["output_relative_to_execution_root"] = "../results/old/nested"
        (old / "nested").mkdir()
        write(root / "docs/registration.json", d)
    elif case == "missing_worker":
        record_path.unlink()
    elif case == "extra_worker":
        write(folder / "workers/extra/worker.json", {})
    elif case == "missing_panel":
        next((folder / "predictions").iterdir()).unlink()
    elif case == "extra_panel":
        (folder / "predictions/extra.npz").write_bytes(b"extra")
    elif case == "missing_fold":
        (folder / "completed.json").unlink()
    elif case == "failed_supervision":
        p = folder / "worker_00.supervision.json"
        d = json.loads(p.read_text())
        d["returncode"] = 1
        write(p, d)
        rehash_fold(folder)
    elif case == "changed_log":
        (folder / "worker_00.log").write_text("Changed log.\n")
        rehash_fold(folder)
    elif case in {"zero_rss", "excess_rss", "gpu_memory", "checkpoint_check", "label_access", "future_dependence"}:
        d = json.loads(record_path.read_text())
        if case == "zero_rss":
            d["memory"]["maximum_sampled_rss_bytes"] = 0
        elif case == "excess_rss":
            d["memory"]["maximum_sampled_rss_bytes"] = 8589934593
        elif case == "gpu_memory":
            d["memory"]["maximum_sampled_gpu_driver_bytes"] = 1
        elif case == "checkpoint_check":
            d["checks"]["BTCUSDT"]["checkpoint_probabilities_exact"] = False
        elif case == "label_access":
            d["assessment_labels_decoded"] = True
        else:
            d["checks"]["BTCUSDT"]["errors"]["future_prefix"] = 0.01
        write(record_path, d)
        rehash_fold(folder)
    elif case in {"old_missing", "old_extra", "old_changed"}:
        p = old / "preserved_000.txt"
        if case == "old_missing":
            p.unlink()
        elif case == "old_extra":
            (old / "extra.txt").write_text("extra")
        else:
            p.write_text("X" * p.stat().st_size)
    elif case in {"probability_drift", "prior_drift", "dtype_drift", "nonfinite", "array_keys"}:
        path = worker / "BTCUSDT_probabilities.npz"
        with np.load(path, allow_pickle=False) as saved:
            values = {k: saved[k].copy() for k in saved.files}
        if case == "probability_drift":
            values["probabilities"][0] += [0.001, -0.001, 0]
        elif case == "prior_drift":
            values["train_priors"] += [0.001, -0.001, 0]
        elif case == "dtype_drift":
            values["probabilities"] = values["probabilities"].astype(np.float32)
        elif case == "nonfinite":
            values["probabilities"][0, 0] = np.nan
        else:
            values["unexpected"] = np.ones(1)
        np.savez_compressed(path, **values)
        rehash_fold(folder)
    else:
        raise ValueError(case)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    base = output / "synthetic_base"
    fixture(base)
    cases = (
        "pass",
        "missing_summary",
        "wrong_identity",
        "protocol_drift",
        "pause_drift",
        "aliased_attempts",
        "nested_attempts",
        "missing_worker",
        "extra_worker",
        "missing_panel",
        "extra_panel",
        "missing_fold",
        "failed_supervision",
        "changed_log",
        "zero_rss",
        "excess_rss",
        "gpu_memory",
        "checkpoint_check",
        "label_access",
        "future_dependence",
        "old_missing",
        "old_extra",
        "old_changed",
        "probability_drift",
        "prior_drift",
        "dtype_drift",
        "nonfinite",
        "array_keys",
        "existing_receipt",
    )
    records = []
    for case in cases:
        root = output / "cases" / case
        shutil.copytree(base, root)
        evidence = root / "receipt.json"
        if case == "existing_receipt":
            evidence.write_text("Preserve this receipt.\n")
        elif case != "pass":
            mutate(root, case)
        command = [
            sys.executable,
            str(ROOT / "scripts/verify_boundary_midpoint_restart.py"),
            "--root",
            str(root),
            "--registration",
            "docs/registration.json",
            "--evidence",
            str(evidence),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=45, check=False)
        (root / "stdout.txt").write_text(result.stdout)
        (root / "stderr.txt").write_text(result.stderr)
        passed = result.returncode == (0 if case == "pass" else 1)
        if case == "existing_receipt":
            passed = passed and evidence.read_text() == "Preserve this receipt.\n"
        else:
            receipt = json.loads(evidence.read_text()) if evidence.exists() else {}
            passed = passed and receipt.get("status") == ("passed" if case == "pass" else "failed")
            if case == "pass":
                passed = passed and receipt["assessment_scores_decoded"] is False
                passed = passed and receipt["replayed_arrays"] == 32 and receipt["new_workers_verified"] == 48
        records.append(
            {
                "case": case,
                "command": command,
                "returncode": result.returncode,
                "passed": passed,
                "receipt_sha256": sha(evidence) if evidence.exists() else None,
                "stdout_sha256": sha(root / "stdout.txt"),
                "stderr_sha256": sha(root / "stderr.txt"),
            }
        )
        print(f"restart_verification_e2e={case} passed={passed}", flush=True)
    report = {
        "status": "passed" if all(r["passed"] for r in records) else "failed",
        "synthetic_only": True,
        "model_fits": 0,
        "market_data_read": False,
        "verifier_sha256": sha(ROOT / "scripts/verify_boundary_midpoint_restart.py"),
        "e2e_runner_sha256": sha(Path(__file__)),
        "cases": records,
    }
    write(output / "evidence.json", report)
    print(
        json.dumps(
            {"status": report["status"], "cases": len(records), "evidence_sha256": sha(output / "evidence.json")}
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output.resolve()))
