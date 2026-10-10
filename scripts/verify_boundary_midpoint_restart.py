"""Verify a completed midpoint restart without decoding assessment scores.

The frozen driver and assessor verify the model inputs. This additive gate checks
their identities, completed output inventories, worker execution, exact replay
of the eight original workers, and preservation of the interrupted attempt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ERROR_FIELDS = {"full_query_prefix", "future_prefix", "individual_query", "shorter_prefix"}
ARRAY_FIELDS = {"probabilities", "train_priors"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def contains(parent, child):
    return child == parent or parent in child.parents


def relative_path(base, name):
    require(isinstance(name, str) and bool(name), "Nonempty relative path required")
    relative = Path(name)
    require(not relative.is_absolute(), f"Absolute inventory or registration path: {name}")
    return (base / relative).resolve()


def inventory_files(folder, excluded=None):
    require(folder.is_dir(), f"Missing attempt or fold directory: {folder}")
    return {str(path.relative_to(folder)): path for path in folder.rglob("*") if path.is_file() and path != excluded}


def verify_fold(folder, day, identity, protocol, receipt):
    marker = folder / "completed.json"
    completed = read_json(marker)
    require(completed["identity"] == identity, f"Wrong completed identity: {day}")
    require(completed["date"] == day, f"Wrong completed date: {day}")
    require(completed["model_fits"] == len(protocol["model_specs"]), f"Incomplete fold: {day}")
    expected = completed["artifact_hashes"]
    require(isinstance(expected, dict), f"Invalid completed inventory: {day}")
    actual = inventory_files(folder, marker)
    require(set(actual) == set(expected), f"Completed artifact file set changed: {day}")
    for name, digest in expected.items():
        path = relative_path(folder, name)
        require(contains(folder, path), f"Completed artifact escapes fold: {day}/{name}")
        require(sha(path) == digest, f"Completed artifact hash changed: {day}/{name}")
    receipt["completed_folds"].append(
        {
            "date": day,
            "completed_sha256": sha(marker),
            "artifact_hashes": expected,
            "artifact_count": len(expected),
        }
    )


def finite_number(value, message):
    require(isinstance(value, (int, float)) and not isinstance(value, bool), message)
    require(math.isfinite(value), message)
    return value


def verify_workers(output, protocol, receipt):
    symbols = protocol["symbols"]
    specifications = protocol["model_specs"]
    limits = protocol["limits"]
    for day in protocol["dates"]:
        folder = output / "dates" / day
        workers = folder / "workers"
        require(workers.is_dir(), f"Missing workers: {day}")
        require({p.name for p in workers.iterdir()} == set(specifications), f"Worker set changed: {day}")
        require(
            {p.name for p in folder.glob("*.supervision.json")}
            == {f"{name}.supervision.json" for name in specifications},
            f"Supervision set changed: {day}",
        )
        for name, specification in specifications.items():
            key = f"{day}/{name}"
            worker_path = workers / name / "worker.json"
            worker = read_json(worker_path)
            supervision_path = folder / f"{name}.supervision.json"
            supervision = read_json(supervision_path)
            log = folder / f"{name}.log"
            require(worker["model"] == name and worker["specification"] == specification, f"Wrong worker: {key}")
            require(worker["assessment_labels_decoded"] is False, f"Worker decoded assessment labels: {key}")
            require(
                supervision["date"] == day
                and supervision["model"] == name
                and supervision["returncode"] == 0
                and supervision["failure"] is None,
                f"Unsuccessful or mismatched supervision: {key}",
            )
            require(sha(log) == supervision["log_sha256"], f"Worker log changed: {key}")
            memory = worker["memory"]
            rss = finite_number(memory["maximum_sampled_rss_bytes"], f"Invalid sampled RSS: {key}")
            total = finite_number(
                memory["maximum_sampled_simultaneous_sum_bytes"], f"Invalid sampled memory sum: {key}"
            )
            gpu = finite_number(memory["maximum_sampled_gpu_driver_bytes"], f"Invalid GPU memory: {key}")
            require(0 < rss <= limits["worker_rss_bytes"], f"Sampled RSS outside frozen ceiling: {key}")
            require(rss == total and total <= limits["rss_plus_driver_bytes"], f"Sampled CPU memory sum invalid: {key}")
            require(gpu == 0, f"Nonzero GPU-driver memory: {key}")
            require(set(worker["checks"]) == set(symbols), f"Missing asset checks: {key}")
            for symbol in symbols:
                check = worker["checks"][symbol]
                require(
                    check["checkpoint_probabilities_exact"] is True and check["training_priors_exact"] is True,
                    f"Checkpoint or prior check failed: {key}/{symbol}",
                )
                errors = check["errors"]
                require(set(errors) == ERROR_FIELDS, f"Incomplete query checks: {key}/{symbol}")
                for field, value in errors.items():
                    tolerance = protocol[
                        "future_prefix_absolute_tolerance"
                        if field == "future_prefix"
                        else "query_partition_absolute_tolerance"
                    ]
                    error = finite_number(value, f"Invalid query error: {key}/{symbol}/{field}")
                    require(0 <= error <= tolerance, f"Query causality check failed: {key}/{symbol}/{field}")
            receipt["workers"].append(
                {
                    "date": day,
                    "model": name,
                    "worker_sha256": sha(worker_path),
                    "supervision_sha256": sha(supervision_path),
                    "log_sha256": supervision["log_sha256"],
                    "memory": memory,
                    "checks": worker["checks"],
                }
            )
            receipt["new_workers_verified"] += 1


def verify_panels(output, protocol, receipt):
    names = [*protocol["baselines"], *protocol["model_specs"], *protocol["blends"]]
    require(len(names) == len(set(names)) == 48, "Wrong fixed prediction-source family")
    for day in protocol["dates"]:
        folder = output / "dates" / day / "predictions"
        expected = {
            f"{symbol}_{name}_{policy}.npz"
            for symbol in protocol["symbols"]
            for name in names
            for policy in ("registered", "forecast_3600")
        }
        require(folder.is_dir(), f"Missing prediction panels: {day}")
        require({p.name for p in folder.iterdir()} == expected, f"Prediction-panel set changed: {day}")
        require(all((folder / name).is_file() for name in expected), f"Prediction panel is not a file: {day}")
        receipt["prediction_panels_verified"] += len(expected)
    require(receipt["prediction_panels_verified"] == 576, "Incomplete prediction panels")


def verify_old_attempt(root, original, pause, registration, receipt):
    expected = pause["local_artifacts"]
    require(len(expected) == registration["preserved_artifacts_verified"] == 173, "Wrong preserved-artifact count")
    observed = {str(path.relative_to(root)): path for path in inventory_files(original).values()}
    require(set(observed) == set(expected), "Original attempt file set changed")
    for name, record in expected.items():
        path = relative_path(root, name)
        require(contains(original, path), f"Preserved artifact escapes original attempt: {name}")
        require(path.stat().st_size == record["bytes"], f"Preserved artifact length changed: {name}")
        require(sha(path) == record["sha256"], f"Preserved artifact hash changed: {name}")
    receipt["preserved_artifacts_verified"] = len(expected)
    receipt["preserved_artifact_bytes"] = sum(record["bytes"] for record in expected.values())
    receipt["preserved_artifacts"] = expected


def load_worker_arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        require(len(saved.files) == 2 and set(saved.files) == ARRAY_FIELDS, f"Worker array keys changed: {path}")
        arrays = {field: saved[field].copy() for field in ARRAY_FIELDS}
    for field, shape in (("probabilities", (7070, 3)), ("train_priors", (3,))):
        array = arrays[field]
        require(
            array.shape == shape and np.issubdtype(array.dtype, np.floating),
            f"Worker array shape/type changed: {path}/{field}",
        )
        require(np.isfinite(array).all(), f"Nonfinite worker array: {path}/{field}")
    return arrays


def array_sha(array):
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)}, sort_keys=True).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def verify_replay(root, original, output, protocol, pause, registration, receipt):
    previous = pause["completed_workers"]
    require(
        len(previous) == pause["completed_fits"] == registration["prior_completed_fits"] == 8,
        "Wrong replay-worker count",
    )
    pairs = {(worker["date"], worker["model"]) for worker in previous}
    require(len(pairs) == 8, "Duplicate replay workers")
    for previous_worker in previous:
        day, name = previous_worker["date"], previous_worker["model"]
        require(day in protocol["dates"] and name in protocol["model_specs"], "Unregistered replay worker")
        old_worker = relative_path(root, previous_worker["worker_path"])
        expected_worker = original / "dates" / day / "workers" / name / "worker.json"
        require(old_worker == expected_worker.resolve(), "Replay worker path differs from original inventory")
        for symbol in protocol["symbols"]:
            old_path = old_worker.parent / f"{symbol}_probabilities.npz"
            new_path = output / "dates" / day / "workers" / name / old_path.name
            old_arrays, new_arrays = load_worker_arrays(old_path), load_worker_arrays(new_path)
            arrays = {}
            for field in sorted(ARRAY_FIELDS):
                before, after = old_arrays[field], new_arrays[field]
                require(
                    before.dtype == after.dtype and before.shape == after.shape,
                    f"Replay dtype/shape changed: {day}/{name}/{symbol}/{field}",
                )
                require(np.array_equal(before, after), f"Replay array changed: {day}/{name}/{symbol}/{field}")
                arrays[field] = {
                    "dtype": before.dtype.str,
                    "shape": list(before.shape),
                    "exact_equal": True,
                    "original_array_sha256": array_sha(before),
                    "new_array_sha256": array_sha(after),
                }
                receipt["replayed_arrays"] += 1
            receipt["replay_comparisons"].append(
                {
                    "date": day,
                    "model": name,
                    "symbol": symbol,
                    "original_file_sha256": sha(old_path),
                    "new_file_sha256": sha(new_path),
                    "arrays": arrays,
                }
            )
    require(
        len(receipt["replay_comparisons"]) == 16 and receipt["replayed_arrays"] == 32,
        "Incomplete original-worker replay",
    )


def verify(root, registration, execution, original, output, receipt):
    require(contains(root, execution) and execution.is_dir(), "Execution root must remain inside the checkout")
    require(contains(root, original), "Original attempt escapes checkout")
    require(contains(execution, output), "New output escapes registered execution root")
    require(not contains(original, output) and not contains(output, original), "Attempts alias or contain each other")
    protocol_path = relative_path(execution, registration["protocol"])
    require(contains(execution, protocol_path), "Protocol escapes execution root")
    require(sha(protocol_path) == registration["protocol_sha256"], "Frozen protocol hash changed")
    pause_path = relative_path(root, registration["pause_evidence"])
    require(contains(root, pause_path), "Pause evidence escapes checkout")
    require(sha(pause_path) == registration["pause_evidence_sha256"], "Pause evidence hash changed")
    protocol, pause = read_json(protocol_path), read_json(pause_path)
    registered = registration.get("registered_fits", registration.get("fresh_fits"))
    expected_workers = len(protocol["dates"]) * len(protocol["model_specs"])
    require(registered == expected_workers == protocol["trainable_model_fits"] == 48, "Wrong registered-fit count")
    require(protocol["post_fit_panels"] == 576, "Wrong registered-panel count")
    require(len(protocol["dates"]) == len(set(protocol["dates"])) == 3, "Wrong development-date family")
    require(len(protocol["symbols"]) == len(set(protocol["symbols"])) == 2, "Wrong registered assets")
    if "registered_fits" in registration:
        require(
            registration["fresh_fits"] + registration.get("reused_fits", 0) == registered,
            "Fresh/reused fit accounting changed",
        )
    identity = read_json(output / "frozen_screen.json")
    require(identity == read_json(original / "frozen_screen.json"), "Original/new frozen identities differ")
    require(identity["protocol_sha256"] == registration["protocol_sha256"], "Wrong frozen identity protocol")
    require(identity["input_hashes"] == protocol["input_hashes"], "Wrong frozen input identity")
    require(len(identity["input_hashes"]) == registration["input_hashes_verified"], "Wrong frozen-input count")
    require(
        {key: identity[key] for key in ("python", "dependencies")} == registration["runtime"],
        "Runtime identity changed",
    )
    summary = output / "summary.json"
    require(summary.is_file() and summary.stat().st_size > 0, "Missing final summary")
    receipt.update(
        {
            "protocol_sha256": sha(protocol_path),
            "pause_evidence_sha256": sha(pause_path),
            "summary_sha256": sha(summary),
            "identity": identity,
            "registered_fits": registered,
            "fresh_fits": registration["fresh_fits"],
            "reused_fits": registration.get("reused_fits", 0),
        }
    )
    dates_folder = output / "dates"
    require(dates_folder.is_dir(), "Missing completed date directories")
    require({path.name for path in dates_folder.iterdir()} == set(protocol["dates"]), "Completed date set changed")
    require(all(path.is_dir() for path in dates_folder.iterdir()), "Completed date entry is not a directory")
    verify_workers(output, protocol, receipt)
    verify_panels(output, protocol, receipt)
    for day in protocol["dates"]:
        verify_fold(output / "dates" / day, day, identity, protocol, receipt)
    verify_old_attempt(root, original, pause, registration, receipt)
    verify_replay(root, original, output, protocol, pause, registration, receipt)
    require(receipt["new_workers_verified"] == 48, "Incomplete completed-worker verification")
    receipt["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    evidence = args.evidence if args.evidence.is_absolute() else root / args.evidence
    receipt = {
        "schema_version": 1,
        "status": "failed",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "assessment_scores_decoded": False,
        "assessment_labels_decoded": False,
        "model_inputs_independently_rehashed": False,
        "model_input_scope": "Pinned mapping identity only; original driver and assessor verify model input hashes.",
        "memory_scope": "Sampled RSS checks are not a proof of a continuous peak bound.",
        "new_workers_verified": 0,
        "prediction_panels_verified": 0,
        "replayed_arrays": 0,
        "workers": [],
        "completed_folds": [],
        "replay_comparisons": [],
        "verifier_sha256": sha(Path(__file__)),
        "root": str(root),
    }
    receipt_allowed = False
    try:
        require(not os.path.lexists(evidence), "Preserve the existing evidence receipt")
        registration_path = args.registration if args.registration.is_absolute() else root / args.registration
        registration = read_json(registration_path)
        execution = relative_path(root, registration["execution_root_relative_to_checkout"])
        original = relative_path(root, registration["original_attempt"])
        output = relative_path(execution, registration["output_relative_to_execution_root"])
        receipt.update(
            {
                "registration": str(registration_path.resolve()),
                "registration_sha256": sha(registration_path),
                "execution_root": str(execution),
                "original_attempt": str(original),
                "new_attempt": str(output),
            }
        )
        require(
            not contains(original, evidence.resolve()) and not contains(output, evidence.resolve()),
            "Evidence receipt must remain outside both attempts",
        )
        receipt_allowed = True
        verify(root, registration, execution, original, output, receipt)
    except Exception as error:
        receipt.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
    if receipt_allowed:
        try:
            serialized = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
            evidence.parent.mkdir(parents=True, exist_ok=True)
            with evidence.open("x") as destination:
                destination.write(serialized)
            result = {
                "status": receipt["status"],
                "evidence": str(evidence.resolve()),
                "evidence_sha256": sha(evidence),
            }
        except Exception as error:
            receipt.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
            result = receipt
    else:
        result = receipt
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
