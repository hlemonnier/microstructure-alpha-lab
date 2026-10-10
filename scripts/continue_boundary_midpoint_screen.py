"""Continue a registered midpoint family using immutable copies of completed work."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_inventory(root, record):
    folder = root / record["attempt_directory"]
    actual = {str(p.relative_to(root)) for p in folder.rglob("*") if p.is_file()}
    require(actual == set(record["local_artifacts"]), "Preserved attempt file set changed")
    for name, entry in record["local_artifacts"].items():
        path = root / name
        require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], name)


def verify_inputs(source, protocol):
    for name, digest in protocol["input_hashes"].items():
        require(sha(source / name) == digest, f"Frozen input changed: {name}")


def worker_record(module, protocol, folder, day, name):
    worker = read(folder / "workers" / name / "worker.json")
    supervision = read(folder / f"{name}.supervision.json")
    require(supervision["date"] == day and supervision["model"] == name, "Wrong supervision identity")
    require(supervision["returncode"] == 0 and supervision["failure"] is None, "Unsuccessful supervision")
    require(sha(folder / f"{name}.log") == supervision["log_sha256"], "Worker log changed")
    require(worker["model"] == name and worker["specification"] == module.MODEL_SPECS[name], "Wrong model")
    require(worker["assessment_labels_decoded"] is False, "Assessment labels exposed to worker")
    metadata = read(folder / "prepared" / "metadata.json")
    require(worker["columns"] == module.model_columns(metadata, name), "Worker columns changed")
    memory = worker["memory"]
    require(0 < memory["maximum_sampled_rss_bytes"] <= protocol["limits"]["worker_rss_bytes"], "Invalid RSS")
    require(memory["maximum_sampled_gpu_driver_bytes"] == 0, "Unexpected GPU allocation")
    require(memory["maximum_sampled_rss_bytes"] <= memory["maximum_sampled_simultaneous_sum_bytes"]
            <= protocol["limits"]["rss_plus_driver_bytes"], "Invalid sampled memory sum")
    for symbol in module.SYMBOLS:
        check = worker["checks"][symbol]
        require(check["checkpoint_probabilities_exact"] and check["training_priors_exact"], "Replay check failed")
        errors = check["errors"]
        require(errors["future_prefix"] <= protocol["future_prefix_absolute_tolerance"], "Future dependence")
        require(all(0 <= v <= protocol["query_partition_absolute_tolerance"] for v in errors.values()), "Query dependence")
    return worker


def run(registration_path, phase):
    registration = read(registration_path)
    require(sha(Path(__file__)) == registration["continuation_driver_sha256"], "Continuation driver changed")
    source = (ROOT / registration["execution_root_relative_to_checkout"]).resolve()
    checkpoint = (ROOT / registration["checkpoint_attempt"]).resolve()
    output = source / registration["output_relative_to_execution_root"]
    protocol_path = source / registration["protocol"]
    require(output.parent.resolve() == source / "results", "Output must be inside historical results")
    for old in (checkpoint, (ROOT / registration["original_attempt"]).resolve()):
        require(output.resolve() != old and old not in output.resolve().parents
                and output.resolve() not in old.parents, "Attempts overlap")
    require(sha(protocol_path) == registration["protocol_sha256"], "Protocol changed")
    protocol = read(protocol_path)
    require(platform.python_version() == registration["runtime"]["python"], "Python changed")
    require({k: version(k) for k in registration["runtime"]["dependencies"]}
            == registration["runtime"]["dependencies"], "Dependency versions changed")
    for entry in registration["preserved_attempt_evidence"]:
        path = ROOT / entry["path"]
        require(sha(path) == entry["sha256"], "Pause evidence changed")
        verify_inventory(ROOT, read(path))
    verify_inputs(source, protocol)
    for path in (source / "src", source / "scripts"):
        sys.path.insert(0, str(path))
    module = importlib.import_module("run_boundary_midpoint_conversion_screen")
    require(Path(module.__file__).resolve() == source / "scripts/run_boundary_midpoint_conversion_screen.py", "Wrong source")
    require(module.MODEL_SPECS == protocol["model_specs"], "Model family changed")
    require(list(module.BASELINES) == protocol["baselines"], "Controls changed")
    require({k: list(v) for k, v in module.BLENDS.items()} == protocol["blends"], "Blends changed")
    identity = read(checkpoint / "frozen_screen.json")
    require(identity == {"protocol_sha256": sha(protocol_path), "input_hashes": protocol["input_hashes"],
                         **registration["runtime"]}, "Checkpoint identity changed")
    reused = {(r["date"], r["model"]) for r in registration["reused_workers"]}
    expected = {(day, name) for day in protocol["dates"] for name in module.MODEL_SPECS}
    require(len(expected) == registration["registered_fits"] == 48, "Wrong registered fit count")
    require(len(reused) == registration["reused_fits"] == 11 and reused < expected, "Wrong reused workers")
    require(len(expected - reused) == registration["fresh_fits"] == 37, "Wrong fresh fit count")
    write = module.write_json

    if phase == "train":
        require(not output.exists(), "Preserve every existing continuation attempt")
        output.mkdir(parents=True)
        shutil.copyfile(checkpoint / "frozen_screen.json", output / "frozen_screen.json")
        copied = {"frozen_screen.json": sha(output / "frozen_screen.json")}

        def preserve(relative):
            original, destination = checkpoint / relative, output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if original.is_dir():
                shutil.copytree(original, destination)
                paths = [p for p in original.rglob("*") if p.is_file()]
            else:
                shutil.copyfile(original, destination)
                paths = [original]
            for path in paths:
                rel = path.relative_to(checkpoint)
                require(sha(path) == sha(output / rel), f"Copy changed: {rel}")
                copied[str(rel)] = sha(path)

        for day in protocol["dates"]:
            preserve(Path("dates") / day / "prepared")
        for day, name in sorted(reused):
            worker_record(module, protocol, checkpoint / "dates" / day, day, name)
            for rel in (Path("workers") / name, Path(f"{name}.supervision.json"), Path(f"{name}.log")):
                preserve(Path("dates") / day / rel)
        write(output / "copy_manifest.json", copied)
        fresh_records = []
        for day in protocol["dates"]:
            folder = output / "dates" / day
            for name in module.MODEL_SPECS:
                if (day, name) in reused:
                    worker_record(module, protocol, folder, day, name)
                    continue
                started = time.monotonic()
                write(output / "progress.json", {"date": day, "model": name, "stage": "training",
                                                "assessment_metrics_sealed": True})
                log_path = folder / f"{name}.log"
                failure, returncode = None, None
                try:
                    with log_path.open("x") as log:
                        result = subprocess.run([sys.executable, str(Path(module.__file__)), "--protocol", str(protocol_path),
                            "--output", str(output), "--worker-date", day, "--worker-model", name], cwd=source,
                            env={**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
                            stdout=log, stderr=subprocess.STDOUT, timeout=protocol["limits"]["per_fit_wall_seconds"], check=False)
                    returncode = result.returncode
                except subprocess.TimeoutExpired:
                    failure = "registered_fit_wall_time_exceeded"
                supervision = {"date": day, "model": name, "returncode": returncode, "failure": failure,
                    "seconds": time.monotonic() - started, "log_sha256": sha(log_path)}
                write(folder / f"{name}.supervision.json", supervision)
                require(returncode == 0 and failure is None, f"Worker failed; preserve attempt: {day}/{name}")
                worker_record(module, protocol, folder, day, name)
                fresh_records.append({"date": day, "model": name})
                print(f"continuation_fit={day}/{name} completed={11 + len(fresh_records)}/48 seconds={supervision['seconds']:.1f}", flush=True)
            gc.collect()
        write(output / "training_completed.json", {"registration_sha256": sha(registration_path),
            "registered_fits": 48, "reused_fits": 11, "fresh_fits": len(fresh_records),
            "fresh_workers": fresh_records, "assessment_metrics_computed": False})
        print("training_complete=48/48 assessment_metrics_sealed=true", flush=True)
        return

    marker = read(output / "training_completed.json")
    require(marker["registration_sha256"] == sha(registration_path) and marker["fresh_fits"] == 37, "Incomplete training")
    require({(r["date"], r["model"]) for r in marker["fresh_workers"]} == expected - reused, "Missing/extra fresh procedure")
    require(not (output / "summary.json").exists(), "Preserve previous finalization")
    actual = {(p.parents[2].name, p.parent.name) for p in output.glob("dates/*/workers/*/worker.json")}
    require(actual == expected, "Missing/extra completed worker")
    for relative, digest in read(output / "copy_manifest.json").items():
        require(sha(output / relative) == digest, f"Copied artifact changed: {relative}")
    for day, name in sorted(expected):
        worker_record(module, protocol, output / "dates" / day, day, name)
    import numpy as np

    old_record = read(ROOT / registration["pause_evidence"])
    original = ROOT / registration["original_attempt"]
    arrays = 0
    for record in old_record["completed_workers"]:
        relative = Path("dates") / record["date"] / "workers" / record["model"]
        for symbol in module.SYMBOLS:
            with np.load(original / relative / f"{symbol}_probabilities.npz", allow_pickle=False) as old:
                with np.load(output / relative / f"{symbol}_probabilities.npz", allow_pickle=False) as new:
                    require(set(old.files) == set(new.files) == {"probabilities", "train_priors"}, "Array fields changed")
                    for field, shape in (("probabilities", (7070, 3)), ("train_priors", (3,))):
                        a, b = old[field], new[field]
                        require(a.dtype == b.dtype and a.shape == b.shape == shape, "Replay shape/dtype changed")
                        require(np.isfinite(a).all() and np.isfinite(b).all() and np.array_equal(a, b), "Exact replay failed")
                        arrays += 1
    require(arrays == 32, "Wrong repeat comparison count")
    write(output / "pre_reveal_compatibility.json", {"status": "passed", "registered_workers": 48,
        "replayed_arrays": arrays, "both_interrupted_inventories_reverified": True,
        "input_hashes_verified": len(protocol["input_hashes"]), "assessment_metrics_decoded": False})
    for day in protocol["dates"]:
        require(not (output / "dates" / day / "completed.json").exists(), "Preserve existing completion")
        module.finish_fold(day, protocol, output / "dates" / day, identity)
        gc.collect()
    write(output / "summary.json", module.reveal(protocol, output, identity))
    print("finalization_complete=48/48 panels=576 scores_not_printed=true", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("train", "finish"))
    args = parser.parse_args()
    run(args.registration.resolve(), args.phase)
