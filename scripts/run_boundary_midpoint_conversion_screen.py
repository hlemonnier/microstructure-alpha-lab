"""Test matched trade-side conversion representations on unchanged targets."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_adaptive_priors import forecast_priors
from lob_forge.boundary_confirm_model import SYMBOLS, fit_confirm_neural
from lob_forge.boundary_event_inputs import utc_ms
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_pooled import PooledForecaster
from lob_forge.boundary_quote_currency_inputs import quote_variant_columns
from lob_forge.boundary_midpoint_conversion_inputs import append_midpoint_conversion, midpoint_model_columns
from lob_forge.boundary_weighted_boost import fit_weighted_boost
from benchmark_boundary_tabicl_feature_budget import memory_guard
from run_boundary_confirmation import save_predictions, write_json
from run_boundary_event_clock_screen import check_completed
from run_boundary_tabicl_screen import verify_artifacts

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ("original_reference", "observations_tree", "observations_neural", "observations_blend",
    "combined_hgb", "combined_neural", "deep500_hgb", "combined_instant_hgb_old_neural")
MODEL_SPECS = {f"{representation}_{variant}_{mode}_{kind}": {"variant": variant, "representation": representation,
    "conversion_mode": mode, "kind": kind, "leaves": 7 if kind == "hgb" else None}
    for representation in ("observations", "combined") for variant in ("fx100", "btc100")
    for mode in ("raw_side", "midpoint_side") for kind in ("hgb", "neural")}
BLENDS = {}
for _representation in ("observations", "combined"):
    for _variant in ("fx100", "btc100"):
        for _mode in ("raw_side", "midpoint_side"):
            _prefix = f"{_representation}_{_variant}_{_mode}"
            BLENDS[f"{_prefix}_blend"] = (f"{_prefix}_hgb", f"{_prefix}_neural")
            BLENDS[f"{_prefix}_hgb_old_neural"] = (f"{_prefix}_hgb", f"{_representation}_neural")
            BLENDS[f"{_prefix}_neural_deep_hgb"] = (f"{_prefix}_neural", "deep500_hgb")


def model_columns(metadata, name):
    spec = MODEL_SPECS[name]
    return midpoint_model_columns(metadata, **{k:spec[k] for k in ("representation", "variant", "conversion_mode")})


def prepare_fold(day, protocol, folder, identity):
    if folder.exists():
        raise ValueError("Preserve every previous quote-source input preparation")
    folder.mkdir(parents=True)
    started = time.monotonic()
    source = ROOT / protocol["original_prepared_run"] / "dates" / day / "prepared"
    frozen = json.loads((source / "prepared.json").read_text())
    if frozen["identity"]["protocol_sha256"] != protocol["original_prepared_protocol_sha256"]:
        raise ValueError("The original frozen selected-row preparation changed")
    verify_artifacts(source, frozen)
    original_metadata = frozen["metadata"]
    if (len(original_metadata["original_columns"]) != 399 or len(original_metadata["observation_columns"]) != 220
        or original_metadata["assessment_labels_decoded_by_training_workers"] or original_metadata["model_fits"] != 0):
        raise ValueError("Only the exact original predictor preparation can be augmented")
    for check in original_metadata["baseline_checks"].values():
        if not check["original_tree_forecasts_exact"] or not check["original_neural_forecasts_exact"]:
            raise ValueError("Original tree and neural forecasts must have exact prior replay evidence")
    feature_manifest = json.loads((ROOT / protocol["feature_manifest"]).read_text())
    if not feature_manifest["complete"] or len(feature_manifest["sessions"]) != 14:
        raise ValueError("Require the complete original alternative-quote feature window")
    sessions = {r["date"]: r for r in feature_manifest["sessions"]}
    partitions = {}
    for current, entries in original_metadata["partitions"].items():
        additions = {}
        record = sessions[current]
        for mode in ("raw_side", "midpoint_side"):
            group = record["groups"][f"{mode}_100"]
            path = ROOT / group["features_path"]
            if sha256_file(path) != group["sha256"]:
                raise ValueError("The original side-aware conversion feature cache changed")
            frame = pd.read_parquet(path)
            if list(frame.columns) != group["columns"] or len(frame) != record["decision_rows"]:
                raise ValueError("The original side-aware conversion cache dimensions changed")
            additions[mode] = frame.set_index("decision_time")
        partitions[current] = {}
        for symbol in SYMBOLS:
            entry = entries[symbol]
            times = np.load(source / entry["times"], allow_pickle=False)
            original = pd.read_parquet(source / entry["features"])
            if list(original.columns) != quote_variant_columns(original_metadata["original_columns"], "both100"):
                raise ValueError("The exact original full quote-source matrix schema changed")
            if len(original) != entry["rows"] or (current not in original_metadata["training_dates"] and len(original) != 7070):
                raise ValueError("Every original selected training, validation and query row must be retained")
            if current == day:
                if "labels" in entry or entry["maximum_label_release_ms"] is not None:
                    raise ValueError("Query input metadata cannot expose labels to a training worker")
            else:
                limit = utc_ms(original_metadata["validation_date"] if current in original_metadata["training_dates"] else day)
                if "labels" not in entry or not entry["maximum_label_release_ms"] < limit:
                    raise ValueError("All training/validation labels must precede their next partition")
            augmented = append_midpoint_conversion(original, times, additions["raw_side"], additions["midpoint_side"])
            destination = folder / "matrices" / current / f"{symbol}.parquet"
            destination.parent.mkdir(parents=True, exist_ok=True)
            augmented.to_parquet(destination, index=False, compression="zstd")
            pd.testing.assert_frame_equal(augmented, pd.read_parquet(destination), check_exact=True)
            clock_path = destination.with_suffix(".times.npy")
            shutil.copyfile(source / entry["times"], clock_path)
            np.testing.assert_array_equal(times, np.load(clock_path, allow_pickle=False))
            selected = {**entry, "features": str(destination.relative_to(folder)), "times": str(clock_path.relative_to(folder))}
            if current != day:
                label_path = destination.with_suffix(".labels.npy")
                shutil.copyfile(source / entry["labels"], label_path)
                if sha256_file(label_path) != sha256_file(source / entry["labels"]):
                    raise ValueError("The exact historical label bytes changed")
                selected["labels"] = str(label_path.relative_to(folder))
            partitions[current][symbol] = selected
            del original, augmented
            gc.collect()
        del additions
        gc.collect()
    for symbol in SYMBOLS:
        sealed = source / f"sealed_assessment_{symbol}.npz"
        shutil.copyfile(sealed, folder / sealed.name)
        if sha256_file(sealed) != sha256_file(folder / sealed.name):
            raise ValueError("The sealed original assessment changed")
    metadata = {**{k: original_metadata[k] for k in ("date", "training_dates", "validation_date", "original_columns", "observation_columns", "baseline_checks")},
        "partitions": partitions, "assessment_labels_decoded_by_training_workers": False, "model_fits": 0,
        "original_selected_feature_values_exact": True, "original_selected_clocks_and_labels_exact": True,
        "original_prepared_sha256": sha256_file(source / "prepared.json"), "seconds": time.monotonic() - started}
    write_json(folder / "metadata.json", metadata)
    record = {"identity": identity, "metadata": metadata,
        "artifact_hashes": {str(p.relative_to(folder)): sha256_file(p) for p in folder.rglob("*") if p.is_file()}}
    write_json(folder / "prepared.json", record)
    return record


def fit_worker(protocol_path, output, day, name):
    import torch

    protocol = json.loads(protocol_path.read_text())
    if protocol["model_specs"] != MODEL_SPECS or day not in protocol["dates"]:
        raise ValueError("Only the frozen midpoint-conversion model family may be fitted")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    spec = MODEL_SPECS[name]
    folder = output / "dates" / day / "workers" / name
    folder.mkdir(parents=True)
    readings = {"maximum_sampled_rss_bytes": 0, "maximum_sampled_gpu_driver_bytes": 0, "maximum_sampled_simultaneous_sum_bytes": 0}
    stop = threading.Event()
    guard = threading.Thread(target=memory_guard, args=("cpu", folder, protocol["limits"], stop, readings), daemon=True)
    guard.start()
    started = time.monotonic()
    prepared = output / "dates" / day / "prepared"
    frozen = json.loads((prepared / "prepared.json").read_text())
    if frozen["identity"]["protocol_sha256"] != sha256_file(protocol_path):
        raise ValueError("Worker inputs belong to a different frozen model protocol")
    verify_artifacts(prepared, frozen)  # Hashing a sealed file does not decode its labels.
    metadata = frozen["metadata"]
    columns = model_columns(metadata, name)
    tx, ty, vx, vy, qx = {}, {}, {}, {}, {}
    for symbol in SYMBOLS:
        train_parts = [metadata["partitions"][current][symbol] for current in metadata["training_dates"]]
        tx[symbol] = pd.concat([pd.read_parquet(prepared / p["features"], columns=columns) for p in train_parts], ignore_index=True)
        ty[symbol] = np.concatenate([np.load(prepared / p["labels"], allow_pickle=False) for p in train_parts])
        val = metadata["partitions"][metadata["validation_date"]][symbol]
        vx[symbol] = pd.read_parquet(prepared / val["features"], columns=columns)
        vy[symbol] = np.load(prepared / val["labels"], allow_pickle=False)
        query = metadata["partitions"][day][symbol]
        if "labels" in query:
            raise ValueError("Worker query metadata must not expose an assessment-label path")
        qx[symbol] = pd.read_parquet(prepared / query["features"], columns=columns)
    fit_started = time.monotonic()
    with threadpool_limits(limits=2):
        if spec["kind"] == "hgb":
            model = fit_weighted_boost(tx, ty, leaves=spec["leaves"], class_balanced=True)
            joblib.dump(model, folder / "model.joblib")
            restored = joblib.load(folder / "model.joblib")
        else:
            model, history = fit_confirm_neural(tx, ty, vx, vy)
            model.save(folder / "model")
            write_json(folder / "training_history.json", history)
            restored = PooledForecaster.load(folder / "model")
        ready_seconds = time.monotonic() - fit_started
        checks = {}
        for asset, symbol in enumerate(SYMBOLS):
            identifier = symbol if spec["kind"] == "hgb" else asset
            expected_priors = np.array([(ty[symbol] == k).mean() for k in (-1, 0, 1)])
            np.testing.assert_array_equal(restored.priors[asset], expected_priors)
            query = qx[symbol]
            p = restored.predict_proba(query, identifier)
            np.testing.assert_array_equal(p, model.predict_proba(query, identifier))
            prefix = query.iloc[:64].copy()
            p0 = restored.predict_proba(prefix, identifier)
            changed = prefix.copy()
            changed.iloc[17:] = 1e6
            altered = restored.predict_proba(changed, identifier)
            shorter = restored.predict_proba(prefix.iloc[:17], identifier)
            individual = np.concatenate([restored.predict_proba(prefix.iloc[i:i+1], identifier) for i in range(7)])
            errors = {"full_query_prefix": float(np.max(np.abs(p[:64] - p0))),
                "future_prefix": float(np.max(np.abs(p0[:17] - altered[:17]))),
                "shorter_prefix": float(np.max(np.abs(p0[:17] - shorter))),
                "individual_query": float(np.max(np.abs(p0[:7] - individual)))}
            if errors["future_prefix"] > protocol["future_prefix_absolute_tolerance"] or max(errors.values()) > protocol["query_partition_absolute_tolerance"]:
                raise ValueError(f"Midpoint-conversion model query dependence exceeded its fixed tolerance: {errors}")
            np.savez_compressed(folder / f"{symbol}_probabilities.npz", probabilities=p, train_priors=expected_priors)
            checks[symbol] = {"checkpoint_probabilities_exact": True, "training_priors_exact": True, "errors": errors}
    stop.set()
    guard.join()
    result = {"model": name, "specification": spec, "columns": columns, "train_rows": {s: len(ty[s]) for s in SYMBOLS},
        "fit_save_reload_seconds": ready_seconds, "seconds": time.monotonic() - started, "memory": readings,
        "checks": checks, "assessment_labels_decoded": False}
    write_json(folder / "worker.json", result)


def finish_fold(day, protocol, folder, identity):
    source = ROOT / protocol["baseline_run"] / "dates" / day
    prepared = folder / "prepared"
    metadata = json.loads((prepared / "metadata.json").read_text())
    for symbol in SYMBOLS:
        with np.load(prepared / f"sealed_assessment_{symbol}.npz", allow_pickle=False) as saved:
            labels, clock, priors = (saved[k].copy() for k in ("labels", "decision_times", "train_priors"))
        predictions = {}
        for name in BASELINES:
            with np.load(source / "predictions" / f"{symbol}_{name}_registered.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(labels, saved["labels"])
                np.testing.assert_array_equal(clock, saved["decision_times"])
                predictions[name] = {k: saved[k].copy() for k in saved.files}
        for name in MODEL_SPECS:
            with np.load(folder / "workers" / name / f"{symbol}_probabilities.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["train_priors"], priors)
                predictions[name] = {"probabilities": saved["probabilities"].copy(), "train_priors": priors, "decision_priors": priors}
        for name, (left, right) in BLENDS.items():
            predictions[name] = {"probabilities": (predictions[left]["probabilities"] + predictions[right]["probabilities"]) / 2,
                "train_priors": priors, "decision_priors": priors}
        with np.load(ROOT / protocol["prior_run"] / "priors" / day / f"{symbol}_original_reference.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(clock, saved["decision_times"])
            initial = saved["labels_cumulative"][0].copy()
        for name, values in predictions.items():
            p = values["probabilities"]
            for policy in ("registered", "forecast_3600"):
                decision_priors = values["decision_priors"] if policy == "registered" else forecast_priors(p, clock, initial, half_life_seconds=3600)
                save_predictions(folder / "predictions" / f"{symbol}_{name}_{policy}.npz", p, labels, clock, values["train_priors"], decision_priors)
    record = {"identity": identity, "date": day, "model_fits": len(MODEL_SPECS), "preparation": metadata,
        "artifact_hashes": {str(p.relative_to(folder)): sha256_file(p) for p in folder.rglob("*") if p.is_file()}}
    write_json(folder / "completed.json", record)


def reveal(protocol, output, identity):
    records = []
    for day in protocol["dates"]:
        folder = output / "dates" / day
        check_completed(folder, identity)
        for symbol in SYMBOLS:
            for name in (*BASELINES, *MODEL_SPECS, *BLENDS):
                for policy in ("registered", "forecast_3600"):
                    with np.load(folder / "predictions" / f"{symbol}_{name}_{policy}.npz", allow_pickle=False) as saved:
                        metrics = classification_metrics(SimpleNamespace(priors=saved["decision_priors"]), saved["probabilities"], saved["labels"])
                    records.append({"date": day, "symbol": symbol, "model": name, "policy": policy, "metrics": metrics})
    if len(records) != protocol["post_fit_panels"]:
        raise ValueError("Every fixed midpoint-conversion model reporting panel must be present")
    leaderboard = []
    for name in (*BASELINES, *MODEL_SPECS, *BLENDS):
        for policy in ("registered", "forecast_3600"):
            rows = [r for r in records if r["model"] == name and r["policy"] == policy]
            leaderboard.append({"model": name, "policy": policy,
                "means": {k: float(np.mean([r["metrics"][k] for r in rows])) for k in ("balanced_accuracy", "natural_accuracy", "log_loss", "macro_f1")},
                "balanced_accuracy_by_asset": {s: float(np.mean([r["metrics"]["balanced_accuracy"] for r in rows if r["symbol"] == s])) for s in SYMBOLS}})
    return {"identity": identity, "dates": protocol["dates"], "trainable_model_fits": len(MODEL_SPECS) * len(protocol["dates"]),
        "post_fit_panels": len(records), "records": records, "leaderboard": sorted(leaderboard, key=lambda r:r["means"]["balanced_accuracy"], reverse=True),
        "evidence_status": "development_only_on_exposed_dates", "substantial_gain_confirmed": False}


def run(protocol_path, output, prepare_only=False):
    protocol = json.loads(protocol_path.read_text())
    if (protocol["model_specs"] != MODEL_SPECS or protocol["baselines"] != list(BASELINES)
        or protocol["blends"] != {k: list(v) for k, v in BLENDS.items()}
        or protocol["evidence_status"] != "development_only_on_exposed_dates"):
        raise ValueError("The exact registered midpoint-conversion model family is required")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen midpoint-conversion model input changed: {path}")
    if output.exists():
        raise ValueError("Preserve every previous midpoint-conversion model attempt; no implicit resume")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"],
        "python": platform.python_version(), "dependencies": {k: version(k) for k in ("numpy", "pandas", "scikit-learn", "torch", "pyarrow", "psutil")}}
    write_json(output / "frozen_screen.json", identity)
    for day in protocol["dates"]:
        folder = output / "dates" / day
        prepare_fold(day, protocol, folder / "prepared", identity)
        print(f"midpoint_inputs_verified={day}", flush=True)
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Input changed before midpoint-conversion model fitting: {path}")
    if prepare_only:
        write_json(output / "preparation_summary.json", {"identity": identity, "all_dates_prepared": True, "model_fits": 0, "assessment_metrics_computed": False})
        return
    for day in protocol["dates"]:
        folder = output / "dates" / day
        for name in MODEL_SPECS:
            started = time.monotonic()
            log_path = folder / f"{name}.log"
            write_json(output / "progress.json", {"date": day, "model": name, "stage": "training", "assessment_metrics_sealed": True})
            failure, returncode = None, None
            try:
                with log_path.open("w") as log:
                    process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--protocol", str(protocol_path),
                        "--output", str(output), "--worker-date", day, "--worker-model", name], cwd=ROOT,
                        env={**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
                        stdout=log, stderr=subprocess.STDOUT, timeout=protocol["limits"]["per_fit_wall_seconds"], check=False)
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                failure = "registered_fit_wall_time_exceeded"
            supervision = {"date": day, "model": name, "returncode": returncode, "failure": failure, "seconds": time.monotonic() - started,
                "log_sha256": sha256_file(log_path)}
            write_json(folder / f"{name}.supervision.json", supervision)
            if returncode != 0 or failure is not None:
                raise ValueError(f"Midpoint-conversion model worker failed; preserve its attempt: {day}/{name}")
            print(f"midpoint_fit={day}/{name} seconds={supervision['seconds']:.1f}", flush=True)
        gc.collect()
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Input changed during midpoint-conversion model execution: {path}")
    for day in protocol["dates"]:
        finish_fold(day, protocol, output / "dates" / day, identity)
        gc.collect()
    write_json(output / "summary.json", reveal(protocol, output, identity))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker-date")
    parser.add_argument("--worker-model")
    args = parser.parse_args()
    if args.worker_date:
        fit_worker(args.protocol.resolve(), args.output.resolve(), args.worker_date, args.worker_model)
    else:
        run(args.protocol.resolve(), args.output.resolve(), args.prepare_only)
