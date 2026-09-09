"""Test matched counted-depth information on the unchanged prediction problem."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import threading
import time
from datetime import date, timedelta
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_adaptive_priors import forecast_priors
from lob_forge.boundary_combined_inputs import load_combined_inputs, original_combined_columns
from lob_forge.boundary_confirm_model import SYMBOLS, fit_confirm_neural
from lob_forge.boundary_event_inputs import utc_ms
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_okx_inputs import OKX_VARIANTS, PEER_COUNTS, PEER_QUANTITY, append_counted_features, counted_columns
from lob_forge.boundary_okx_quantity_scale import quantity_scale_features
from lob_forge.boundary_pooled import PooledForecaster
from lob_forge.boundary_prequential import decoded_release_clock
from lob_forge.boundary_regime_coverage import calendar_stride_mask, history_cohorts
from lob_forge.boundary_weighted_boost import fit_weighted_boost
from benchmark_boundary_tabicl_feature_budget import memory_guard
from run_boundary_confirmation import noon, save_predictions, write_json
from run_boundary_event_clock_screen import check_completed
from run_boundary_tabicl_screen import verify_artifacts

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ("original_reference", "observations_tree", "observations_neural", "observations_blend",
    "combined_hgb", "combined_neural", "deep500_hgb", "combined_instant_hgb_old_neural")
MODEL_SPECS = {f"{variant}_{kind}": {"variant": variant, "representation": "combined", "kind": kind, "leaves": 7 if kind == "hgb" else None}
    for variant in OKX_VARIANTS for kind in ("hgb", "neural")}
MODEL_SPECS.update({f"{variant}_hgb31": {"variant": variant, "representation": "combined", "kind": "hgb", "leaves": 31}
                   for variant in ("q100_100", "n100_100")})
OBSERVATION_VARIANTS = ("q100_100", "n100_100", "q100_500", "n100_500")
MODEL_SPECS.update({f"observations_{variant}_{kind}": {"variant": variant, "representation": "observations",
    "kind": kind, "leaves": 7 if kind == "hgb" else None}
    for variant in OBSERVATION_VARIANTS for kind in ("hgb", "neural")})
BLENDS = {**{f"{v}_blend": (f"{v}_hgb", f"{v}_neural") for v in OKX_VARIANTS},
    **{f"{v}_hgb_old_neural": (f"{v}_hgb", "combined_neural") for v in OKX_VARIANTS},
    **{f"{v}_neural_deep_hgb": (f"{v}_neural", "deep500_hgb") for v in OKX_VARIANTS},
    **{f"{v}_hgb31_old_neural": (f"{v}_hgb31", "combined_neural") for v in ("q100_100", "n100_100")}}
BLENDS.update({**{f"observations_{v}_blend": (f"observations_{v}_hgb", f"observations_{v}_neural") for v in OBSERVATION_VARIANTS},
    **{f"observations_{v}_hgb_old_neural": (f"observations_{v}_hgb", "observations_neural") for v in OBSERVATION_VARIANTS},
    **{f"observations_{v}_neural_deep_hgb": (f"observations_{v}_neural", "deep500_hgb") for v in OBSERVATION_VARIANTS}})
GROUPS = ((25, 100), (100, 100), (100, 500))


def model_columns(metadata, name):
    spec = MODEL_SPECS[name]
    columns = metadata["variant_columns"][spec["variant"]]
    if spec["representation"] == "observations":
        columns = [c for c in columns if c not in metadata["original_columns"] or c in metadata["observation_columns"]]
    return list(columns)


def wide_counted_inputs(original, clocks, own, peer):
    """Store shared columns once; each variant reads its exact named subset."""
    pieces, columns = [original.reset_index(drop=True)], {}
    for levels, delay in GROUPS:
        group = f"top{levels}_{delay}"
        joined = append_counted_features(original, clocks, own[group], peer[group], include_counts=True)
        addition = joined.iloc[:, len(original.columns):].copy()
        rename = {c: (f"peer_okx_{group}__" + c[len("peer_okx_"):] if c.startswith("peer_okx_")
                      else f"okx_{group}__" + c[len("okx_"):]) for c in addition}
        pieces.append(addition.rename(columns=rename))
        del joined, addition
    frame = pd.concat(pieces, axis=1)
    for variant, (levels, delay, counts) in OKX_VARIANTS.items():
        group = f"top{levels}_{delay}"
        columns[variant] = [*original.columns,
            *(f"okx_{group}__{c}" for c in counted_columns(own[group].columns, counts)),
            *(f"peer_okx_{group}__{c}" for c in (*PEER_QUANTITY, *(PEER_COUNTS if counts else ())))]
        if not set(columns[variant]).issubset(frame.columns):
            raise ValueError("A registered variant is missing its exact observed fields")
    if frame.columns.duplicated().any():
        raise ValueError("Shared counted-feature cache requires unique column names")
    pd.testing.assert_frame_equal(frame[list(original.columns)], original.reset_index(drop=True), check_exact=True)
    return frame, columns


def prepare_fold(day, protocol, folder, identity):
    import torch

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if folder.exists():
        raise ValueError("Preserve each previous counted-model input preparation")
    folder.mkdir(parents=True)
    start = time.monotonic()
    manifest = json.loads((ROOT / protocol["manifest"]).read_text())
    feature_manifest = json.loads((ROOT / protocol["feature_manifest"]).read_text())
    if not feature_manifest["complete"] or len(feature_manifest["sessions"]) != 28:
        raise ValueError("All 28 registered source feature caches are required before model work")
    features = {(r["symbol"], r["date"]): r for r in feature_manifest["sessions"]}
    training_days = history_cohorts(day)["recent4_stride4"]["train_dates"]
    validation_day = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    source = ROOT / protocol["baseline_run"] / "dates" / day
    check_completed(source, json.loads((source.parent.parent / "frozen_screen.json").read_text()))
    original_tree = joblib.load(ROOT / protocol["newton_run"] / "dates" / day / "combined_hgb.joblib")
    original_neural = PooledForecaster.load(ROOT / protocol["basis_run"] / "dates" / day / "combined_neural")
    partitions, variants, all_y, assessments, baseline_checks = {}, None, {}, {}, {}
    for current in [*training_days, validation_day, day]:
        entries = [r for r in manifest["sessions"] if r["session_date"] == current]
        if len(entries) != 2 or {r["symbol"] for r in entries} != set(SYMBOLS):
            raise ValueError("Exactly two original asset sources per date are required")
        part = folder / "inputs" / f"{current}.json"
        write_json(part, {"sessions": entries})
        x, y, clocks = load_combined_inputs(ROOT, part, ROOT / protocol["depth_manifest"], ROOT / protocol["spot_manifest"])
        additions = {}
        for symbol in SYMBOLS:
            record = features[symbol, current]
            raw = next(r for r in entries if r["symbol"] == symbol)
            if record["original_features_sha256"] != raw["sha256"]:
                raise ValueError("Counted-feature and original observation identities differ")
            source_record_path = ROOT / record["source_record"]
            if sha256_file(source_record_path) != record["source_record_sha256"]:
                raise ValueError("The source record for absolute quantity scale changed")
            source_record = json.loads(source_record_path.read_text())
            sidecar = ROOT / source_record["observation_path"]
            if sha256_file(sidecar) != record["source_observation_sha256"]:
                raise ValueError("The absolute quantity sidecar changed")
            with np.load(sidecar, allow_pickle=False) as saved:
                observations = {k: saved[k] for k in saved.files}
            additions[symbol] = {}
            for levels, delay in GROUPS:
                group = f"top{levels}_{delay}"
                record_group = record["groups"][group]
                path = ROOT / record_group["features_path"]
                if sha256_file(path) != record_group["sha256"]:
                    raise ValueError("Frozen counted-feature cache changed")
                frame = pd.read_parquet(path).set_index("decision_time")
                if list(frame.columns) != record_group["columns"] or len(frame) != record["decision_rows"]:
                    raise ValueError("Counted-feature schema or row inventory changed")
                np.testing.assert_array_equal(frame.index.to_numpy(), observations["decision_times"])
                scale = quantity_scale_features(observations, levels=levels, delay_ms=delay)
                scale.index = frame.index
                additions[symbol][group] = pd.concat([frame, scale], axis=1)
            del observations
            gc.collect()
        partitions[current] = {}
        for asset, symbol in enumerate(SYMBOLS):
            key = symbol, current
            peer = "ETHUSDT" if symbol == "BTCUSDT" else "BTCUSDT"
            original = x[key].reset_index(drop=True)
            if (list(original.columns) != original_tree.columns or list(original.columns) != original_neural.columns
                or len(original.columns) != 399 or len(original_combined_columns(original)) != 220):
                raise ValueError("The original 399-column predictor schema changed")
            mask = calendar_stride_mask(clocks[key], utc_ms(current), stride_seconds=4) if current in training_days else noon(clocks[key], current)
            selected_times, labels = clocks[key][mask], y[key][mask]
            if current not in training_days and len(selected_times) != 7070:
                raise ValueError("All original 7070 noon decisions must remain in each asset/date")
            raw = next(r for r in entries if r["symbol"] == symbol)
            if current != day:
                outcomes = pd.read_parquet(ROOT / raw["features_path"], columns=["decision_time", "label", "future_event_time"]).set_index("decision_time").loc[selected_times]
                np.testing.assert_array_equal(outcomes.label.to_numpy(), labels)
                release = decoded_release_clock(outcomes.future_event_time.to_numpy())
                limit = utc_ms(validation_day if current in training_days else day)
                if (release >= limit).any():
                    raise ValueError("A historical outcome was not released before the next partition")
                maximum_release = int(release.max())
                all_y[symbol, current] = labels.copy()
            else:
                maximum_release = None
                with np.load(source / "predictions" / f"{symbol}_combined_hgb_registered.npz", allow_pickle=False) as saved:
                    np.testing.assert_array_equal(selected_times, saved["decision_times"])
                    np.testing.assert_array_equal(labels, saved["labels"])
                    np.testing.assert_array_equal(original_tree.predict_proba(original.loc[mask], symbol), saved["probabilities"])
                with np.load(source / "predictions" / f"{symbol}_combined_neural_registered.npz", allow_pickle=False) as saved:
                    np.testing.assert_array_equal(original_neural.predict_proba(original.loc[mask], asset), saved["probabilities"])
                assessments[symbol] = {"labels": labels.copy(), "decision_times": selected_times.copy()}
                baseline_checks[symbol] = {"original_tree_forecasts_exact": True, "original_neural_forecasts_exact": True}
            wide, selected_columns = wide_counted_inputs(original, clocks[key], additions[symbol], additions[peer])
            if variants is not None and variants != selected_columns:
                raise ValueError("Variant schemas must remain identical across every source partition")
            variants = selected_columns
            destination = folder / "matrices" / current / f"{symbol}.parquet"
            destination.parent.mkdir(parents=True, exist_ok=True)
            chosen = wide.loc[mask].reset_index(drop=True)
            chosen.to_parquet(destination, index=False, compression="zstd")
            pd.testing.assert_frame_equal(chosen, pd.read_parquet(destination), check_exact=True)
            clock_path = destination.with_suffix(".times.npy")
            np.save(clock_path, selected_times, allow_pickle=False)
            entry = {"features": str(destination.relative_to(folder)), "times": str(clock_path.relative_to(folder)),
                "rows": len(chosen), "maximum_label_release_ms": maximum_release}
            if current != day:
                label_path = destination.with_suffix(".labels.npy")
                np.save(label_path, labels, allow_pickle=False)
                entry["labels"] = str(label_path.relative_to(folder))
            partitions[current][symbol] = entry
            del wide, chosen, original
            gc.collect()
        del additions, x, y, clocks
        gc.collect()
    for asset, symbol in enumerate(SYMBOLS):
        labels = np.concatenate([all_y[symbol, current] for current in training_days])
        priors = np.array([(labels == k).mean() for k in (-1, 0, 1)])
        np.testing.assert_array_equal(priors, original_tree.priors[asset])
        np.testing.assert_array_equal(priors, original_neural.priors[asset])
        np.savez_compressed(folder / f"sealed_assessment_{symbol}.npz", **assessments[symbol], train_priors=priors)
    metadata = {"date": day, "training_dates": training_days, "validation_date": validation_day,
        "original_columns": original_tree.columns, "observation_columns": original_combined_columns(pd.DataFrame(columns=original_tree.columns)),
        "partitions": partitions, "variant_columns": variants, "baseline_checks": baseline_checks,
        "assessment_labels_decoded_by_training_workers": False, "model_fits": 0, "seconds": time.monotonic() - start}
    write_json(folder / "metadata.json", metadata)
    record = {"identity": identity, "metadata": metadata,
        "artifact_hashes": {str(p.relative_to(folder)): sha256_file(p) for p in folder.rglob("*") if p.is_file()}}
    write_json(folder / "prepared.json", record)
    return record


def fit_worker(protocol_path, output, day, name):
    import torch

    protocol = json.loads(protocol_path.read_text())
    if protocol["model_specs"] != MODEL_SPECS or day not in protocol["dates"]:
        raise ValueError("Only the frozen counted-model family may be fitted")
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
                raise ValueError(f"Counted-model query dependence exceeded its fixed tolerance: {errors}")
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
        raise ValueError("Every fixed counted-model reporting panel must be present")
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
        raise ValueError("The exact registered counted-model family is required")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen counted-model input changed: {path}")
    if output.exists():
        raise ValueError("Preserve every previous counted-model attempt; no implicit resume")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"],
        "python": platform.python_version(), "dependencies": {k: version(k) for k in ("numpy", "pandas", "scikit-learn", "torch", "pyarrow", "psutil")}}
    write_json(output / "frozen_screen.json", identity)
    for day in protocol["dates"]:
        folder = output / "dates" / day
        prepare_fold(day, protocol, folder / "prepared", identity)
        print(f"counted_inputs_verified={day}", flush=True)
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Input changed before counted-model fitting: {path}")
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
                raise ValueError(f"Counted-model worker failed; preserve its attempt: {day}/{name}")
            print(f"counted_fit={day}/{name} seconds={supervision['seconds']:.1f}", flush=True)
        finish_fold(day, protocol, folder, identity)
        gc.collect()
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Input changed during counted-model execution: {path}")
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
