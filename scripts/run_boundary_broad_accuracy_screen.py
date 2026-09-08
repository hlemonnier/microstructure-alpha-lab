"""Audit unchanged observation models across all twenty exposed dates."""

from __future__ import annotations

import argparse
import gc
import json
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

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_adaptive_priors import forecast_priors
from lob_forge.boundary_confirmation_integrity import verify_frozen_fold
from lob_forge.boundary_confirm_model import SYMBOLS, fit_confirm_neural
from lob_forge.boundary_event_inputs import utc_ms
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_pooled import PooledForecaster
from lob_forge.boundary_regime_coverage import calendar_stride_mask, history_cohorts
from lob_forge.boundary_weighted_boost import fit_weighted_boost
from benchmark_boundary_tabicl_feature_budget import memory_guard
from run_boundary_confirmation import noon, save_predictions, write_json
from run_boundary_event_clock_screen import check_completed
from run_boundary_tabicl_screen import verify_artifacts

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ("original_reference", "matched_data_control", "original_blend", "pooled_balanced_tree_blend")
MODEL_SPECS = {"observations_tree": "tree", "observations_neural": "neural"}
BLENDS = {"observations_blend": ("observations_tree", "observations_neural")}


def prepare(day, protocol, folder, identity):
    if folder.exists():
        raise ValueError("Preserve every previous broad-anchor preparation attempt")
    folder.mkdir(parents=True)
    started = time.monotonic()
    context_path = ROOT / protocol["context_manifest"]
    if sha256_file(context_path) != protocol["input_hashes"][protocol["context_manifest"]]:
        raise ValueError("The frozen full-calendar context inventory changed")
    cache = json.loads(context_path.read_text())
    training_days = history_cohorts(day)["recent4_stride4"]["train_dates"]
    validation_day = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    tx, ty, tt, vx, vy, qx, qy, qt, releases = {}, {}, {}, {}, {}, {}, {}, {}, []
    for current in [*training_days, validation_day, day]:
        for symbol in SYMBOLS:
            record = next(r for r in cache["sessions"] if (r["symbol"], r["date"]) == (symbol, current))
            for key in ("features", "outcomes"):
                if sha256_file(ROOT / record[key]) != record[key + "_sha256"]:
                    raise ValueError("The original full-calendar cached observations changed")
            frame = pd.read_parquet(ROOT / record["features"])
            if list(frame.columns) != cache["columns"] or len(frame.columns) != 220:
                raise ValueError("The original 220 observation features must remain")
            with np.load(ROOT / record["outcomes"], allow_pickle=False) as saved:
                target, clock, released = (saved[k].copy() for k in ("labels", "decision_times", "release_times"))
            keep = calendar_stride_mask(clock, utc_ms(current), stride_seconds=4) if current in training_days else noon(clock, current)
            cutoff = utc_ms(validation_day if current in training_days else day)
            if current != day and (released[keep] >= cutoff).any():
                raise ValueError("Every historical or validation label must be released before the next partition")
            if current in training_days:
                tx[symbol, current], ty[symbol, current], tt[symbol, current] = frame.loc[keep].copy(), target[keep], clock[keep]
            else:
                if keep.sum() != 7070:
                    raise ValueError("Every original noon query is required")
                if current == validation_day:
                    vx[symbol], vy[symbol] = frame.loc[keep].copy(), target[keep]
                else:
                    qx[symbol], qy[symbol], qt[symbol] = frame.loc[keep].copy(), target[keep], clock[keep]
                    path = ROOT / protocol["reference_run"] / "dates" / day / "predictions" / f"{symbol}_original_reference.npz"
                    with np.load(path, allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved["labels"], qy[symbol])
                        np.testing.assert_array_equal(saved["decision_times"], qt[symbol])
            releases.append({"date": current, "symbol": symbol, "selected_rows": int(keep.sum()),
                "maximum_selected_release_ms": int(released[keep].max()), "past_release_cutoff": cutoff if current != day else None})
            del frame, target, clock, released
        gc.collect()
    tx = {s: pd.concat([tx[s, d] for d in training_days], ignore_index=True) for s in SYMBOLS}
    ty = {s: np.concatenate([ty[s, d] for d in training_days]) for s in SYMBOLS}
    tt = {s: np.concatenate([tt[s, d] for d in training_days]) for s in SYMBOLS}
    population = {s: [(ty[s] == c).mean() for c in (-1, 0, 1)] for s in SYMBOLS}
    joblib.dump({"features": tx, "labels": ty, "times": tt, "validation_features": vx, "validation_labels": vy}, folder / "training.joblib")
    joblib.dump(qx, folder / "queries.joblib")
    np.savez_compressed(folder / "assessment.npz", **{f"{s}_labels": qy[s] for s in SYMBOLS}, **{f"{s}_times": qt[s] for s in SYMBOLS})
    write_json(folder / "preparation_metadata.json", {"training_dates": training_days, "validation_date": validation_day,
        "population_priors": population, "training_rows": {s: len(ty[s]) for s in SYMBOLS}, "source_release_checks": releases})
    write_json(folder / "prepared.json", {"identity": identity, "date": day, "seconds": time.monotonic() - started,
        "artifact_hashes": {str(p.relative_to(folder)): sha256_file(p) for p in folder.rglob("*") if p.is_file()}})


def worker(protocol_path, output, day, name):
    import psutil
    import torch
    from threadpoolctl import threadpool_limits

    psutil.Process().memory_info()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads(protocol_path.read_text())
    identity = json.loads((output / "frozen_screen.json").read_text())
    if identity["protocol_sha256"] != sha256_file(protocol_path):
        raise ValueError("Frozen broad-anchor worker identity changed")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen broad-anchor input changed: {path}")
    folder, target = output / "dates" / day, output / "dates" / day / "workers" / name
    prepared = json.loads((folder / "prepared.json").read_text())
    for input_name in ("training.joblib", "queries.joblib"):
        if sha256_file(folder / input_name) != prepared["artifact_hashes"][input_name]:
            raise ValueError("Frozen original training/query input changed")
    readings = {"maximum_sampled_rss_bytes": 0, "maximum_sampled_gpu_driver_bytes": 0, "maximum_sampled_simultaneous_sum_bytes": 0}
    stop = threading.Event()
    guard = threading.Thread(target=memory_guard, args=("cpu", target, protocol["operational_limits"], stop, readings), daemon=True)
    guard.start()
    begin = time.monotonic()
    training, queries = joblib.load(folder / "training.joblib"), joblib.load(folder / "queries.joblib")
    kind = MODEL_SPECS[name]
    if kind == "neural":
        model, history = fit_confirm_neural(training["features"], training["labels"], training["validation_features"], training["validation_labels"])
        model.save(target / "model")
        restored = PooledForecaster.load(target / "model")
        write_json(target / "training_history.json", history)
    else:
        with threadpool_limits(limits=2):
            model = fit_weighted_boost(training["features"], training["labels"], leaves=7, class_balanced=True)
        joblib.dump(model, target / "model.joblib")
        restored = joblib.load(target / "model.joblib")
    ready = time.monotonic() - begin
    original = None
    if day in protocol["three_date_development_subset"]:
        parent = ROOT / protocol["context_run"] / "dates" / day
        original = PooledForecaster.load(parent / name) if kind == "neural" else joblib.load(parent / f"{name}.joblib")
        if original.columns != model.columns:
            raise ValueError("Original three-date anchor feature schema changed")
        if kind == "neural":
            if any(not torch.equal(v, model.network.state_dict()[k]) for k, v in original.network.state_dict().items()):
                raise ValueError("Original three-date neural parameters must reproduce exactly")
            np.testing.assert_array_equal(original.active, model.active)
            np.testing.assert_array_equal(original.normalizer.quantiles_, model.normalizer.quantiles_)
            np.testing.assert_array_equal(original.normalizer.references_, model.normalizer.references_)
        else:
            np.testing.assert_array_equal(original.lower, model.lower)
            np.testing.assert_array_equal(original.upper, model.upper)
    forecasts, checks = {}, {}
    for asset, symbol in enumerate(SYMBOLS):
        identifier = asset if kind == "neural" else symbol
        p = model.predict_proba(queries[symbol], identifier)
        np.testing.assert_array_equal(p, restored.predict_proba(queries[symbol], identifier))
        np.testing.assert_array_equal(model.priors[asset], [(training["labels"][symbol] == c).mean() for c in (-1, 0, 1)])
        if original is not None:
            np.testing.assert_array_equal(original.priors[asset], model.priors[asset])
            np.testing.assert_array_equal(p, original.predict_proba(queries[symbol], identifier))
            with np.load(parent / "predictions" / f"{symbol}_{name}_registered.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(p, saved["probabilities"])
        prefix = queries[symbol].iloc[:128]
        before = model.predict_proba(prefix, identifier)
        later = prefix.astype(float).copy()
        later.iloc[37:] = 1e100
        future = model.predict_proba(later, identifier)
        short = model.predict_proba(prefix.iloc[:37], identifier)
        single = np.concatenate([model.predict_proba(prefix.iloc[i:i + 1], identifier) for i in range(7)])
        errors = {"future_prefix": float(np.max(np.abs(before[:37] - future[:37]))),
            "full_query_prefix": float(np.max(np.abs(p[:128] - before))),
            "shorter_prefix": float(np.max(np.abs(p[:37] - short))), "single_query": float(np.max(np.abs(p[:7] - single)))}
        if max(errors.values()) > protocol["query_probability_absolute_tolerance"]:
            raise ValueError(f"Registered broad-anchor query independence failed: {errors}")
        forecasts[symbol] = p
        checks[symbol] = {"checkpoint_forecasts_exact": True, "training_priors_exact": True,
            "original_subset_anchor_exact": True if original is not None else None, "errors": errors}
    np.savez_compressed(target / "forecasts.npz", **forecasts)
    stop.set()
    guard.join()
    write_json(target / "operational.json", {"model": name, "backend": "cpu", "dimensions": len(model.columns) + 1,
        "training_rows": sum(len(y) for y in training["labels"].values()), "fit_save_reload_ready_seconds": ready,
        "seconds": time.monotonic() - begin, "checks": checks, "memory": readings, "assessment_labels_opened": False})
    write_json(target / "succeeded.json", {"date": day, "model": name,
        "artifact_hashes": {str(p.relative_to(target)): sha256_file(p) for p in target.rglob("*") if p.is_file() and p.name != "stdout.log"}})


def supervise(protocol_path, output, day, name, protocol):
    folder = output / "dates" / day / "workers" / name
    if folder.exists():
        raise ValueError("Preserve an existing broad_accuracy worker; no implicit retry")
    folder.mkdir(parents=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--protocol", str(protocol_path), "--output", str(output), "--worker", name, "--day", day]
    begin, failure, code = time.monotonic(), None, None
    with (folder / "stdout.log").open("w") as stream:
        try:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=protocol["operational_limits"]["worker_seconds"])
            code = result.returncode
            if code:
                failure = f"Worker exited {code}"
        except subprocess.TimeoutExpired:
            failure = "Registered broad_accuracy worker deadline exceeded"
    record = {"returncode": code, "failure": failure, "seconds": time.monotonic() - begin, "log_sha256": sha256_file(folder / "stdout.log")}
    write_json(folder / "supervision.json", record)
    if failure or not (folder / "succeeded.json").exists():
        raise ValueError(f"Preserve failed broad_accuracy worker {day}/{name}: {failure}")
    verify_artifacts(folder, json.loads((folder / "succeeded.json").read_text()))
    return record


def finish_day(day, protocol, output, identity):
    folder = output / "dates" / day
    preparation = json.loads((folder / "preparation_metadata.json").read_text())
    for source in {r["run"] for r in protocol["source_specs"]}:
        parent = ROOT / source
        verify_frozen_fold(parent / "dates" / day, json.loads((parent / "frozen_study.json").read_text()))
    with np.load(folder / "assessment.npz", allow_pickle=False) as saved:
        labels = {s: saved[f"{s}_labels"].copy() for s in SYMBOLS}
        clocks = {s: saved[f"{s}_times"].copy() for s in SYMBOLS}
    for symbol in SYMBOLS:
        forecasts = {}
        for spec in protocol["source_specs"]:
            path = ROOT / spec["run"] / "dates" / day / "predictions" / f"{symbol}_{spec['prediction']}.npz"
            with np.load(path, allow_pickle=False) as saved:
                forecasts[spec["name"]] = {k: saved[k].copy() for k in saved.files}
            np.testing.assert_array_equal(forecasts[spec["name"]]["labels"], labels[symbol])
            np.testing.assert_array_equal(forecasts[spec["name"]]["decision_times"], clocks[symbol])
        common = np.asarray(preparation["population_priors"][symbol])
        for name in MODEL_SPECS:
            with np.load(folder / "workers" / name / "forecasts.npz", allow_pickle=False) as saved:
                forecasts[name] = {"probabilities": saved[symbol].copy(), "train_priors": common, "decision_priors": common}
        forecasts["observations_blend"] = {"probabilities": (forecasts["observations_tree"]["probabilities"] + forecasts["observations_neural"]["probabilities"]) / 2,
            "train_priors": common, "decision_priors": common}
        with np.load(ROOT / protocol["prior_run"] / "priors" / day / f"{symbol}_original_reference.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["decision_times"], clocks[symbol])
            initial = saved["labels_cumulative"][0].copy()
        for name, values in forecasts.items():
            for policy in protocol["decision_policies"]:
                pi = values["decision_priors"] if policy == "registered" else forecast_priors(values["probabilities"], clocks[symbol], initial, half_life_seconds=3600)
                save_predictions(folder / "predictions" / f"{symbol}_{name}_{policy}.npz", values["probabilities"], labels[symbol], clocks[symbol], values["train_priors"], pi)
                if name in BASELINES:
                    with np.load(ROOT / protocol["prior_run"] / "priors" / day / f"{symbol}_{name}.npz", allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved[policy], pi)
    write_json(folder / "completed.json", {"identity": identity, "date": day, "model_procedures": 2,
        "all_original_rows_labels_and_retained_forecasts_exact": True,
        "artifact_hashes": {str(p.relative_to(folder)): sha256_file(p) for p in folder.rglob("*") if p.is_file()}})


def reveal(output, protocol, identity):
    records = []
    for day in protocol["dates"]:
        folder = output / "dates" / day
        check_completed(folder, identity)
        for symbol in SYMBOLS:
            for name in [*BASELINES, *MODEL_SPECS, *BLENDS]:
                for policy in protocol["decision_policies"]:
                    with np.load(folder / "predictions" / f"{symbol}_{name}_{policy}.npz", allow_pickle=False) as saved:
                        metrics = classification_metrics(SimpleNamespace(priors=saved["decision_priors"]), saved["probabilities"], saved["labels"])
                    records.append({"date": day, "symbol": symbol, "model": name, "policy": policy, "metrics": metrics})
    if len(records) != protocol["post_fit_panels"]:
        raise ValueError("Incomplete broad_accuracy development family")
    board = []
    for name, policy in sorted({(r["model"], r["policy"]) for r in records}):
        rows = [r for r in records if (r["model"], r["policy"]) == (name, policy)]
        board.append({"model": name, "policy": policy, "asset_dates": len(rows),
            "means": {k: float(np.mean([r["metrics"][k] for r in rows])) for k in ("balanced_accuracy", "natural_accuracy", "log_loss", "macro_f1")},
            "balanced_accuracy_by_asset": {s: float(np.mean([r["metrics"]["balanced_accuracy"] for r in rows if r["symbol"] == s])) for s in SYMBOLS}})
    for row in board:
        matching = [r for r in records if (r["model"], r["policy"]) == (row["model"], row["policy"])]
        row["descriptive_subsets"] = {}
        for label, days in (("three_recent_development_dates", set(protocol["three_date_development_subset"])),
            ("other_seventeen_already_exposed_dates", set(protocol["dates"]) - set(protocol["three_date_development_subset"]))):
            chosen = [r for r in matching if r["date"] in days]
            row["descriptive_subsets"][label] = {"dates": sorted(days), "independent": False,
                "means": {k: float(np.mean([r["metrics"][k] for r in chosen])) for k in ("balanced_accuracy", "natural_accuracy", "log_loss", "macro_f1")},
                "balanced_accuracy_by_asset": {s: float(np.mean([r["metrics"]["balanced_accuracy"] for r in chosen if r["symbol"] == s])) for s in SYMBOLS}}
    return {"identity": identity, "evidence_status": protocol["evidence_status"], "model_procedures": len(MODEL_SPECS) * len(protocol["dates"]),
        "trainable_model_fits": 40, "post_fit_panels": len(records), "substantial_gain_confirmed": False,
        "leaderboard": sorted(board, key=lambda r: r["means"]["balanced_accuracy"], reverse=True), "records": records}


def run(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if protocol["evidence_status"] != "development_only_on_exposed_dates" or protocol["model_cases"] != list(MODEL_SPECS) or protocol["decision_policies"] != ["registered", "forecast_3600"]:
        raise ValueError("Only the frozen historical broad_accuracy family is registered")
    names = [*BASELINES, *MODEL_SPECS, *BLENDS]
    if (len(names) != len(set(names)) or len(names) != protocol["forecast_sources"]
        or list(BASELINES) != protocol["retained_sources"] or {k: list(v) for k, v in BLENDS.items()} != protocol["blend_members"]):
        raise ValueError("The complete fixed broad-anchor comparison family changed")
    if ([r["name"] for r in protocol["source_specs"]] != list(BASELINES) or len(set(protocol["dates"])) != 20
        or len(protocol["dates"]) != 20 or len(protocol["three_date_development_subset"]) != 3
        or not set(protocol["three_date_development_subset"]).issubset(protocol["dates"])):
        raise ValueError("The twenty-date cohort and four original probability sources must remain")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen broad_accuracy input changed: {path}")
    for prerequisite in protocol["prerequisite_studies"]:
        result = json.loads((ROOT / prerequisite).read_text())
        if result.get("substantial_gain_confirmed") is not False or not result.get("leaderboard"):
            raise ValueError("Registered preceding development families must be complete before this audit")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"], "python": platform.python_version(),
        "backend": "cpu", "dependencies": {n: version(n) for n in ("numpy", "pandas", "scikit-learn", "torch", "scipy", "pyarrow")}}
    if output.exists():
        raise ValueError("Preserve any existing broad_accuracy study; no implicit retry")
    write_json(output / "frozen_screen.json", identity)
    for count, day in enumerate(protocol["dates"], 1):
        write_json(output / "progress.json", {"date": day, "stage": "historical_input_verification", "assessment_scores_sealed": True})
        prepare(day, protocol, output / "dates" / day, identity)
        for name in MODEL_SPECS:
            write_json(output / "progress.json", {"date": day, "model": name, "stage": "training", "assessment_scores_sealed": True})
            result = supervise(protocol_path, output, day, name, protocol)
            print(f"broad_accuracy_fit={day}/{name} seconds={result['seconds']:.1f}", flush=True)
        finish_day(day, protocol, output, identity)
        print(f"broad_accuracy_dates_frozen={count}/{len(protocol['dates'])}", flush=True)
    write_json(output / "summary.json", reveal(output, protocol, identity))
    print(f"broad_accuracy_development_complete {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=list(MODEL_SPECS))
    parser.add_argument("--day")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.protocol.resolve(), args.output.resolve(), args.day, args.worker)
    elif args.run:
        run(args.protocol.resolve(), args.output.resolve())
    else:
        print("Pass --run for the frozen historical broad_accuracy development study.")
