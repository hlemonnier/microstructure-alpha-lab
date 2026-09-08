"""Compare historical feature budgets and nested frozen-transformer contexts."""

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
from lob_forge.boundary_confirm_model import SYMBOLS, fit_confirm_neural
from lob_forge.boundary_event_inputs import utc_ms
from lob_forge.boundary_feature_budget import HistoricalFeatureBudget, nested_balanced_context
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_long_context_inputs import context_columns, load_context_inputs
from lob_forge.boundary_pooled import PooledForecaster
from lob_forge.boundary_prequential import decoded_release_clock
from lob_forge.boundary_regime_coverage import calendar_stride_mask, history_cohorts
from lob_forge.boundary_tabicl import case_control_posterior
from lob_forge.boundary_tabicl_backend import fit_backend_context, load_backend_context
from lob_forge.boundary_weighted_boost import fit_weighted_boost
from benchmark_boundary_tabicl_feature_budget import memory_guard
from run_boundary_confirmation import noon, save_predictions, write_json
from run_boundary_event_clock_screen import check_completed
from run_boundary_learned_memory_screen import BASELINES as OLD_BASELINES, BLENDS as OLD_BLENDS, MODEL_SPECS as OLD_MODELS
from run_boundary_tabicl_screen import verify_artifacts

ROOT = Path(__file__).resolve().parents[1]
BASELINES = (*OLD_BASELINES, *OLD_MODELS, *OLD_BLENDS)
MODEL_SPECS = {f"{variant}96_c{rows}_{learner}": (variant, rows, learner)
    for variant, rows in (("random", 3072), ("selected", 3072), ("selected", 12288)) for learner in ("tabicl", "hgb")}
MODEL_SPECS.update({f"selected96_full_{learner}": ("selected", None, learner) for learner in ("hgb", "neural")})
BLENDS = {f"{name}_{suffix}": (name, target)
    for name, (_, _, learner) in MODEL_SPECS.items() if learner == "tabicl"
    for suffix, target in (("old_neural", "observations_neural"), ("deep500_hgb", "deep500_hgb"), ("selected_full_hgb", "selected96_full_hgb"))}
BLENDS.update({"selected96_full_blend": ("selected96_full_hgb", "selected96_full_neural"),
    "selected96_full_hgb_old_neural": ("selected96_full_hgb", "observations_neural"),
    "selected96_full_neural_deep500_hgb": ("selected96_full_neural", "deep500_hgb"),
    "selected96_full_hgb_deep500_hgb": ("selected96_full_hgb", "deep500_hgb")})


def prepare(day, protocol, folder, identity):
    if (folder / "prepared.json").exists():
        record = json.loads((folder / "prepared.json").read_text())
        if record["identity"] != identity:
            raise ValueError("Prepared feature/context identity changed")
        return verify_artifacts(folder, record)
    started = time.monotonic()
    training_days = history_cohorts(day)["recent4_stride4"]["train_dates"]
    validation_day = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    manifest = json.loads((ROOT / protocol["manifest"]).read_text())
    tx, ty, tt, vx, vy, qx, qy, qt, sources = {}, {}, {}, {}, {}, {}, {}, {}, []
    for current in [*training_days, validation_day, day]:
        sessions = [r for r in manifest["sessions"] if r["session_date"] == current]
        if len(sessions) != 2 or {r["symbol"] for r in sessions} != set(SYMBOLS):
            raise ValueError("Both original assets required on every source day")
        partition = folder / "inputs" / f"{current}.json"
        write_json(partition, {"sessions": sessions})
        x, y, clocks = load_context_inputs(ROOT, partition)
        for symbol in SYMBOLS:
            key = symbol, current
            columns = context_columns(x[key], "observations")
            if len(columns) != 220:
                raise ValueError("Exactly the original observation-context features required")
            keep = calendar_stride_mask(clocks[key], utc_ms(current), stride_seconds=4) if current in training_days else noon(clocks[key], current)
            raw = next(r for r in sessions if r["symbol"] == symbol)
            released = pd.read_parquet(ROOT / raw["features_path"], columns=["decision_time", "label", "future_event_time"]).set_index("decision_time").loc[clocks[key][keep]]
            np.testing.assert_array_equal(released.label.to_numpy(), y[key][keep])
            release = decoded_release_clock(released.future_event_time.to_numpy())
            cutoff = utc_ms(validation_day if current in training_days else day)
            if current != day and (release >= cutoff).any():
                raise ValueError("Actual historical and validation label releases must precede the next partition")
            frame = x[key].loc[keep, columns].copy()
            if current in training_days:
                tx[key], ty[key], tt[key] = frame, y[key][keep].copy(), clocks[key][keep].copy()
            else:
                if keep.sum() != 7070:
                    raise ValueError("Every original noon observation must remain eligible")
                if current == validation_day:
                    vx[symbol], vy[symbol] = frame, y[key][keep].copy()
                else:
                    qx[symbol], qy[symbol], qt[symbol] = frame, y[key][keep].copy(), clocks[key][keep].copy()
            sources.append({"date": current, "symbol": symbol, "selected_rows": int(keep.sum()),
                "maximum_selected_release_ms": int(release.max()), "past_release_cutoff": cutoff if current != day else None})
        del x, y, clocks, frame
        gc.collect()
    tx = {s: pd.concat([tx[s, d] for d in training_days], ignore_index=True) for s in SYMBOLS}
    ty = {s: np.concatenate([ty[s, d] for d in training_days]) for s in SYMBOLS}
    tt = {s: np.concatenate([tt[s, d] for d in training_days]) for s in SYMBOLS}
    selected, population, sampled = nested_balanced_context(ty)
    original_folder = ROOT / protocol["original_tabicl_run"] / "dates" / day
    original_context = joblib.load(original_folder / "context_recent4_stride4.joblib")
    old_details = json.loads((original_folder / "context_metadata.json").read_text())["recent4_stride4"]["assets"]
    for symbol in SYMBOLS:
        take = selected[3072][symbol]
        pd.testing.assert_frame_equal(tx[symbol].iloc[take].reset_index(drop=True),
            original_context["features"][symbol][list(tx[symbol].columns)].reset_index(drop=True), check_exact=True)
        np.testing.assert_array_equal(ty[symbol][take], original_context["labels"][symbol])
        np.testing.assert_array_equal(tt[symbol][take], old_details[symbol]["selected_decision_times"])
        np.testing.assert_array_equal(population[symbol], original_context["population_priors"][symbol])
    selector = HistoricalFeatureBudget.fit(tx, ty)
    selector.save(folder / "feature_selector.json")
    restored = HistoricalFeatureBudget.load(folder / "feature_selector.json")
    for variant, rows in (("random", 3072), ("selected", 3072), ("selected", 12288)):
        features = {s: selector.transform(tx[s].iloc[selected[rows][s]], variant) for s in SYMBOLS}
        for symbol in SYMBOLS:
            pd.testing.assert_frame_equal(features[symbol], restored.transform(tx[symbol].iloc[selected[rows][symbol]], variant), check_exact=True)
        joblib.dump({"features": features, "labels": {s: ty[s][selected[rows][s]] for s in SYMBOLS},
            "population_priors": population, "sampled_priors": sampled}, folder / f"context_{variant}_{rows}.joblib")
    joblib.dump({"features": {s: selector.transform(tx[s], "selected") for s in SYMBOLS}, "labels": ty,
        "validation_features": {s: selector.transform(vx[s], "selected") for s in SYMBOLS}, "validation_labels": vy,
        "population_priors": population}, folder / "selected_full.joblib")
    joblib.dump({variant: {s: selector.transform(qx[s], variant) for s in SYMBOLS} for variant in ("random", "selected")}, folder / "query_features.joblib")
    np.savez_compressed(folder / "assessment.npz", **{f"{s}_labels": qy[s] for s in SYMBOLS}, **{f"{s}_times": qt[s] for s in SYMBOLS})
    write_json(folder / "preparation_metadata.json", {"training_dates": training_days, "validation_date": validation_day,
        "sources": sources, "original_context_rows_labels_times_features_exact": True,
        "selector_uses_full_historical_labels": True, "population_priors": {s: population[s].tolist() for s in SYMBOLS},
        "context_selected_times": {str(n): {s: tt[s][indices[s]].tolist() for s in SYMBOLS} for n, indices in selected.items()},
        "feature_budget": 96, "original_columns": 220, "transformer_columns_including_asset": 97})
    record = {"identity": identity, "date": day, "seconds": time.monotonic() - started,
        "artifact_hashes": {str(p.relative_to(folder)): sha256_file(p) for p in folder.rglob("*") if p.is_file()}}
    write_json(folder / "prepared.json", record)
    return record


def worker(protocol_path, output, day, name):
    import psutil
    import torch

    # Fail in the main worker before fitting if its monitor/runtime is missing.
    psutil.Process().memory_info()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads(protocol_path.read_text())
    identity = json.loads((output / "frozen_screen.json").read_text())
    if identity["protocol_sha256"] != sha256_file(protocol_path):
        raise ValueError("Frozen worker identity changed")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen worker input changed: {path}")
    variant, rows, learner = MODEL_SPECS[name]
    folder = output / "dates" / day
    target = folder / "workers" / name
    prepared = json.loads((folder / "prepared.json").read_text())
    training_file = f"context_{variant}_{rows}.joblib" if rows else "selected_full.joblib"
    # This worker verifies/opens training and label-free query inputs only.
    for input_name in (training_file, "query_features.joblib"):
        if sha256_file(folder / input_name) != prepared["artifact_hashes"][input_name]:
            raise ValueError("Frozen training/query file changed")
    backend = protocol["selected_backend"] if learner == "tabicl" else "cpu"
    readings = {"maximum_sampled_rss_bytes": 0, "maximum_sampled_gpu_driver_bytes": 0, "maximum_sampled_simultaneous_sum_bytes": 0}
    stop = threading.Event()
    guard = threading.Thread(target=memory_guard, args=(backend, target, protocol["operational_limits"], stop, readings), daemon=True)
    guard.start()
    started = time.monotonic()
    context, queries = joblib.load(folder / training_file), joblib.load(folder / "query_features.joblib")[variant]
    history = {}
    if learner == "tabicl":
        model = fit_backend_context(context["features"], context["labels"], context["population_priors"],
            checkpoint=ROOT / protocol["checkpoint"], checkpoint_sha256=protocol["checkpoint_sha256"], backend=backend)
        model.save(target / "model")
        restored = load_backend_context(target / "model", backend=backend)
    elif learner == "hgb":
        model = fit_weighted_boost(context["features"], context["labels"], leaves=7, class_balanced=True)
        joblib.dump(model, target / "model.joblib")
        restored = joblib.load(target / "model.joblib")
    else:
        model, history = fit_confirm_neural(context["features"], context["labels"], context["validation_features"], context["validation_labels"])
        model.save(target / "model")
        restored = PooledForecaster.load(target / "model")
        write_json(target / "training_history.json", history)
    ready = time.monotonic() - started

    def predict(estimator, frame, symbol, *, single=False):
        if learner == "tabicl":
            return estimator.predict_proba(frame, symbol, batch_size=1 if single else 64)
        asset = SYMBOLS.index(symbol)
        p = estimator.predict_proba(frame, asset, batch_size=1 if single else 2048) if learner == "neural" else estimator.predict_proba(frame, symbol)
        return case_control_posterior(p, estimator.priors[asset], context["population_priors"][symbol]) if rows else p

    forecasts, checks = {}, {}
    for symbol in SYMBOLS:
        p = predict(model, queries[symbol], symbol)
        np.testing.assert_array_equal(p, predict(restored, queries[symbol], symbol))
        prefix = queries[symbol].iloc[:64]
        before = predict(model, prefix, symbol)
        later = prefix.astype(float).copy()
        later.iloc[17:] = 1e100
        changed = predict(model, later, symbol)
        short = predict(model, prefix.iloc[:17], symbol)
        single = predict(model, prefix.iloc[:7], symbol, single=True)
        errors = {"future_prefix": float(np.max(np.abs(before[:17] - changed[:17]))),
            "shorter_prefix": float(np.max(np.abs(before[:17] - short))),
            "single_query": float(np.max(np.abs(before[:7] - single))),
            "full_query_prefix": float(np.max(np.abs(before - p[:64])))}
        if errors["future_prefix"] > protocol["future_prefix_absolute_tolerance"] or max(errors[k] for k in ("shorter_prefix", "single_query", "full_query_prefix")) > protocol["query_partition_absolute_tolerance"]:
            raise ValueError(f"Registered market query causality failed: {errors}")
        forecasts[symbol] = p
        checks[symbol] = {"checkpoint_forecasts_exact": True, "errors": errors}
    stop.set()
    guard.join()
    np.savez_compressed(target / "forecasts.npz", **forecasts)
    write_json(target / "operational.json", {"selector": variant, "context_rows": rows, "learner": learner, "backend": backend,
        "training_rows": {s: len(context["labels"][s]) for s in SYMBOLS}, "columns": list(context["features"][SYMBOLS[0]].columns),
        "fit_save_reload_ready_seconds": ready, "seconds": time.monotonic() - started, "checks": checks,
        "best_epoch": history.get("best_epoch"), "memory": dict(readings), "assessment_labels_opened": False})
    write_json(target / "succeeded.json", {"date": day, "model": name,
        "artifact_hashes": {str(p.relative_to(target)): sha256_file(p) for p in target.rglob("*") if p.is_file() and p.name != "stdout.log"}})


def supervise(protocol_path, output, day, name, protocol):
    folder = output / "dates" / day / "workers" / name
    if folder.exists():
        raise ValueError("Preserve an existing worker attempt; no implicit retry is allowed")
    folder.mkdir(parents=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--protocol", str(protocol_path), "--output", str(output), "--worker", name, "--day", day]
    started, failure, code = time.monotonic(), None, None
    with (folder / "stdout.log").open("w") as stream:
        try:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=protocol["operational_limits"]["worker_seconds"])
            code = result.returncode
            if code:
                failure = f"Worker exited {code}"
        except subprocess.TimeoutExpired:
            failure = "Registered worker deadline exceeded"
    record = {"returncode": code, "failure": failure, "seconds": time.monotonic() - started, "log_sha256": sha256_file(folder / "stdout.log")}
    write_json(folder / "supervision.json", record)
    if failure or not (folder / "succeeded.json").exists():
        raise ValueError(f"Preserve failed feature-budget worker {day}/{name}: {failure}")
    verify_artifacts(folder, json.loads((folder / "succeeded.json").read_text()))
    return record


def finish_day(day, protocol, output, identity):
    folder = output / "dates" / day
    parent = ROOT / protocol["baseline_run"] / "dates" / day
    check_completed(parent, json.loads((parent.parent.parent / "frozen_screen.json").read_text()))
    population = json.loads((folder / "preparation_metadata.json").read_text())["population_priors"]
    with np.load(folder / "assessment.npz", allow_pickle=False) as saved:
        labels = {s: saved[f"{s}_labels"].copy() for s in SYMBOLS}
        clocks = {s: saved[f"{s}_times"].copy() for s in SYMBOLS}
    for symbol in SYMBOLS:
        forecasts = {}
        for name in BASELINES:
            with np.load(parent / "predictions" / f"{symbol}_{name}_registered.npz", allow_pickle=False) as saved:
                forecasts[name] = {k: saved[k].copy() for k in saved.files}
            np.testing.assert_array_equal(labels[symbol], forecasts[name]["labels"])
            np.testing.assert_array_equal(clocks[symbol], forecasts[name]["decision_times"])
        common = forecasts["observations_tree"]["train_priors"]
        np.testing.assert_array_equal(common, population[symbol])
        for name in MODEL_SPECS:
            with np.load(folder / "workers" / name / "forecasts.npz", allow_pickle=False) as saved:
                forecasts[name] = {"probabilities": saved[symbol].copy(), "train_priors": common, "decision_priors": common}
        for name, (left, right) in BLENDS.items():
            forecasts[name] = {"probabilities": (forecasts[left]["probabilities"] + forecasts[right]["probabilities"]) / 2,
                "train_priors": common, "decision_priors": common}
        with np.load(ROOT / protocol["prior_run"] / "priors" / day / f"{symbol}_original_reference.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["decision_times"], clocks[symbol])
            initial = saved["labels_cumulative"][0].copy()
        for name, values in forecasts.items():
            for policy in protocol["decision_policies"]:
                pi = values["decision_priors"] if policy == "registered" else forecast_priors(values["probabilities"], clocks[symbol], initial, half_life_seconds=3600)
                save_predictions(folder / "predictions" / f"{symbol}_{name}_{policy}.npz", values["probabilities"], labels[symbol], clocks[symbol], values["train_priors"], pi)
                if name in BASELINES:
                    with np.load(parent / "predictions" / f"{symbol}_{name}_{policy}.npz", allow_pickle=False) as old:
                        np.testing.assert_array_equal(old["probabilities"], values["probabilities"])
                        np.testing.assert_array_equal(old["decision_priors"], pi)
    write_json(folder / "completed.json", {"identity": identity, "date": day, "model_fits": len(MODEL_SPECS),
        "all_original_rows_labels_priors_and_saved_controls_exact": True,
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
        raise ValueError("Incomplete feature-budget development family")
    board = []
    for name, policy in sorted({(r["model"], r["policy"]) for r in records}):
        rows = [r for r in records if (r["model"], r["policy"]) == (name, policy)]
        board.append({"model": name, "policy": policy, "asset_dates": len(rows),
            "means": {k: float(np.mean([r["metrics"][k] for r in rows])) for k in ("balanced_accuracy", "natural_accuracy", "log_loss", "macro_f1")},
            "balanced_accuracy_by_asset": {s: float(np.mean([r["metrics"]["balanced_accuracy"] for r in rows if r["symbol"] == s])) for s in SYMBOLS}})
    return {"identity": identity, "evidence_status": protocol["evidence_status"], "model_fits": len(MODEL_SPECS) * len(protocol["dates"]),
        "post_fit_panels": len(records), "substantial_gain_confirmed": False,
        "leaderboard": sorted(board, key=lambda r: r["means"]["balanced_accuracy"], reverse=True), "records": records}


def run(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if protocol["evidence_status"] != "development_only_on_exposed_dates" or protocol["model_cases"] != list(MODEL_SPECS) or protocol["decision_policies"] != ["registered", "forecast_3600"]:
        raise ValueError("Only the frozen feature-budget/context family is registered")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen feature-budget input changed: {path}")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"], "python": platform.python_version(),
        "backend": protocol["selected_backend"], "dependencies": {n: version(n) for n in ("numpy", "pandas", "scikit-learn", "torch", "scipy", "pyarrow")}}
    if output.exists():
        raise ValueError("Preserve an existing feature-budget experiment; no implicit rerun")
    write_json(output / "frozen_screen.json", identity)
    for count, day in enumerate(protocol["dates"], 1):
        folder = output / "dates" / day
        write_json(output / "progress.json", {"date": day, "stage": "historical_feature_selection", "assessment_scores_sealed": True})
        prepare(day, protocol, folder, identity)
        for name in MODEL_SPECS:
            write_json(output / "progress.json", {"date": day, "model": name, "stage": "training", "assessment_scores_sealed": True})
            result = supervise(protocol_path, output, day, name, protocol)
            print(f"feature_budget_fit={day}/{name} seconds={result['seconds']:.1f}", flush=True)
        finish_day(day, protocol, output, identity)
        print(f"feature_budget_dates_frozen={count}/{len(protocol['dates'])}", flush=True)
    write_json(output / "summary.json", reveal(output, protocol, identity))
    print(f"feature_budget_development_complete {output / 'summary.json'}", flush=True)


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
        print("Pass --run for the frozen feature-budget/context development family.")
