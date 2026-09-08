"""Evaluate fixed-budget ordered and Langevin boosting on exposed dates."""

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
from lob_forge.boundary_combined_inputs import load_combined_inputs, original_combined_columns
from lob_forge.boundary_confirm_model import SYMBOLS
from lob_forge.boundary_event_inputs import utc_ms
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_pooled import PooledForecaster
from lob_forge.boundary_prequential import decoded_release_clock
from lob_forge.boundary_regime_coverage import calendar_stride_mask, history_cohorts
from lob_forge.boundary_stochastic_boost import VARIANTS, StochasticBoostForecaster, fit_stochastic_boost
from benchmark_boundary_tabicl_feature_budget import memory_guard
from run_boundary_confirmation import noon, save_predictions, write_json
from run_boundary_event_clock_screen import check_completed
from run_boundary_retrieval_screen import BASELINES as OLD_BASELINES, BLENDS as OLD_BLENDS, MODEL_SPECS as OLD_MODELS
from run_boundary_tabicl_screen import verify_artifacts

ROOT = Path(__file__).resolve().parents[1]
BASELINES = (*OLD_BASELINES, *OLD_MODELS, *OLD_BLENDS)
MODEL_SPECS = {f"{representation}_cat_{variant}": (representation, variant)
    for representation in ("observations", "combined") for variant in VARIANTS}
BLENDS = {f"{name}_{suffix}": (name, target)
    for name, (representation, _) in MODEL_SPECS.items()
    for suffix, target in (("old_neural", f"{representation}_neural"), ("deep500_hgb", "deep500_hgb"))}


def original_path(protocol, day, representation):
    return ROOT / protocol["context_run" if representation == "observations" else "basis_run"] / "dates" / day / f"{representation}_neural"


def prepare(day, protocol, folder, identity):
    if folder.exists():
        raise ValueError("Preserve any previous native boosting preparation attempt")
    folder.mkdir(parents=True)
    started = time.monotonic()
    training_days = history_cohorts(day)["recent4_stride4"]["train_dates"]
    validation_day = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    manifest = json.loads((ROOT / protocol["manifest"]).read_text())
    source = ROOT / protocol["baseline_run"] / "dates" / day
    check_completed(source, json.loads((source.parent.parent / "frozen_screen.json").read_text()))
    tx, ty, tt, qx, qy, qt, releases = {}, {}, {}, {}, {}, {}, []
    for current in [*training_days, day]:
        sessions = [r for r in manifest["sessions"] if r["session_date"] == current]
        if len(sessions) != 2 or {r["symbol"] for r in sessions} != set(SYMBOLS):
            raise ValueError("Both original assets required on every source date")
        partition = folder / "inputs" / f"{current}.json"
        write_json(partition, {"sessions": sessions})
        x, y, times = load_combined_inputs(ROOT, partition, ROOT / protocol["depth_manifest"], ROOT / protocol["spot_manifest"])
        for symbol in SYMBOLS:
            key = symbol, current
            keep = calendar_stride_mask(times[key], utc_ms(current), stride_seconds=4) if current in training_days else noon(times[key], current)
            raw = next(r for r in sessions if r["symbol"] == symbol)
            outcomes = pd.read_parquet(ROOT / raw["features_path"], columns=["decision_time", "label", "future_event_time"]).set_index("decision_time").loc[times[key][keep]]
            np.testing.assert_array_equal(outcomes.label.to_numpy(), y[key][keep])
            released = decoded_release_clock(outcomes.future_event_time.to_numpy())
            if current in training_days:
                if (released >= utc_ms(validation_day)).any():
                    raise ValueError("Every historical training label must be released before the preceding validation date")
                tx[key], ty[key], tt[key] = x[key].loc[keep].copy(), y[key][keep].copy(), times[key][keep].copy()
            else:
                if keep.sum() != 7070:
                    raise ValueError("All original 7070 assessment queries must remain")
                qx[symbol], qy[symbol], qt[symbol] = x[key].loc[keep].copy(), y[key][keep].copy(), times[key][keep].copy()
            releases.append({"date": current, "symbol": symbol, "rows": int(keep.sum()), "maximum_release_ms": int(released.max()),
                "historical_release_cutoff": utc_ms(validation_day) if current in training_days else None})
        del x, y, times
        gc.collect()
    tx = {s: pd.concat([tx[s, d] for d in training_days], ignore_index=True) for s in SYMBOLS}
    ty = {s: np.concatenate([ty[s, d] for d in training_days]) for s in SYMBOLS}
    tt = {s: np.concatenate([tt[s, d] for d in training_days]) for s in SYMBOLS}
    records = {}
    for representation in ("observations", "combined"):
        original = PooledForecaster.load(original_path(protocol, day, representation))
        columns = original_combined_columns(tx[SYMBOLS[0]]) if representation == "observations" else list(tx[SYMBOLS[0]].columns)
        if columns != original.columns or len(columns) != protocol["representations"][representation]:
            raise ValueError("Original native boosting feature schema changed")
        reference = joblib.load(source / representation / "training.joblib")
        for asset, symbol in enumerate(SYMBOLS):
            selected = reference["assets"] == asset
            np.testing.assert_array_equal(ty[symbol], reference["labels"][selected])
            np.testing.assert_array_equal(tt[symbol], reference["times"][selected])
            np.testing.assert_array_equal(original.matrix(tx[symbol][columns], asset), reference["matrix"][selected])
            np.testing.assert_array_equal(original.priors[asset], [(ty[symbol] == c).mean() for c in (-1, 0, 1)])
            with np.load(source / "predictions" / f"{symbol}_{representation}_neural_registered.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["labels"], qy[symbol])
                np.testing.assert_array_equal(saved["decision_times"], qt[symbol])
                np.testing.assert_array_equal(saved["probabilities"], original.predict_proba(qx[symbol][columns], asset))
                np.testing.assert_array_equal(saved["train_priors"], original.priors[asset])
        destination = folder / representation
        destination.mkdir()
        joblib.dump({"features": {s: tx[s][columns] for s in SYMBOLS}, "labels": ty, "times": tt}, destination / "training.joblib")
        joblib.dump({s: qx[s][columns] for s in SYMBOLS}, destination / "queries.joblib")
        records[representation] = {"historical_rows_labels_clocks_normalized_features_exact": True,
            "original_neural_assessment_forecasts_exact": True, "training_rows": sum(len(v) for v in ty.values()),
            "raw_dimensions": len(columns), "population_priors": {str(a): original.priors[a].tolist() for a in (0, 1)}}
        del reference, original
        gc.collect()
    np.savez_compressed(folder / "assessment.npz", **{f"{a}_labels": qy[s] for a, s in enumerate(SYMBOLS)},
        **{f"{a}_times": qt[s] for a, s in enumerate(SYMBOLS)})
    write_json(folder / "preparation_metadata.json", {"training_dates": training_days, "representations": records,
        "sources": releases, "previous_noon_used_for_training_or_selection": False})
    write_json(folder / "prepared.json", {"identity": identity, "date": day, "seconds": time.monotonic() - started,
        "artifact_hashes": {str(p.relative_to(folder)): sha256_file(p) for p in folder.rglob("*") if p.is_file()}})


def worker(protocol_path, output, day, name):
    import psutil

    psutil.Process().memory_info()
    protocol = json.loads(protocol_path.read_text())
    identity = json.loads((output / "frozen_screen.json").read_text())
    if identity["protocol_sha256"] != sha256_file(protocol_path):
        raise ValueError("Frozen native boosting worker identity changed")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen native boosting input changed: {path}")
    representation, variant = MODEL_SPECS[name]
    folder, target = output / "dates" / day, output / "dates" / day / "workers" / name
    prepared = json.loads((folder / "prepared.json").read_text())
    for input_name in (f"{representation}/training.joblib", f"{representation}/queries.joblib"):
        if sha256_file(folder / input_name) != prepared["artifact_hashes"][input_name]:
            raise ValueError("Frozen historical/query matrix changed")
    readings = {"maximum_sampled_rss_bytes": 0, "maximum_sampled_gpu_driver_bytes": 0, "maximum_sampled_simultaneous_sum_bytes": 0}
    stop = threading.Event()
    guard = threading.Thread(target=memory_guard, args=("cpu", target, protocol["operational_limits"], stop, readings), daemon=True)
    guard.start()
    begin = time.monotonic()
    training = joblib.load(folder / representation / "training.joblib")
    queries = joblib.load(folder / representation / "queries.joblib")
    learner = fit_stochastic_boost(training["features"], training["labels"], training["times"], variant=variant, iterations=protocol["iterations"])
    learner.save(target / "model")
    restored = StochasticBoostForecaster.load(target / "model")
    ready = time.monotonic() - begin
    forecasts, checks = {}, {}
    for asset, symbol in enumerate(SYMBOLS):
        np.testing.assert_array_equal(learner.priors[asset], [(training["labels"][symbol] == c).mean() for c in (-1, 0, 1)])
        p = learner.predict_proba(queries[symbol], symbol)
        np.testing.assert_array_equal(p, restored.predict_proba(queries[symbol], symbol))
        prefix = queries[symbol].iloc[:128]
        later = prefix.astype(float).copy()
        later.iloc[37:] = 1e100
        future = learner.predict_proba(later, symbol)
        short = learner.predict_proba(prefix.iloc[:37], symbol)
        single = np.concatenate([learner.predict_proba(prefix.iloc[i:i + 1], symbol) for i in range(7)])
        errors = {"future_prefix": float(np.max(np.abs(p[:37] - future[:37]))),
            "shorter_prefix": float(np.max(np.abs(p[:37] - short))), "single_query": float(np.max(np.abs(p[:7] - single)))}
        if max(errors.values()) > protocol["query_probability_absolute_tolerance"]:
            raise ValueError(f"Registered market native boosting causality/partition check failed: {errors}")
        forecasts[symbol] = p
        checks[symbol] = {"checkpoint_forecasts_exact": True, "training_priors_exact": True, "errors": errors}
    np.savez_compressed(target / "forecasts.npz", **forecasts)
    stop.set()
    guard.join()
    write_json(target / "operational.json", {"representation": representation, "variant": variant, "backend": "cpu",
        "training_rows": learner.training_rows, "dimensions": len(learner.columns) + 1,
        "fit_save_reload_ready_seconds": ready, "seconds": time.monotonic() - begin, "final_tree_count": learner.booster.tree_count_,
        "effective_parameters": learner.effective_parameters, "checks": checks, "memory": readings,
        "assessment_labels_opened": False, "previous_noon_used_for_training_or_selection": False,
        "historical_training_dates": history_cohorts(day)["recent4_stride4"]["train_dates"]})
    write_json(target / "succeeded.json", {"date": day, "model": name,
        "artifact_hashes": {str(p.relative_to(target)): sha256_file(p) for p in target.rglob("*") if p.is_file() and p.name != "stdout.log"}})


def supervise(protocol_path, output, day, name, protocol):
    folder = output / "dates" / day / "workers" / name
    if folder.exists():
        raise ValueError("Preserve an existing stochastic_boost worker; no implicit retry")
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
            failure = "Registered stochastic_boost worker deadline exceeded"
    record = {"returncode": code, "failure": failure, "seconds": time.monotonic() - begin, "log_sha256": sha256_file(folder / "stdout.log")}
    write_json(folder / "supervision.json", record)
    if failure or not (folder / "succeeded.json").exists():
        raise ValueError(f"Preserve failed stochastic_boost worker {day}/{name}: {failure}")
    verify_artifacts(folder, json.loads((folder / "succeeded.json").read_text()))
    return record


def finish_day(day, protocol, output, identity):
    folder = output / "dates" / day
    parent = ROOT / protocol["baseline_run"] / "dates" / day
    check_completed(parent, json.loads((parent.parent.parent / "frozen_screen.json").read_text()))
    preparation = json.loads((folder / "preparation_metadata.json").read_text())
    with np.load(folder / "assessment.npz", allow_pickle=False) as saved:
        labels = {a: saved[f"{a}_labels"].copy() for a in (0, 1)}
        clocks = {a: saved[f"{a}_times"].copy() for a in (0, 1)}
    for asset, symbol in enumerate(SYMBOLS):
        forecasts = {}
        for name in BASELINES:
            with np.load(parent / "predictions" / f"{symbol}_{name}_registered.npz", allow_pickle=False) as saved:
                forecasts[name] = {k: saved[k].copy() for k in saved.files}
            np.testing.assert_array_equal(forecasts[name]["labels"], labels[asset])
            np.testing.assert_array_equal(forecasts[name]["decision_times"], clocks[asset])
        common = forecasts["observations_tree"]["train_priors"]
        for representation in ("observations", "combined"):
            np.testing.assert_array_equal(common, preparation["representations"][representation]["population_priors"][str(asset)])
        for name in MODEL_SPECS:
            with np.load(folder / "workers" / name / "forecasts.npz", allow_pickle=False) as saved:
                forecasts[name] = {"probabilities": saved[symbol].copy(), "train_priors": common, "decision_priors": common}
        for name, (left, right) in BLENDS.items():
            forecasts[name] = {"probabilities": (forecasts[left]["probabilities"] + forecasts[right]["probabilities"]) / 2,
                "train_priors": common, "decision_priors": common}
        with np.load(ROOT / protocol["prior_run"] / "priors" / day / f"{symbol}_original_reference.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["decision_times"], clocks[asset])
            initial = saved["labels_cumulative"][0].copy()
        for name, values in forecasts.items():
            for policy in protocol["decision_policies"]:
                pi = values["decision_priors"] if policy == "registered" else forecast_priors(values["probabilities"], clocks[asset], initial, half_life_seconds=3600)
                save_predictions(folder / "predictions" / f"{symbol}_{name}_{policy}.npz", values["probabilities"], labels[asset], clocks[asset], values["train_priors"], pi)
                if name in BASELINES:
                    with np.load(parent / "predictions" / f"{symbol}_{name}_{policy}.npz", allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved["probabilities"], values["probabilities"])
                        np.testing.assert_array_equal(saved["decision_priors"], pi)
    write_json(folder / "completed.json", {"identity": identity, "date": day, "model_procedures": len(MODEL_SPECS),
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
        raise ValueError("Incomplete stochastic_boost development family")
    board = []
    for name, policy in sorted({(r["model"], r["policy"]) for r in records}):
        rows = [r for r in records if (r["model"], r["policy"]) == (name, policy)]
        board.append({"model": name, "policy": policy, "asset_dates": len(rows),
            "means": {k: float(np.mean([r["metrics"][k] for r in rows])) for k in ("balanced_accuracy", "natural_accuracy", "log_loss", "macro_f1")},
            "balanced_accuracy_by_asset": {s: float(np.mean([r["metrics"]["balanced_accuracy"] for r in rows if r["symbol"] == s])) for s in SYMBOLS}})
    return {"identity": identity, "evidence_status": protocol["evidence_status"], "model_procedures": len(MODEL_SPECS) * len(protocol["dates"]),
        "trainable_model_fits": 24, "post_fit_panels": len(records), "substantial_gain_confirmed": False,
        "leaderboard": sorted(board, key=lambda r: r["means"]["balanced_accuracy"], reverse=True), "records": records}


def run(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if protocol["evidence_status"] != "development_only_on_exposed_dates" or protocol["model_cases"] != list(MODEL_SPECS) or protocol["decision_policies"] != ["registered", "forecast_3600"]:
        raise ValueError("Only the frozen historical stochastic_boost family is registered")
    names = [*BASELINES, *MODEL_SPECS, *BLENDS]
    if (len(names) != len(set(names)) or len(names) != protocol["forecast_sources"] or protocol["iterations"] != 1000
        or list(BASELINES) != protocol["retained_sources"] or {k: list(v) for k, v in BLENDS.items()} != protocol["blend_members"]):
        raise ValueError("The fixed native boosting tree budget or complete comparison family changed")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen stochastic_boost input changed: {path}")
    preflight = json.loads((ROOT / protocol["preflight_summary"]).read_text())
    if preflight["all_cases_passed"] is not True or len(preflight["records"]) != 8:
        raise ValueError("All eight prospectively bounded native runtime cases must have passed")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"], "python": platform.python_version(),
        "backend": "cpu", "dependencies": {n: version(n) for n in ("numpy", "pandas", "scikit-learn", "torch", "scipy", "pyarrow", "catboost")}}
    if output.exists():
        raise ValueError("Preserve any existing stochastic_boost study; no implicit retry")
    write_json(output / "frozen_screen.json", identity)
    for count, day in enumerate(protocol["dates"], 1):
        write_json(output / "progress.json", {"date": day, "stage": "historical_input_verification", "assessment_scores_sealed": True})
        prepare(day, protocol, output / "dates" / day, identity)
        for name in MODEL_SPECS:
            write_json(output / "progress.json", {"date": day, "model": name, "stage": "training", "assessment_scores_sealed": True})
            result = supervise(protocol_path, output, day, name, protocol)
            print(f"stochastic_boost_fit={day}/{name} seconds={result['seconds']:.1f}", flush=True)
        finish_day(day, protocol, output, identity)
        print(f"stochastic_boost_dates_frozen={count}/{len(protocol['dates'])}", flush=True)
    write_json(output / "summary.json", reveal(output, protocol, identity))
    print(f"stochastic_boost_development_complete {output / 'summary.json'}", flush=True)


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
        print("Pass --run for the frozen historical stochastic_boost development study.")
