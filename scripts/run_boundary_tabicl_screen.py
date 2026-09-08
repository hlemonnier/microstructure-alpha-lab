"""Frozen historical/morning in-context forecasts on exposed development dates."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
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
from lob_forge.boundary_event_inputs import utc_ms
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_morning_adaptation import morning_masks
from lob_forge.boundary_regime_coverage import calendar_stride_mask, history_cohorts
from lob_forge.boundary_tabicl import SYMBOLS, TABICL_SETTINGS, TabiclForecaster, case_control_posterior, fit_tabicl_context, select_balanced_context
from lob_forge.boundary_tabicl_preprocessing import fallback_counts
from lob_forge.boundary_weighted_boost import fit_weighted_boost
from run_boundary_confirmation import noon, save_predictions, write_json
from run_boundary_event_clock_screen import check_completed
from run_boundary_multihorizon_screen import BASELINES as OLD_BASELINES, BLENDS as OLD_BLENDS, MODEL_SPECS as OLD_MODELS

ROOT = Path(__file__).resolve().parents[1]
BASELINES = (*OLD_BASELINES, *OLD_MODELS, *OLD_BLENDS)
MODEL_SPECS = {f"{history}_{representation}_{algorithm}": (history, representation, algorithm)
    for history in ("recent4_stride4", "full_morning") for representation in ("observations", "combined")
    for algorithm in ("hgb", "tabicl")}
BLENDS = {f"{name}_{suffix}": (name, "observations_tree" if representation == "observations" else "combined_hgb")
    if suffix == "matched_hgb" else (name, suffix)
    for name, (_, representation, algorithm) in MODEL_SPECS.items() if algorithm == "tabicl"
    for suffix in ("matched_hgb", "deep500_hgb", "observations_neural")}


def verify_artifacts(folder, record):
    for name, checksum in record["artifact_hashes"].items():
        if sha256_file(folder / name) != checksum:
            raise ValueError(f"Frozen local context artifact changed: {name}")
    return record


def prepare_day(day, protocol, folder, identity):
    marker = folder / "prepared.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        if record["identity"] != identity:
            raise ValueError("Frozen context preparation identity changed")
        return verify_artifacts(folder, record)
    started = time.monotonic()
    training_days = history_cohorts(day)["recent4_stride4"]["train_dates"]
    cutoff = utc_ms((date.fromisoformat(day) - timedelta(days=1)).isoformat())
    manifest = json.loads((ROOT / protocol["manifest"]).read_text())
    pools_x, pools_y, pools_t, pools_release = {}, {}, {}, {}
    queries, assessment_y, assessment_t = {}, {}, {}
    for current in [*training_days, day]:
        sessions = [r for r in manifest["sessions"] if r["session_date"] == current]
        if len(sessions) != 2 or {r["symbol"] for r in sessions} != set(SYMBOLS):
            raise ValueError("Both exact-label assets required on every source date")
        partition = folder / "inputs" / f"{current}.json"
        write_json(partition, {"sessions": sessions})
        x, y, clocks = load_combined_inputs(ROOT, partition, ROOT / protocol["depth_manifest"], ROOT / protocol["spot_manifest"])
        for symbol in SYMBOLS:
            key = symbol, current
            raw = next(r for r in sessions if r["symbol"] == symbol)
            resolved = pd.read_parquet(ROOT / raw["features_path"], columns=["decision_time", "label", "future_event_time"]).set_index("decision_time").loc[clocks[key]]
            np.testing.assert_array_equal(resolved.label.to_numpy(), y[key])
            release = resolved.future_event_time.to_numpy()
            if current in training_days:
                mask = calendar_stride_mask(clocks[key], utc_ms(current), stride_seconds=4)
                if (release[mask] >= cutoff).any():
                    raise ValueError("Historical labels must be released before the previous day")
                pool_key = "recent4_stride4", symbol, current
            else:
                mask, _ = morning_masks(clocks[key], release, utc_ms(day), window="full_morning")
                pool_key = "full_morning", symbol, current
                test = noon(clocks[key], day)
                if test.sum() != 7070:
                    raise ValueError("The unchanged 7070-row assessment universe is required")
                queries[symbol] = x[key].loc[test].copy()
                assessment_y[symbol], assessment_t[symbol] = y[key][test].copy(), clocks[key][test].copy()
            pools_x[pool_key], pools_y[pool_key] = x[key].loc[mask].copy(), y[key][mask].copy()
            pools_t[pool_key], pools_release[pool_key] = clocks[key][mask].copy(), release[mask].copy()
        del x, y, clocks, resolved
        gc.collect()
    details = {}
    for history in ("recent4_stride4", "full_morning"):
        days = training_days if history == "recent4_stride4" else [day]
        context_x = {s: pd.concat([pools_x[history, s, d] for d in days], ignore_index=True) for s in SYMBOLS}
        context_y = {s: np.concatenate([pools_y[history, s, d] for d in days]) for s in SYMBOLS}
        context_t = {s: np.concatenate([pools_t[history, s, d] for d in days]) for s in SYMBOLS}
        context_release = {s: np.concatenate([pools_release[history, s, d] for d in days]) for s in SYMBOLS}
        selected, population, sampled = select_balanced_context(context_y, requested_rows=protocol["context_budget_rows"])
        chosen = {s: context_x[s].iloc[selected[s]].copy() for s in SYMBOLS}
        for symbol in SYMBOLS:
            if len(np.unique(context_t[symbol][selected[symbol]])) != len(selected[symbol]):
                raise ValueError("Context decisions must be unique for each asset")
            if len(chosen[symbol].columns) != 399 or len(original_combined_columns(chosen[symbol])) != 220:
                raise ValueError("The registered observed schemas must be unchanged")
        joblib.dump({"features": chosen, "labels": {s: context_y[s][selected[s]] for s in SYMBOLS},
            "population_priors": population, "sampled_priors": sampled}, folder / f"context_{history}.joblib")
        details[history] = {"training_dates": days, "selection_seed": 20260908, "assets": {s: {
            "pool_rows": len(context_y[s]), "pool_class_counts": [int((context_y[s] == k).sum()) for k in (-1, 0, 1)],
            "sampled_rows": len(selected[s]), "sampled_class_counts": [int((context_y[s][selected[s]] == k).sum()) for k in (-1, 0, 1)],
            "population_prior": population[s].tolist(), "sampled_prior": sampled[s].tolist(),
            "selected_decision_times": context_t[s][selected[s]].tolist(),
            "latest_pool_label_release_ms": int(context_release[s].max()),
            "latest_sampled_label_release_ms": int(context_release[s][selected[s]].max())} for s in SYMBOLS}}
        del context_x, context_y, context_t, context_release, chosen
        gc.collect()
    joblib.dump(queries, folder / "query_features.joblib")
    np.savez_compressed(folder / "assessment.npz", **{f"{s}_labels": assessment_y[s] for s in SYMBOLS},
                        **{f"{s}_times": assessment_t[s] for s in SYMBOLS})
    write_json(folder / "context_metadata.json", details)
    paths = [*folder.glob("context_*.joblib"), folder / "query_features.joblib", folder / "assessment.npz", folder / "context_metadata.json", *folder.glob("inputs/*.json")]
    record = {"identity": identity, "date": day, "seconds": time.monotonic() - started,
        "artifact_hashes": {str(p.relative_to(folder)): sha256_file(p) for p in paths}}
    write_json(marker, record)
    return record


def train_and_forecast(features, labels, population, queries, algorithm, folder, protocol):
    started = time.monotonic()
    if algorithm == "tabicl":
        model = fit_tabicl_context(features, labels, population, checkpoint=ROOT / protocol["checkpoint"], checkpoint_sha256=protocol["checkpoint_sha256"])
        fit_seconds = time.monotonic() - started
        model.save(folder / "model")
    elif algorithm == "hgb":
        model = fit_weighted_boost(features, labels, leaves=7, class_balanced=True)
        fit_seconds = time.monotonic() - started
        joblib.dump(model, folder / "model.joblib")
    else:
        raise ValueError("Only the frozen transformer and matched HGB are registered")

    def predict(learner, frame, symbol):
        if algorithm == "tabicl":
            return learner.predict_proba(frame, symbol, batch_size=64)
        asset = learner.symbols.index(symbol)
        return case_control_posterior(learner.predict_proba(frame, symbol), learner.priors[asset], population[symbol])

    serialized_seconds = time.monotonic() - started
    # Reload before any query checks. This measures readiness independent of
    # later query values; their invariance is checked separately below.
    restored = TabiclForecaster.load(folder / "model") if algorithm == "tabicl" else joblib.load(folder / "model.joblib")
    ready_seconds = time.monotonic() - started
    checks = {}
    for symbol in SYMBOLS:
        prefix = queries[symbol].iloc[:64]
        p = predict(model, prefix, symbol)
        reloaded = predict(restored, prefix, symbol)
        np.testing.assert_array_equal(reloaded, p)
        changed = prefix.copy()
        changed.iloc[16:] = -7 * changed.iloc[16:].to_numpy() + 123
        later = predict(restored, changed, symbol)
        shorter = predict(restored, prefix.iloc[:16], symbol)
        np.testing.assert_allclose(later[:16], p[:16], rtol=0, atol=2e-6)
        np.testing.assert_allclose(shorter, p[:16], rtol=0, atol=2e-5)
        begin = time.monotonic()
        single = predict(restored, prefix.iloc[:1], symbol)
        single_seconds = time.monotonic() - begin
        np.testing.assert_allclose(single, p[:1], rtol=0, atol=2e-5)
        checks[symbol] = {"checkpoint_forecast_exact": True,
            "later_query_max_error": float(np.max(np.abs(later[:16] - p[:16]))),
            "shorter_query_max_error": float(np.max(np.abs(shorter - p[:16]))),
            "single_query_max_error": float(np.max(np.abs(single - p[:1]))), "single_query_seconds": single_seconds}
    del model
    gc.collect()
    if algorithm == "tabicl":
        fallback_counts(restored.estimator, reset=True)
    begin = time.monotonic()
    forecasts = {s: predict(restored, queries[s], s) for s in SYMBOLS}
    prediction_seconds = time.monotonic() - begin
    metadata = {"algorithm": algorithm, "fit_and_cache_seconds": fit_seconds, "fit_and_serialization_seconds": serialized_seconds,
        "fit_save_reload_ready_seconds": ready_seconds, "full_universe_query_seconds": prediction_seconds,
        "query_rows": sum(len(queries[s]) for s in SYMBOLS), "checks": checks,
        "posterior_contract": "sampled_context_to_full_context_pool_prior_correction",
        "population_priors": {s: np.asarray(population[s]).tolist() for s in SYMBOLS},
        "actual_query_power_fallback_rows": fallback_counts(restored.estimator) if algorithm == "tabicl" else {},
        "model_procedures": 1, "optimizer_training": algorithm == "hgb"}
    return forecasts, metadata


def worker(protocol_path, day, name, output):
    protocol = json.loads(protocol_path.read_text())
    identity = json.loads((output / "frozen_screen.json").read_text())
    if identity["protocol_sha256"] != sha256_file(protocol_path) or identity["input_hashes"] != protocol["input_hashes"]:
        raise ValueError("Frozen worker protocol identity changed")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen worker input changed: {path}")
    folder = output / "dates" / day
    history, representation, algorithm = MODEL_SPECS[name]
    prepared = json.loads((folder / "prepared.json").read_text())
    # Workers validate only their label-free query inputs and training context;
    # the assessment-label file is never opened by this process.
    for input_name in (f"context_{history}.joblib", "query_features.joblib"):
        if sha256_file(folder / input_name) != prepared["artifact_hashes"][input_name]:
            raise ValueError("Frozen training/query input changed")
    context = joblib.load(folder / f"context_{history}.joblib")
    queries = joblib.load(folder / "query_features.joblib")
    columns = original_combined_columns(context["features"][SYMBOLS[0]]) if representation == "observations" else list(context["features"][SYMBOLS[0]].columns)
    features = {s: context["features"][s][columns] for s in SYMBOLS}
    queries = {s: queries[s][columns] for s in SYMBOLS}
    target = folder / "workers" / name
    forecasts, metadata = train_and_forecast(features, context["labels"], context["population_priors"], queries, algorithm, target, protocol)
    metadata.update({"history": history, "representation": representation, "columns": columns,
        "context_rows": {s: len(context["labels"][s]) for s in SYMBOLS},
        "offline_preparation_plus_model_readiness_seconds": prepared["seconds"] + metadata["fit_save_reload_ready_seconds"],
        "morning_ready_gap_seconds": 1920,
        "readiness_scope": "Measured offline preparation/cache/serialization work, compared with the morning gap; not a live latency certification. Assessment-feature batch processing and causality verification occur after model readiness."})
    np.savez_compressed(target / "forecasts.npz", **forecasts)
    write_json(target / "operational.json", metadata)
    paths = [p for p in target.rglob("*") if p.is_file() and p.name not in ("stdout.log", "launch.json")]
    write_json(target / "succeeded.json", {"day": day, "name": name,
        "artifact_hashes": {str(p.relative_to(target)): sha256_file(p) for p in paths}})


def supervise(day, name, protocol_path, protocol, output):
    import psutil

    folder = output / "dates" / day / "workers" / name
    marker = folder / "succeeded.json"
    if marker.exists():
        verify_artifacts(folder, json.loads(marker.read_text()))
        record = json.loads((folder / "supervision.json").read_text())
        if record["returncode"] != 0 or record["stop_reason"] is not None:
            raise ValueError("A failed worker cannot be resumed as a completed model")
        return record
    if folder.exists():
        raise ValueError("Preserve the earlier incomplete context procedure and register any retry separately")
    folder.mkdir(parents=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--protocol", str(protocol_path), "--output", str(output), "--worker", name, "--day", day]
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"}
    started, peak, reason = time.monotonic(), 0, None
    with (folder / "stdout.log").open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
        write_json(folder / "launch.json", {"pid": process.pid, "command": command})
        monitored = psutil.Process(process.pid)
        try:
            while process.poll() is None:
                try:
                    peak = max(peak, monitored.memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
                if peak > protocol["operational_limits"]["worker_rss_bytes"]:
                    reason = "registered_worker_rss_limit"
                elif time.monotonic() - started > protocol["operational_limits"]["worker_seconds"]:
                    reason = "registered_worker_time_limit"
                if reason:
                    process.kill()
                    break
                time.sleep(.2)
        finally:
            if process.poll() is None:
                process.kill()
            code = process.wait()
    record = {"returncode": code, "stop_reason": reason, "seconds": time.monotonic() - started,
        "peak_observed_worker_rss_bytes": peak, "log_sha256": sha256_file(folder / "stdout.log")}
    write_json(folder / "supervision.json", record)
    if code or reason or not marker.exists():
        raise ValueError(f"Context procedure {day}/{name} did not complete; preserve its artifacts")
    verify_artifacts(folder, json.loads(marker.read_text()))
    return record


def finish_day(day, protocol, output, identity, supervision):
    folder = output / "dates" / day
    parent = ROOT / protocol["baseline_run"] / "dates" / day
    check_completed(parent, json.loads((parent.parent.parent / "frozen_screen.json").read_text()))
    context_details = json.loads((folder / "context_metadata.json").read_text())
    with np.load(folder / "assessment.npz", allow_pickle=False) as saved:
        labels = {s: saved[f"{s}_labels"].copy() for s in SYMBOLS}
        clocks = {s: saved[f"{s}_times"].copy() for s in SYMBOLS}
    for symbol in SYMBOLS:
        forecasts = {}
        for name in BASELINES:
            with np.load(parent / "predictions" / f"{symbol}_{name}_registered.npz", allow_pickle=False) as saved:
                forecasts[name] = {k: saved[k].copy() for k in saved.files}
            np.testing.assert_array_equal(forecasts[name]["labels"], labels[symbol])
            np.testing.assert_array_equal(forecasts[name]["decision_times"], clocks[symbol])
        common_prior = forecasts["observations_tree"]["train_priors"]
        for name, (history, _, _) in MODEL_SPECS.items():
            with np.load(folder / "workers" / name / "forecasts.npz", allow_pickle=False) as saved:
                probabilities = saved[symbol].copy()
            population = np.array(context_details[history]["assets"][symbol]["population_prior"])
            if history == "recent4_stride4":
                np.testing.assert_array_equal(population, common_prior)
            forecasts[name] = {"probabilities": probabilities, "train_priors": population, "decision_priors": common_prior}
        for name, (left, right) in BLENDS.items():
            forecasts[name] = {"probabilities": (forecasts[left]["probabilities"] + forecasts[right]["probabilities"]) / 2,
                "train_priors": common_prior, "decision_priors": common_prior}
        with np.load(ROOT / protocol["prior_run"] / "priors" / day / f"{symbol}_original_reference.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["decision_times"], clocks[symbol])
            initial = saved["labels_cumulative"][0].copy()
        for name, values in forecasts.items():
            for policy in protocol["decision_policies"]:
                pi = values["decision_priors"] if policy == "registered" else forecast_priors(values["probabilities"], clocks[symbol], initial, half_life_seconds=3600)
                save_predictions(folder / "predictions" / f"{symbol}_{name}_{policy}.npz", values["probabilities"], labels[symbol], clocks[symbol], values["train_priors"], pi)
                if name in BASELINES:
                    with np.load(parent / "predictions" / f"{symbol}_{name}_{policy}.npz", allow_pickle=False) as original:
                        np.testing.assert_array_equal(original["probabilities"], values["probabilities"])
                        np.testing.assert_array_equal(original["decision_priors"], pi)
    write_json(folder / "worker_summary.json", supervision)
    write_json(folder / "completed.json", {"identity": identity, "date": day, "model_procedures": len(MODEL_SPECS),
        "all_original_rows_labels_and_saved_controls_exact": True,
        "blend_prior_scope": "Each constituent retains its own pool prior in context_metadata/model metadata. Fixed blends combine natural probability estimates; the legacy train_priors NPZ field is only the common historical reference, not a claim of one joint training cohort.",
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
        raise ValueError("Incomplete frozen in-context comparison")
    board = []
    for name, policy in sorted({(r["model"], r["policy"]) for r in records}):
        rows = [r for r in records if (r["model"], r["policy"]) == (name, policy)]
        board.append({"model": name, "policy": policy, "asset_dates": len(rows),
            "means": {k: float(np.mean([r["metrics"][k] for r in rows])) for k in ("balanced_accuracy", "natural_accuracy", "log_loss", "macro_f1")},
            "balanced_accuracy_by_asset": {s: float(np.mean([r["metrics"]["balanced_accuracy"] for r in rows if r["symbol"] == s])) for s in SYMBOLS}})
    return {"identity": identity, "evidence_status": protocol["evidence_status"], "tabicl_context_procedures": 12, "matched_hgb_fits": 12,
        "post_fit_panels": len(records), "substantial_gain_confirmed": False,
        "leaderboard": sorted(board, key=lambda r: r["means"]["balanced_accuracy"], reverse=True), "records": records}


def run(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if protocol["evidence_status"] != "development_only_on_exposed_dates" or protocol["model_procedures"] != list(MODEL_SPECS) or protocol["decision_policies"] != ["registered", "forecast_3600"]:
        raise ValueError("Only the frozen exposed-date in-context family is registered")
    for key, value in TABICL_SETTINGS.items():
        source_key = "checkpoint" if key == "checkpoint_version" else key
        if protocol["model"][source_key] != value:
            raise ValueError(f"Registered transformer setting changed: {key}")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen in-context input changed: {path}")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"], "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "torch", "pyarrow", "scipy")}}
    frozen = output / "frozen_screen.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve this in-context experiment after an identity change")
    write_json(frozen, identity)
    for count, day in enumerate(protocol["dates"], 1):
        folder = output / "dates" / day
        if (folder / "completed.json").exists():
            check_completed(folder, identity)
            continue
        write_json(output / "progress.json", {"date": day, "stage": "context_preparation", "assessment_scores_sealed": True})
        prepare_day(day, protocol, folder, identity)
        gc.collect()
        supervision = {}
        for name in MODEL_SPECS:
            write_json(output / "progress.json", {"date": day, "stage": "model_procedure", "model": name, "assessment_scores_sealed": True})
            supervision[name] = supervise(day, name, protocol_path, protocol, output)
            print(f"tabicl_procedure={day}/{name} seconds={supervision[name]['seconds']:.1f}", flush=True)
        finish_day(day, protocol, output, identity, supervision)
        print(f"tabicl_dates_frozen={count}/{len(protocol['dates'])}", flush=True)
    write_json(output / "summary.json", reveal(output, protocol, identity))
    print(f"tabicl_development_complete {output / 'summary.json'}", flush=True)


def synthetic(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen wrapper preflight input changed: {path}")
    if output.exists():
        raise ValueError("Preserve all previous synthetic wrapper attempts")
    output.mkdir(parents=True)
    rng = np.random.default_rng(20260908)
    labels = {s: np.tile([-1, 0, 1], 16) for s in SYMBOLS}
    features = {s: pd.DataFrame(rng.normal(size=(48, 4)), columns=list("abcd")) for s in SYMBOLS}
    queries = {s: pd.DataFrame(rng.normal(size=(64, 4)), columns=list("abcd")) for s in SYMBOLS}
    population = {SYMBOLS[0]: np.array([.1, .8, .1]), SYMBOLS[1]: np.array([.2, .6, .2])}
    forecasts, metadata = train_and_forecast(features, labels, population, queries, "tabicl", output, protocol)
    for s in SYMBOLS:
        if forecasts[s].shape != (64, 3) or not np.isfinite(forecasts[s]).all() or not np.allclose(forecasts[s].sum(axis=1), 1):
            raise ValueError("Finite normalized synthetic forecasts required")
    write_json(output / "summary.json", {"protocol_sha256": sha256_file(protocol_path), "market_model_procedures": 0,
        "evidence_status": "synthetic_operational_and_causality_checks_only", "metadata": metadata})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--worker", choices=MODEL_SPECS)
    parser.add_argument("--day")
    args = parser.parse_args()
    if args.worker:
        worker(args.protocol.resolve(), args.day, args.worker, args.output.resolve())
    elif args.synthetic:
        synthetic(args.protocol.resolve(), args.output.resolve())
    elif args.run:
        run(args.protocol.resolve(), args.output.resolve())
    else:
        print("Pass --run for the frozen development comparison, or --synthetic for wrapper verification.")
