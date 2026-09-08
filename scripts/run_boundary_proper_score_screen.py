"""Compare bounded proper neural losses on the unchanged exposed cohort."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import threading
import time
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_adaptive_priors import forecast_priors
from lob_forge.boundary_confirm_model import SYMBOLS
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_pooled import PooledForecaster
from lob_forge.boundary_regime_coverage import history_cohorts
from lob_forge.boundary_proper_score import VARIANTS, fit_proper_score, predict_matrix
from benchmark_boundary_tabicl_feature_budget import memory_guard
from run_boundary_confirmation import save_predictions, write_json
from run_boundary_event_clock_screen import check_completed
from run_boundary_stochastic_boost_screen import BASELINES as OLD_BASELINES, BLENDS as OLD_BLENDS, MODEL_SPECS as OLD_MODELS
from run_boundary_retrieval_screen import original_path, prepare
from run_boundary_tabicl_screen import verify_artifacts

ROOT = Path(__file__).resolve().parents[1]
BASELINES = (*OLD_BASELINES, *OLD_MODELS, *OLD_BLENDS)
MODEL_SPECS = {f"{representation}_proper_{variant}": (representation, variant)
    for representation in ("observations", "combined") for variant in VARIANTS}
BLENDS = {f"{name}_{suffix}": (name, target)
    for name, (representation, _) in MODEL_SPECS.items()
    for suffix, target in (("matched_hgb", "observations_tree" if representation == "observations" else "combined_hgb"), ("deep500_hgb", "deep500_hgb"))}


def worker(protocol_path, output, day, name):
    import psutil
    import torch

    psutil.Process().memory_info()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads(protocol_path.read_text())
    identity = json.loads((output / "frozen_screen.json").read_text())
    if identity["protocol_sha256"] != sha256_file(protocol_path):
        raise ValueError("Frozen proper-score worker identity changed")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen proper-score input changed: {path}")
    representation, variant = MODEL_SPECS[name]
    folder, target = output / "dates" / day, output / "dates" / day / "workers" / name
    prepared = json.loads((folder / "prepared.json").read_text())
    for input_name in (f"{representation}/training.joblib", f"{representation}/queries.npz"):
        if sha256_file(folder / input_name) != prepared["artifact_hashes"][input_name]:
            raise ValueError("Frozen original normalized training/query matrix changed")
    readings = {"maximum_sampled_rss_bytes": 0, "maximum_sampled_gpu_driver_bytes": 0, "maximum_sampled_simultaneous_sum_bytes": 0}
    stop = threading.Event()
    guard = threading.Thread(target=memory_guard, args=("cpu", target, protocol["operational_limits"], stop, readings), daemon=True)
    guard.start()
    begin = time.monotonic()
    original = PooledForecaster.load(original_path(protocol, day, representation))
    training = joblib.load(folder / representation / "training.joblib")
    with np.load(folder / representation / "queries.npz", allow_pickle=False) as saved:
        queries = {a: saved[str(a)].copy() for a in (0, 1)}
    learner, history = fit_proper_score(original, training["matrix"], training["labels"], training["assets"],
        training["validation"], training["validation_labels"], variant=variant)
    exact_control = None
    if variant == "log":
        expected, actual = original.network.state_dict(), learner.network.state_dict()
        if expected.keys() != actual.keys() or any(not torch.equal(expected[key], actual[key]) for key in expected):
            raise ValueError("The log-loss control must reproduce every original network parameter exactly")
        exact_control = True
    learner.save(target / "model")
    write_json(target / "training_history.json", history)
    restored = PooledForecaster.load(target / "model")
    ready = time.monotonic() - begin
    forecasts, checks = {}, {}
    for asset, symbol in enumerate(SYMBOLS):
        p = predict_matrix(learner, queries[asset], asset)
        np.testing.assert_array_equal(p, predict_matrix(restored, queries[asset], asset))
        np.testing.assert_array_equal(learner.priors[asset], original.priors[asset])
        if variant == "log":
            np.testing.assert_array_equal(p, predict_matrix(original, queries[asset], asset))
        prefix = queries[asset][:128]
        before = predict_matrix(learner, prefix, asset)
        later = prefix.copy()
        later[37:, :-1] = 1e4
        future = predict_matrix(learner, later, asset)
        short = predict_matrix(learner, prefix[:37], asset)
        single = predict_matrix(learner, prefix[:7], asset, batch_size=1)
        errors = {"future_prefix": float(np.max(np.abs(before[:37] - future[:37]))),
            "full_query_prefix": float(np.max(np.abs(p[:128] - before))),
            "shorter_prefix": float(np.max(np.abs(p[:37] - short))), "single_query": float(np.max(np.abs(p[:7] - single)))}
        if max(errors.values()) > protocol["query_probability_absolute_tolerance"]:
            raise ValueError(f"Registered proper-score causality/partition check failed: {errors}")
        forecasts[symbol] = p
        checks[symbol] = {"checkpoint_forecasts_exact": True, "original_priors_exact": True,
            "original_log_control_forecasts_exact": exact_control, "errors": errors}
    np.savez_compressed(target / "forecasts.npz", **forecasts)
    stop.set()
    guard.join()
    write_json(target / "operational.json", {"representation": representation, "variant": variant, "backend": "cpu",
        "training_rows": len(training["labels"]), "dimensions": training["matrix"].shape[1],
        "fit_save_reload_ready_seconds": ready, "seconds": time.monotonic() - begin,
        "optimizer_steps": history["optimizer_steps"], "best_epoch": history["best_epoch"],
        "original_log_control_state_exact": exact_control, "trainable_parameters": sum(p.numel() for p in learner.network.parameters()),
        "checks": checks, "memory": readings, "assessment_labels_opened": False,
        "historical_training_dates": history_cohorts(day)["recent4_stride4"]["train_dates"]})
    write_json(target / "succeeded.json", {"date": day, "model": name,
        "artifact_hashes": {str(p.relative_to(target)): sha256_file(p) for p in target.rglob("*") if p.is_file() and p.name != "stdout.log"}})


def supervise(protocol_path, output, day, name, protocol):
    folder = output / "dates" / day / "workers" / name
    if folder.exists():
        raise ValueError("Preserve an existing proper_score worker; no implicit retry")
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
            failure = "Registered proper_score worker deadline exceeded"
    record = {"returncode": code, "failure": failure, "seconds": time.monotonic() - begin, "log_sha256": sha256_file(folder / "stdout.log")}
    write_json(folder / "supervision.json", record)
    if failure or not (folder / "succeeded.json").exists():
        raise ValueError(f"Preserve failed proper_score worker {day}/{name}: {failure}")
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
        raise ValueError("Incomplete proper_score development family")
    board = []
    for name, policy in sorted({(r["model"], r["policy"]) for r in records}):
        rows = [r for r in records if (r["model"], r["policy"]) == (name, policy)]
        board.append({"model": name, "policy": policy, "asset_dates": len(rows),
            "means": {k: float(np.mean([r["metrics"][k] for r in rows])) for k in ("balanced_accuracy", "natural_accuracy", "log_loss", "macro_f1")},
            "balanced_accuracy_by_asset": {s: float(np.mean([r["metrics"]["balanced_accuracy"] for r in rows if r["symbol"] == s])) for s in SYMBOLS}})
    return {"identity": identity, "evidence_status": protocol["evidence_status"], "model_procedures": len(MODEL_SPECS) * len(protocol["dates"]),
        "trainable_model_fits": 24, "log_control_fits": 6, "post_fit_panels": len(records), "substantial_gain_confirmed": False,
        "leaderboard": sorted(board, key=lambda r: r["means"]["balanced_accuracy"], reverse=True), "records": records}


def run(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if protocol["evidence_status"] != "development_only_on_exposed_dates" or protocol["model_cases"] != list(MODEL_SPECS) or protocol["decision_policies"] != ["registered", "forecast_3600"]:
        raise ValueError("Only the frozen historical proper_score family is registered")
    names = [*BASELINES, *MODEL_SPECS, *BLENDS]
    if (len(names) != len(set(names)) or len(names) != protocol["forecast_sources"]
        or list(BASELINES) != protocol["retained_sources"] or {k: list(v) for k, v in BLENDS.items()} != protocol["blend_members"]):
        raise ValueError("The complete proper-score comparison family changed")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen proper_score input changed: {path}")
    preflight = json.loads((ROOT / protocol["preflight_summary"]).read_text())
    if preflight["all_cases_passed"] is not True or len(preflight["records"]) != 8:
        raise ValueError("All eight frozen proper-score synthetic cases must pass")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"], "python": platform.python_version(),
        "backend": "cpu", "dependencies": {n: version(n) for n in ("numpy", "pandas", "scikit-learn", "torch", "scipy", "pyarrow")}}
    if output.exists():
        raise ValueError("Preserve any existing proper_score study; no implicit retry")
    write_json(output / "frozen_screen.json", identity)
    for count, day in enumerate(protocol["dates"], 1):
        write_json(output / "progress.json", {"date": day, "stage": "historical_input_verification", "assessment_scores_sealed": True})
        prepare(day, protocol, output / "dates" / day, identity)
        for name in MODEL_SPECS:
            write_json(output / "progress.json", {"date": day, "model": name, "stage": "training", "assessment_scores_sealed": True})
            result = supervise(protocol_path, output, day, name, protocol)
            print(f"proper_score_fit={day}/{name} seconds={result['seconds']:.1f}", flush=True)
        finish_day(day, protocol, output, identity)
        print(f"proper_score_dates_frozen={count}/{len(protocol['dates'])}", flush=True)
    write_json(output / "summary.json", reveal(output, protocol, identity))
    print(f"proper_score_development_complete {output / 'summary.json'}", flush=True)


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
        print("Pass --run for the frozen historical proper_score development study.")
