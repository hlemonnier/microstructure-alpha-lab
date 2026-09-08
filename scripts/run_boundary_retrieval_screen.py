"""Evaluate date-excluded historical retrieval on the frozen development cohort."""

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
from lob_forge.boundary_confirm_model import SYMBOLS
from lob_forge.boundary_event_inputs import utc_ms
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_learned_memory import SequenceInputs
from lob_forge.boundary_pooled import PooledForecaster
from lob_forge.boundary_prequential import decoded_release_clock
from lob_forge.boundary_regime_coverage import history_cohorts
from lob_forge.boundary_retrieval import ARCHITECTURES, HistoricalBankSampler, RetrievalForecaster, fit_retrieval
from benchmark_boundary_tabicl_feature_budget import memory_guard
from run_boundary_confirmation import noon, save_predictions, write_json
from run_boundary_event_clock_screen import check_completed
from run_boundary_tabicl_feature_budget import BASELINES as OLD_BASELINES, BLENDS as OLD_BLENDS, MODEL_SPECS as OLD_MODELS
from run_boundary_tabicl_screen import verify_artifacts

ROOT = Path(__file__).resolve().parents[1]
BASELINES = (*OLD_BASELINES, *OLD_MODELS, *OLD_BLENDS)
MODEL_SPECS = {f"{representation}_{architecture}": (representation, architecture)
    for representation in ("observations", "combined") for architecture in ARCHITECTURES}
BLENDS = {f"{name}_{suffix}": (name, target)
    for name, (representation, _) in MODEL_SPECS.items()
    for suffix, target in (("matched_hgb", "observations_tree" if representation == "observations" else "combined_hgb"), ("deep500_hgb", "deep500_hgb"))}


def original_path(protocol, day, representation):
    return ROOT / protocol["context_run" if representation == "observations" else "basis_run"] / "dates" / day / f"{representation}_neural"


def prepare(day, protocol, folder, identity):
    if folder.exists():
        raise ValueError("Preserve any previous retrieval preparation attempt")
    folder.mkdir(parents=True)
    started = time.monotonic()
    training_days = history_cohorts(day)["recent4_stride4"]["train_dates"]
    validation_day = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    manifest = json.loads((ROOT / protocol["manifest"]).read_text())
    source = ROOT / protocol["memory_run"] / "dates" / day
    check_completed(source, json.loads((source.parent.parent / "frozen_screen.json").read_text()))
    source_record = json.loads((source / "completed.json").read_text())
    assessments, records = {}, {}
    for representation in ("observations", "combined"):
        original = PooledForecaster.load(original_path(protocol, day, representation))
        if len(original.columns) != protocol["representations"][representation]:
            raise ValueError("Original retrieval feature schema changed")
        storage = source / representation / "training"
        for name in ("matrix", "labels", "supervised", "available"):
            path = storage / f"{name}.npy"
            if sha256_file(path) != source_record["artifact_hashes"][str(path.relative_to(source))]:
                raise ValueError("Frozen historical sequence input changed")
        matrix = np.load(storage / "matrix.npy", mmap_mode="r", allow_pickle=False)
        labels = np.load(storage / "labels.npy", mmap_mode="r", allow_pickle=False)
        supervised = np.load(storage / "supervised.npy", mmap_mode="r", allow_pickle=False)
        available = np.load(storage / "available.npy", mmap_mode="r", allow_pickle=False)
        if matrix.shape != (8, 86400, int(original.active.sum()) + 1) or labels.shape != (8, 86400) or np.any(supervised & ~available):
            raise ValueError("Original full historical grid identity changed")
        chunks, yy, aa, dd, clocks, releases = [], [], [], [], [], []
        for stream in range(8):
            asset, ordinal = divmod(stream, 4)
            current, symbol = training_days[ordinal], SYMBOLS[asset]
            seconds = np.flatnonzero(supervised[stream])
            if np.any(seconds % 4):
                raise ValueError("Original historical stride-four rows changed")
            clock = utc_ms(current) + 1000 * seconds
            raw = next(r for r in manifest["sessions"] if (r["symbol"], r["session_date"]) == (symbol, current))
            outcomes = pd.read_parquet(ROOT / raw["features_path"], columns=["decision_time", "label", "future_event_time"]).set_index("decision_time").loc[clock]
            np.testing.assert_array_equal(outcomes.label.to_numpy(), labels[stream, seconds])
            release = decoded_release_clock(outcomes.future_event_time.to_numpy())
            if (release >= utc_ms(validation_day)).any():
                raise ValueError("Every historical key label must be released before validation")
            chunks.append(matrix[stream, seconds].copy())
            yy.append(labels[stream, seconds].copy())
            aa.append(np.full(len(seconds), asset, dtype=np.int64))
            dd.append(np.full(len(seconds), ordinal, dtype=np.int64))
            clocks.append(clock)
            releases.append({"date": current, "symbol": symbol, "rows": len(seconds), "maximum_release_ms": int(release.max())})
        train = {"matrix": np.concatenate(chunks), "labels": np.concatenate(yy), "assets": np.concatenate(aa),
            "dates": np.concatenate(dd), "times": np.concatenate(clocks)}
        sampler = HistoricalBankSampler(train["labels"], train["assets"], train["dates"])
        for asset in (0, 1):
            np.testing.assert_array_equal(sampler.priors[asset], original.priors[asset])
        validation, validation_y, query, query_y, query_t = {}, {}, {}, {}, {}
        for current in (validation_day, day):
            for asset, symbol in enumerate(SYMBOLS):
                path = source / representation / f"{symbol}_{current}_sequence.npz"
                if sha256_file(path) != source_record["artifact_hashes"][str(path.relative_to(source))]:
                    raise ValueError("Frozen normalized query sequence changed")
                seq = SequenceInputs.load(path)
                selected = seq.selected
                clock = seq.times[selected]
                if len(clock) != 7070 or not np.all(noon(clock, current)):
                    raise ValueError("All original 7070 noon queries must remain")
                raw = next(r for r in manifest["sessions"] if (r["symbol"], r["session_date"]) == (symbol, current))
                outcomes = pd.read_parquet(ROOT / raw["features_path"], columns=["decision_time", "label", "future_event_time"]).set_index("decision_time").loc[clock]
                released = decoded_release_clock(outcomes.future_event_time.to_numpy())
                if current == validation_day:
                    if (released >= utc_ms(day)).any():
                        raise ValueError("Every validation outcome must be released before assessment")
                    validation[asset], validation_y[asset] = seq.matrix[selected].copy(), outcomes.label.to_numpy(dtype=np.int64)
                else:
                    query[asset], query_y[asset], query_t[asset] = seq.matrix[selected].copy(), outcomes.label.to_numpy(dtype=np.int64), clock.copy()
                    with np.load(ROOT / protocol["baseline_run"] / "dates" / day / "predictions" / f"{symbol}_{representation}_neural_registered.npz", allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved["labels"], query_y[asset])
                        np.testing.assert_array_equal(saved["decision_times"], clock)
                        np.testing.assert_array_equal(saved["probabilities"], seq.base_probabilities)
        destination = folder / representation
        destination.mkdir()
        joblib.dump({**train, "validation": validation, "validation_labels": validation_y}, destination / "training.joblib")
        np.savez_compressed(destination / "queries.npz", **{str(a): x for a, x in query.items()})
        assessments[representation] = {**{f"{a}_labels": query_y[a] for a in (0, 1)}, **{f"{a}_times": query_t[a] for a in (0, 1)}}
        records[representation] = {"historical_inputs_exact": True, "original_neural_assessment_forecasts_exact": True,
            "training_rows": len(train["labels"]), "dimensions": train["matrix"].shape[1], "training_releases": releases,
            "historical_cell_counts": sampler.counts.tolist(), "population_priors": {str(a): p.tolist() for a, p in sampler.priors.items()}}
        del matrix, labels, available, supervised, train, chunks, original, validation, query
        gc.collect()
    for field in assessments["observations"]:
        np.testing.assert_array_equal(assessments["observations"][field], assessments["combined"][field])
    np.savez_compressed(folder / "assessment.npz", **assessments["observations"])
    write_json(folder / "preparation_metadata.json", {"training_dates": training_days, "validation_date": validation_day, "representations": records})
    write_json(folder / "prepared.json", {"identity": identity, "date": day, "seconds": time.monotonic() - started,
        "artifact_hashes": {str(p.relative_to(folder)): sha256_file(p) for p in folder.rglob("*") if p.is_file()}})


def worker(protocol_path, output, day, name):
    import psutil
    import torch

    psutil.Process().memory_info()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads(protocol_path.read_text())
    identity = json.loads((output / "frozen_screen.json").read_text())
    if identity["protocol_sha256"] != sha256_file(protocol_path):
        raise ValueError("Frozen retrieval worker identity changed")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen retrieval input changed: {path}")
    backend = protocol["selected_backend"]
    if backend == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Registered retrieval backend unavailable")
    representation, architecture = MODEL_SPECS[name]
    folder, target = output / "dates" / day, output / "dates" / day / "workers" / name
    prepared = json.loads((folder / "prepared.json").read_text())
    for input_name in (f"{representation}/training.joblib", f"{representation}/queries.npz"):
        if sha256_file(folder / input_name) != prepared["artifact_hashes"][input_name]:
            raise ValueError("Frozen historical/query matrix changed")
    readings = {"maximum_sampled_rss_bytes": 0, "maximum_sampled_gpu_driver_bytes": 0, "maximum_sampled_simultaneous_sum_bytes": 0}
    stop = threading.Event()
    guard = threading.Thread(target=memory_guard, args=(backend, target, protocol["operational_limits"], stop, readings), daemon=True)
    guard.start()
    begin = time.monotonic()
    original = PooledForecaster.load(original_path(protocol, day, representation))
    training = joblib.load(folder / representation / "training.joblib")
    with np.load(folder / representation / "queries.npz", allow_pickle=False) as saved:
        queries = {a: saved[str(a)].copy() for a in (0, 1)}
    learner, history = fit_retrieval(original, training["matrix"], training["labels"], training["assets"], training["dates"],
        training["validation"], training["validation_labels"], architecture=architecture, device=backend)
    learner.save(target / "model")
    write_json(target / "training_history.json", history)
    restored = RetrievalForecaster.load(target / "model", device=backend)
    ready = time.monotonic() - begin
    forecasts, checks = {}, {}
    for asset, symbol in enumerate(SYMBOLS):
        p = learner.predict_matrix(queries[asset], asset)
        np.testing.assert_array_equal(p, restored.predict_matrix(queries[asset], asset))
        prefix = queries[asset][:128]
        later = prefix.copy()
        later[37:] = 1e4
        future = learner.predict_matrix(later, asset)
        short = learner.predict_matrix(prefix[:37], asset)
        single = learner.predict_matrix(prefix[:7], asset, batch_size=1)
        partition = learner.predict_matrix(prefix, asset, batch_size=31, key_chunk_rows=2048)
        errors = {"future_prefix": float(np.max(np.abs(p[:37] - future[:37]))),
            "shorter_prefix": float(np.max(np.abs(p[:37] - short))), "single_query": float(np.max(np.abs(p[:7] - single))),
            "query_key_partition": float(np.max(np.abs(p[:128] - partition)))}
        if max(errors.values()) > protocol["market_partition_probability_absolute_tolerance"]:
            raise ValueError(f"Registered market retrieval causality/partition check failed: {errors}")
        forecasts[symbol] = p
        checks[symbol] = {"checkpoint_forecasts_exact": True, "errors": errors}
    np.savez_compressed(target / "forecasts.npz", **forecasts)
    stop.set()
    guard.join()
    write_json(target / "operational.json", {"representation": representation, "architecture": architecture, "backend": backend,
        "training_rows": len(training["labels"]), "dimensions": training["matrix"].shape[1],
        "fit_save_reload_ready_seconds": ready, "seconds": time.monotonic() - begin,
        "optimizer_steps": history["optimizer_steps"], "best_epoch": history["best_epoch"],
        "trainable_parameters": sum(p.numel() for p in learner.network.parameters()), "checks": checks, "memory": readings,
        "assessment_labels_opened": False, "historical_training_dates": history_cohorts(day)["recent4_stride4"]["train_dates"]})
    write_json(target / "succeeded.json", {"date": day, "model": name,
        "artifact_hashes": {str(p.relative_to(target)): sha256_file(p) for p in target.rglob("*") if p.is_file() and p.name != "stdout.log"}})


def supervise(protocol_path, output, day, name, protocol):
    folder = output / "dates" / day / "workers" / name
    if folder.exists():
        raise ValueError("Preserve an existing retrieval worker; no implicit retry")
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
            failure = "Registered retrieval worker deadline exceeded"
    record = {"returncode": code, "failure": failure, "seconds": time.monotonic() - begin, "log_sha256": sha256_file(folder / "stdout.log")}
    write_json(folder / "supervision.json", record)
    if failure or not (folder / "succeeded.json").exists():
        raise ValueError(f"Preserve failed retrieval worker {day}/{name}: {failure}")
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
        raise ValueError("Incomplete retrieval development family")
    board = []
    for name, policy in sorted({(r["model"], r["policy"]) for r in records}):
        rows = [r for r in records if (r["model"], r["policy"]) == (name, policy)]
        board.append({"model": name, "policy": policy, "asset_dates": len(rows),
            "means": {k: float(np.mean([r["metrics"][k] for r in rows])) for k in ("balanced_accuracy", "natural_accuracy", "log_loss", "macro_f1")},
            "balanced_accuracy_by_asset": {s: float(np.mean([r["metrics"]["balanced_accuracy"] for r in rows if r["symbol"] == s])) for s in SYMBOLS}})
    return {"identity": identity, "evidence_status": protocol["evidence_status"], "model_procedures": len(MODEL_SPECS) * len(protocol["dates"]),
        "trainable_model_fits": 18, "fixed_metric_procedures": 6, "post_fit_panels": len(records), "substantial_gain_confirmed": False,
        "leaderboard": sorted(board, key=lambda r: r["means"]["balanced_accuracy"], reverse=True), "records": records}


def run(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if protocol["evidence_status"] != "development_only_on_exposed_dates" or protocol["model_cases"] != list(MODEL_SPECS) or protocol["decision_policies"] != ["registered", "forecast_3600"]:
        raise ValueError("Only the frozen historical retrieval family is registered")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen retrieval input changed: {path}")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"], "python": platform.python_version(),
        "backend": protocol["selected_backend"], "dependencies": {n: version(n) for n in ("numpy", "pandas", "scikit-learn", "torch", "scipy", "pyarrow")}}
    if output.exists():
        raise ValueError("Preserve any existing retrieval study; no implicit retry")
    write_json(output / "frozen_screen.json", identity)
    for count, day in enumerate(protocol["dates"], 1):
        write_json(output / "progress.json", {"date": day, "stage": "historical_input_verification", "assessment_scores_sealed": True})
        prepare(day, protocol, output / "dates" / day, identity)
        for name in MODEL_SPECS:
            write_json(output / "progress.json", {"date": day, "model": name, "stage": "training", "assessment_scores_sealed": True})
            result = supervise(protocol_path, output, day, name, protocol)
            print(f"retrieval_fit={day}/{name} seconds={result['seconds']:.1f}", flush=True)
        finish_day(day, protocol, output, identity)
        print(f"retrieval_dates_frozen={count}/{len(protocol['dates'])}", flush=True)
    write_json(output / "summary.json", reveal(output, protocol, identity))
    print(f"retrieval_development_complete {output / 'summary.json'}", flush=True)


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
        print("Pass --run for the frozen historical retrieval development study.")
