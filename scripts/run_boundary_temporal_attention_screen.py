"""Compare registered temporal readouts on the unchanged exposed cohort."""

from __future__ import annotations

import argparse
import copy
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

import numpy as np
import pandas as pd

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_adaptive_priors import forecast_priors
from lob_forge.boundary_confirm_model import SYMBOLS
from lob_forge.boundary_event_inputs import utc_ms
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_learned_memory import SequenceInputs, TrainingGrid, corrected_posterior, matrix_logits
from lob_forge.boundary_pooled import PooledForecaster
from lob_forge.boundary_prequential import decoded_release_clock
from lob_forge.boundary_proper_score import predict_matrix
from lob_forge.boundary_regime_coverage import history_cohorts
from lob_forge.boundary_temporal_attention import ARCHITECTURES, CONFIG, TemporalAttentionForecaster, fit_temporal_attention, predict_deltas, sequence_probabilities
from benchmark_boundary_tabicl_feature_budget import memory_guard
from run_boundary_confirmation import noon, save_predictions, write_json
from run_boundary_event_clock_screen import check_completed
from run_boundary_proper_score_screen import BASELINES as OLD_BASELINES, BLENDS as OLD_BLENDS, MODEL_SPECS as OLD_MODELS
from run_boundary_retrieval_screen import original_path
from run_boundary_tabicl_screen import verify_artifacts

ROOT = Path(__file__).resolve().parents[1]
BASELINES = (*OLD_BASELINES, *OLD_MODELS, *OLD_BLENDS)
MODEL_SPECS = {f"{representation}_attention_{architecture}": (representation, architecture)
    for representation in ("observations", "combined") for architecture in ARCHITECTURES}
BLENDS = {f"{name}_{suffix}": (name, target)
    for name, (representation, _) in MODEL_SPECS.items()
    for suffix, target in (("matched_hgb", "observations_tree" if representation == "observations" else "combined_hgb"), ("deep500_hgb", "deep500_hgb"))}
GRID_FIELDS = ("matrix", "base_logits", "labels", "weights", "available", "supervised")


def load_grid(paths):
    return TrainingGrid(**{name: np.load(ROOT / paths[name], mmap_mode="r", allow_pickle=False) for name in GRID_FIELDS}, assets=np.repeat([0, 1], 4))


def prepare(day, protocol, folder, identity):
    """Verify existing grids and original forecasts; no normalizer/model fitting."""
    import torch

    torch.set_num_threads(2)
    if folder.exists():
        raise ValueError("Preserve every previous attention preparation attempt")
    folder.mkdir(parents=True)
    started = time.monotonic()
    parent = ROOT / protocol["baseline_run"] / "dates" / day
    memory = ROOT / protocol["memory_run"] / "dates" / day
    for source in (parent, memory):
        check_completed(source, json.loads((source.parent.parent / "frozen_screen.json").read_text()))
    training_days = history_cohorts(day)["recent4_stride4"]["train_dates"]
    validation_day = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    manifest = json.loads((ROOT / protocol["manifest"]).read_text())
    sessions = {(row["session_date"], row["symbol"]): row for row in manifest["sessions"]}
    if len(sessions) != len(manifest["sessions"]):
        raise ValueError("Exactly one original input session per asset/date is required")
    input_paths = {ROOT / protocol["manifest"], parent / "completed.json", memory / "completed.json"}
    outcomes = {}
    for current in [*training_days, validation_day, day]:
        for symbol in SYMBOLS:
            session = sessions[current, symbol]
            source = ROOT / session["features_path"]
            outcomes[symbol, current] = pd.read_parquet(source, columns=["decision_time", "label", "future_event_time"]).set_index("decision_time")
            input_paths.add(source)
    assessments, records, sources = {}, {}, {}
    for representation in ("observations", "combined"):
        original_folder = original_path(protocol, day, representation)
        original = PooledForecaster.load(original_folder)
        if len(original.columns) != protocol["representations"][representation]:
            raise ValueError("Original temporal feature schema changed")
        input_paths.update(p for p in original_folder.rglob("*") if p.is_file())
        memory_metadata = memory / representation / "preparation.json"
        memory_prepared = json.loads(memory_metadata.read_text())
        if (memory_prepared["training_dates"] != training_days or memory_prepared["validation_date"] != validation_day
            or memory_prepared["stream_date_order"] != [*training_days, *training_days]
            or memory_prepared["stream_assets"] != np.repeat([0, 1], 4).tolist()):
            raise ValueError("Original stream/date identities changed")
        input_paths.add(memory_metadata)
        grid_paths = {name: str((memory / representation / "training" / f"{name}.npy").relative_to(ROOT)) for name in GRID_FIELDS}
        input_paths.update(ROOT / path for path in grid_paths.values())
        grid = load_grid(grid_paths)
        dimensions = int(original.active.sum()) + 1
        grid.validate(dimensions, original.priors)
        release_records = []
        for stream in range(8):
            asset, ordinal = divmod(stream, 4)
            symbol, current = SYMBOLS[asset], training_days[ordinal]
            available = grid.available[stream]
            if not np.isfinite(grid.matrix[stream, available]).all() or not np.all(grid.matrix[stream, available, -1] == 2 * asset - 1):
                raise ValueError("Finite normalized history and original asset indicators required")
            seconds = np.flatnonzero(grid.supervised[stream])
            clock = utc_ms(current) + 1000 * seconds
            raw = outcomes[symbol, current].loc[clock]
            np.testing.assert_array_equal(raw.label.to_numpy(), grid.labels[stream, seconds])
            release = decoded_release_clock(raw.future_event_time.to_numpy())
            if (release >= utc_ms(validation_day)).any():
                raise ValueError("Every training outcome must have been released before validation")
            np.testing.assert_array_equal(matrix_logits(original, grid.matrix[stream, seconds].copy()), grid.base_logits[stream, seconds])
            release_records.append({"date": current, "symbol": symbol, "rows": len(seconds), "maximum_release_ms": int(release.max())})
        validation_labels, query_labels, query_times = {}, {}, {}
        validation_paths, query_paths = {}, {}
        for current in (validation_day, day):
            for asset, symbol in enumerate(SYMBOLS):
                path = memory / representation / f"{symbol}_{current}_sequence.npz"
                input_paths.add(path)
                seq = SequenceInputs.load(path)
                seq.validate(dimensions)
                clock = seq.times[seq.selected]
                if len(clock) != 7070 or not np.all(noon(clock, current)):
                    raise ValueError("Every original 7070-row noon must remain")
                raw = outcomes[symbol, current].loc[clock]
                released = decoded_release_clock(raw.future_event_time.to_numpy())
                np.testing.assert_array_equal(matrix_logits(original, seq.matrix[seq.selected]), seq.base_logits)
                np.testing.assert_array_equal(predict_matrix(original, seq.matrix[seq.selected], asset), seq.base_probabilities)
                if current == validation_day:
                    if (released >= utc_ms(day)).any():
                        raise ValueError("Every validation outcome must have been released before assessment")
                    validation_labels[symbol] = raw.label.to_numpy(dtype=np.int64)
                    validation_paths[symbol] = str(path.relative_to(ROOT))
                else:
                    query_labels[asset], query_times[asset] = raw.label.to_numpy(dtype=np.int64), clock.copy()
                    query_paths[symbol] = str(path.relative_to(ROOT))
                    previous = parent / "predictions" / f"{symbol}_{representation}_neural_registered.npz"
                    input_paths.add(previous)
                    with np.load(previous, allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved["labels"], query_labels[asset])
                        np.testing.assert_array_equal(saved["decision_times"], clock)
                        np.testing.assert_array_equal(saved["probabilities"], seq.base_probabilities)
                        np.testing.assert_array_equal(saved["train_priors"], original.priors[asset])
                release_records.append({"date": current, "symbol": symbol, "rows": len(clock), "maximum_release_ms": int(released.max())})
        destination = folder / representation
        destination.mkdir()
        np.savez_compressed(destination / "validation_labels.npz", **validation_labels)
        assessments[representation] = {**{f"{a}_labels": query_labels[a] for a in (0, 1)}, **{f"{a}_times": query_times[a] for a in (0, 1)}}
        sources[representation] = {"grid": grid_paths, "validation": validation_paths, "query": query_paths,
            "original_model": str(original_folder.relative_to(ROOT))}
        records[representation] = {"training_rows": int(grid.supervised.sum()), "dimensions": dimensions,
            "population_priors": {str(a): p.tolist() for a, p in original.priors.items()}, "release_checks": release_records,
            "historical_logits_exact": True, "validation_and_query_logits_and_original_probabilities_exact": True,
            "original_assessment_labels_times_probabilities_and_priors_exact": True}
        del grid, original, seq
    for field in assessments["observations"]:
        np.testing.assert_array_equal(assessments["observations"][field], assessments["combined"][field])
    np.savez_compressed(folder / "assessment.npz", **assessments["observations"])
    write_json(folder / "preparation_metadata.json", {"training_dates": training_days, "validation_date": validation_day, "representations": records})
    write_json(folder / "prepared.json", {"identity": identity, "date": day, "seconds": time.monotonic() - started,
        "sources": sources, "input_artifacts": {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(input_paths)},
        "artifact_hashes": {str(p.relative_to(folder)): sha256_file(p) for p in folder.rglob("*") if p.is_file()},
        "market_model_fits": 0, "new_assessment_metrics_computed": False})


def worker(protocol_path, output, day, name):
    import psutil
    import torch

    psutil.Process().memory_info()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads(protocol_path.read_text())
    identity = json.loads((output / "frozen_screen.json").read_text())
    if identity["protocol_sha256"] != sha256_file(protocol_path):
        raise ValueError("Frozen attention worker identity changed")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen attention input changed: {path}")
    representation, architecture = MODEL_SPECS[name]
    backend = protocol["selected_backend"]
    folder, target = output / "dates" / day, output / "dates" / day / "workers" / name
    prepared = json.loads((folder / "prepared.json").read_text())
    if prepared["identity"] != identity:
        raise ValueError("Original attention preparation identity changed")
    verify_artifacts(folder, prepared)
    for path, checksum in prepared["input_artifacts"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError("Previously verified original sequence input changed")
    readings = {"maximum_sampled_rss_bytes": 0, "maximum_sampled_gpu_driver_bytes": 0, "maximum_sampled_simultaneous_sum_bytes": 0}
    stop = threading.Event()
    guard = threading.Thread(target=memory_guard, args=(backend, target, protocol["operational_limits"], stop, readings), daemon=True)
    guard.start()
    begin = time.monotonic()
    sources = prepared["sources"][representation]
    original = PooledForecaster.load(ROOT / sources["original_model"])
    original_state = copy.deepcopy(original.network.state_dict())
    grid = load_grid(sources["grid"])
    validation = {s: SequenceInputs.load(ROOT / path) for s, path in sources["validation"].items()}
    with np.load(folder / representation / "validation_labels.npz", allow_pickle=False) as saved:
        validation_labels = {s: saved[s].copy() for s in SYMBOLS}
    learner, history = fit_temporal_attention(original, grid, validation, validation_labels, architecture=architecture, device=backend)
    if any(not torch.equal(value, original.network.state_dict()[name]) for name, value in original_state.items()):
        raise ValueError("The original neural model must remain exactly unchanged")
    learner.save(target / "model")
    write_json(target / "training_history.json", history)
    restored = TemporalAttentionForecaster.load(target / "model", device=backend)
    if any(not torch.equal(value, restored.original.network.state_dict()[name]) for name, value in original_state.items()):
        raise ValueError("Saved original neural parameters changed")
    ready = time.monotonic() - begin
    forecasts, checks = {}, {}
    for asset, symbol in enumerate(SYMBOLS):
        sequence = SequenceInputs.load(ROOT / sources["query"][symbol])
        p, inference = sequence_probabilities(learner.adapter, sequence, original.priors[asset])
        replay, _ = sequence_probabilities(restored.adapter, sequence, restored.original.priors[asset])
        np.testing.assert_array_equal(p, replay)
        np.testing.assert_array_equal(original.priors[asset], restored.original.priors[asset])
        prefix_end = int(np.flatnonzero(sequence.selected)[63]) + 1
        mask = sequence.selected[:prefix_end]
        delta, _ = predict_deltas(learner.adapter, sequence.matrix[:prefix_end], sequence.times[:prefix_end], selected=mask)
        altered = sequence.matrix[:prefix_end + 32].copy()
        altered[prefix_end:, :-1] = 1e4
        future_mask = np.r_[mask, np.zeros(len(altered) - prefix_end, dtype=bool)]
        future, _ = predict_deltas(learner.adapter, altered, sequence.times[:len(altered)], selected=future_mask)
        short_p = corrected_posterior(sequence.base_logits[:64], delta, original.priors[asset], sequence.base_probabilities[:64])
        future_p = corrected_posterior(sequence.base_logits[:64], future, original.priors[asset], sequence.base_probabilities[:64])
        partition, _ = sequence_probabilities(learner.adapter, sequence, original.priors[asset], chunk_rows=97)
        errors = {"future_prefix": float(np.max(np.abs(short_p - future_p))),
            "shorter_prefix": float(np.max(np.abs(p[:64] - short_p))), "query_partition": float(np.max(np.abs(p - partition)))}
        if max(errors.values()) > protocol["query_probability_absolute_tolerance"]:
            raise ValueError(f"Registered attention query causality/partition tolerance exceeded: {errors}")
        forecasts[symbol] = p
        checks[symbol] = {"checkpoint_forecasts_exact": True, "original_priors_exact": True, "errors": errors, "inference": inference}
    np.savez_compressed(target / "forecasts.npz", **forecasts)
    stop.set()
    guard.join()
    write_json(target / "operational.json", {"representation": representation, "architecture": architecture, "backend": backend,
        "training_rows": int(grid.supervised.sum()), "dimensions": grid.matrix.shape[2], "config": dict(CONFIG),
        "fit_save_reload_ready_seconds": ready, "seconds": time.monotonic() - begin,
        "optimizer_steps": history["optimizer_steps"], "best_epoch": history["best_epoch"], "original_parameters_and_saved_state_exact": True,
        "trainable_parameters": history["trainable_parameters"], "checks": checks, "memory": readings, "assessment_labels_opened": False,
        "historical_training_dates": history_cohorts(day)["recent4_stride4"]["train_dates"]})
    write_json(target / "succeeded.json", {"date": day, "model": name,
        "artifact_hashes": {str(p.relative_to(target)): sha256_file(p) for p in target.rglob("*") if p.is_file() and p.name != "stdout.log"}})

def supervise(protocol_path, output, day, name, protocol):
    folder = output / "dates" / day / "workers" / name
    if folder.exists():
        raise ValueError("Preserve an existing temporal_attention worker; no implicit retry")
    folder.mkdir(parents=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--protocol", str(protocol_path), "--output", str(output), "--worker", name, "--day", day]
    begin, failure, code = time.monotonic(), None, None
    with (folder / "stdout.log").open("w") as stream:
        try:
            result = subprocess.run(command, cwd=ROOT, env={**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTORCH_ENABLE_MPS_FALLBACK": "0"}, stdout=stream, stderr=subprocess.STDOUT, timeout=protocol["operational_limits"]["worker_seconds"])
            code = result.returncode
            if code:
                failure = f"Worker exited {code}"
        except subprocess.TimeoutExpired:
            failure = "Registered temporal_attention worker deadline exceeded"
    record = {"returncode": code, "failure": failure, "seconds": time.monotonic() - begin, "log_sha256": sha256_file(folder / "stdout.log")}
    write_json(folder / "supervision.json", record)
    if failure or not (folder / "succeeded.json").exists():
        raise ValueError(f"Preserve failed temporal_attention worker {day}/{name}: {failure}")
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
        raise ValueError("Incomplete temporal_attention development family")
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
        raise ValueError("Only the frozen historical temporal_attention family is registered")
    names = [*BASELINES, *MODEL_SPECS, *BLENDS]
    if (len(names) != len(set(names)) or len(names) != protocol["forecast_sources"]
        or list(BASELINES) != protocol["retained_sources"] or {k: list(v) for k, v in BLENDS.items()} != protocol["blend_members"]):
        raise ValueError("The complete temporal-attention comparison family changed")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen temporal_attention input changed: {path}")
    preflight = json.loads((ROOT / protocol["preflight_summary"]).read_text())
    chosen = protocol["selected_backend"]
    cases = [r for r in preflight["records"] if r["backend"] == chosen]
    if (chosen not in ("cpu", "mps") or preflight["selected_backend"] != chosen or len(preflight["records"]) != 16
        or len(cases) != 8 or any(r["status"] != "passed" for r in cases)
        or {(r["architecture"], r["dimensions"]) for r in cases} != {(a, d) for a in ARCHITECTURES for d in (221, 400)}):
        raise ValueError("One common backend must pass all eight registered attention cases")
    broad = json.loads((ROOT / protocol["completed_broad_audit_summary"]).read_text())
    if broad["trainable_model_fits"] != 40 or len(broad["records"]) != 560 or protocol["config"] != CONFIG:
        raise ValueError("The fixed attention configuration and complete earlier broad audit are required")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"], "python": platform.python_version(),
        "backend": chosen, "dependencies": {n: version(n) for n in ("numpy", "pandas", "scikit-learn", "torch", "scipy", "pyarrow")}}
    if output.exists():
        raise ValueError("Preserve any existing temporal_attention study; no implicit retry")
    write_json(output / "frozen_screen.json", identity)
    for count, day in enumerate(protocol["dates"], 1):
        write_json(output / "progress.json", {"date": day, "stage": "historical_input_verification", "assessment_scores_sealed": True})
        prepare(day, protocol, output / "dates" / day, identity)
        for name in MODEL_SPECS:
            write_json(output / "progress.json", {"date": day, "model": name, "stage": "training", "assessment_scores_sealed": True})
            result = supervise(protocol_path, output, day, name, protocol)
            print(f"temporal_attention_fit={day}/{name} seconds={result['seconds']:.1f}", flush=True)
        finish_day(day, protocol, output, identity)
        print(f"temporal_attention_dates_frozen={count}/{len(protocol['dates'])}", flush=True)
    write_json(output / "summary.json", reveal(output, protocol, identity))
    print(f"temporal_attention_development_complete {output / 'summary.json'}", flush=True)


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
        print("Pass --run for the frozen historical temporal_attention development study.")
