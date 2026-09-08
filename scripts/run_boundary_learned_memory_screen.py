"""Fit frozen neural, GRU and LSTM logit corrections on historical streams."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from datetime import date, timedelta
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_adaptive_priors import forecast_priors
from lob_forge.boundary_combined_inputs import load_combined_inputs, original_combined_columns
from lob_forge.boundary_confirm_model import SYMBOLS
from lob_forge.boundary_event_inputs import utc_ms
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_learned_memory import ARCHITECTURES, LearnedMemoryForecaster, SequenceInputs, TrainingGrid, fit_learned_memory, matrix_logits, predict_deltas
from lob_forge.boundary_pooled import PooledForecaster
from lob_forge.boundary_prequential import decoded_release_clock
from lob_forge.boundary_regime_coverage import calendar_stride_mask, history_cohorts
from run_boundary_confirmation import noon, save_predictions, write_json
from run_boundary_event_clock_screen import check_completed
from run_boundary_fading_memory_screen import BASELINES as OLD_BASELINES, BLENDS as OLD_BLENDS, MODEL_SPECS as OLD_MODELS

ROOT = Path(__file__).resolve().parents[1]
BASELINES = (*OLD_BASELINES, *OLD_MODELS, *OLD_BLENDS)
MODEL_SPECS = {f"{representation}_{architecture}": (representation, architecture)
    for representation in ("observations", "combined") for architecture in ARCHITECTURES}
BLENDS = {f"{name}_{suffix}": (name, f"{representation}_tree" if representation == "observations" else "combined_hgb")
    for name, (representation, _) in MODEL_SPECS.items() for suffix in ("matched_hgb",)}
BLENDS.update({f"{name}_deep500_hgb": (name, "deep500_hgb") for name in MODEL_SPECS})


def training_storage(folder, dimensions):
    folder.mkdir(parents=True)
    shape = (8, 86400)
    fields = {}
    for name, dtype, dimensions_ in [("matrix", np.float32, (*shape, dimensions)), ("base_logits", np.float32, (*shape, 3)),
        ("labels", np.int8, shape), ("weights", np.float32, shape), ("available", bool, shape), ("supervised", bool, shape)]:
        field = np.lib.format.open_memmap(folder / f"{name}.npy", mode="w+", dtype=dtype, shape=dimensions_)
        field[:] = -2 if name == "labels" else 0
        fields[name] = field
    return TrainingGrid(**fields, assets=np.repeat([0, 1], 4))


def prepare(day, representation, original, protocol, folder):
    begin = time.monotonic()
    training_days = history_cohorts(day)["recent4_stride4"]["train_dates"]
    validation_day = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    manifest = json.loads((ROOT / protocol["manifest"]).read_text())
    storage = folder / representation / "training"
    grid = training_storage(storage, int(original.active.sum()) + 1)
    validation, validation_labels, query, query_labels, records = {}, {}, {}, {}, []
    for current in [*training_days, validation_day, day]:
        sessions = [r for r in manifest["sessions"] if r["session_date"] == current]
        if len(sessions) != 2 or {r["symbol"] for r in sessions} != set(SYMBOLS):
            raise ValueError("Both original assets required on every registered source date")
        partition = folder / "inputs" / f"{current}.json"
        write_json(partition, {"sessions": sessions})
        x, y, times = load_combined_inputs(ROOT, partition, ROOT / protocol["depth_manifest"], ROOT / protocol["spot_manifest"])
        for asset, symbol in enumerate(SYMBOLS):
            key = symbol, current
            columns = original_combined_columns(x[key]) if representation == "observations" else list(x[key].columns)
            if columns != original.columns or len(columns) != protocol["representations"][representation]:
                raise ValueError("Original learned-memory feature schema changed")
            frame, clock = x[key][columns], times[key]
            selected = calendar_stride_mask(clock, utc_ms(current), stride_seconds=4) if current in training_days else noon(clock, current)
            raw = next(r for r in sessions if r["symbol"] == symbol)
            released = pd.read_parquet(ROOT / raw["features_path"], columns=["decision_time", "label", "future_event_time"]).set_index("decision_time").loc[clock[selected]]
            np.testing.assert_array_equal(released.label.to_numpy(), y[key][selected])
            release_clock = decoded_release_clock(released.future_event_time.to_numpy())
            cutoff = utc_ms(validation_day if current in training_days else day)
            if current != day and (release_clock >= cutoff).any():
                raise ValueError("Historical and validation outcomes must precede the next partition")
            if current in training_days:
                seconds, remainder = np.divmod(clock - utc_ms(current), 1000)
                if remainder.any() or (seconds < 0).any() or (seconds >= 86400).any() or len(np.unique(seconds)) != len(seconds):
                    raise ValueError("Training streams require exact unique within-day integer seconds")
                stream = asset * 4 + training_days.index(current)
                transformed = original.matrix(frame, asset)
                grid.matrix[stream, seconds] = transformed
                grid.available[stream, seconds] = True
                grid.supervised[stream, seconds[selected]] = True
                grid.labels[stream, seconds] = y[key]
                grid.base_logits[stream, seconds[selected]] = matrix_logits(original, transformed[selected])
                np.testing.assert_array_equal(selected, seconds % 4 == 0)
            else:
                if selected.sum() != 7070:
                    raise ValueError("All 7070 original noon observations must be preserved")
                observed = clock <= clock[selected][-1]
                prediction_frame = frame.loc[selected].copy()
                matrix = original.matrix(frame.loc[observed].copy(), asset)
                sequence = SequenceInputs(matrix, clock[observed].copy(), selected[observed].copy(),
                    matrix_logits(original, matrix[selected[observed]]), original.predict_proba(prediction_frame, asset))
                sequence.validate(int(original.active.sum()) + 1)
                destination = folder / representation / f"{symbol}_{current}_sequence.npz"
                sequence.save(destination)
                restored = SequenceInputs.load(destination)
                for field in ("matrix", "times", "selected", "base_logits", "base_probabilities"):
                    np.testing.assert_array_equal(getattr(restored, field), getattr(sequence, field))
                if current == validation_day:
                    validation[symbol], validation_labels[symbol] = sequence, y[key][selected].copy()
                else:
                    query[symbol], query_labels[symbol] = sequence, y[key][selected].copy()
            records.append({"date": current, "symbol": symbol, "observed_rows": len(frame), "selected_rows": int(selected.sum()),
                "selected_label_max_release_time": int(release_clock.max()), "past_label_cutoff": cutoff if current != day else None})
        del x, y, times, frame
        gc.collect()
    total = int(grid.supervised.sum())
    for asset in (0, 1):
        for label in (-1, 0, 1):
            keep = grid.supervised & (grid.assets[:, None] == asset) & (grid.labels == label)
            if not keep.any():
                raise ValueError("Every original historical asset/class cell must be represented")
            grid.weights[keep] = total / (6 * int(keep.sum()))
    grid.validate(int(original.active.sum()) + 1, original.priors)
    for name in ("matrix", "base_logits", "labels", "weights", "available", "supervised"):
        getattr(grid, name).flush()
    write_json(folder / representation / "preparation.json", {"training_dates": training_days, "validation_date": validation_day,
        "stream_assets": grid.assets.tolist(), "stream_date_order": [*training_days, *training_days], "sessions": records,
        "total_supervised_rows": total, "seconds": time.monotonic() - begin,
        "assessment_labels_excluded_from_fit_interface": True, "all_original_priors_exact": True})
    return grid, validation, validation_labels, query, query_labels


def run_day(day, protocol, output, identity):
    folder = output / "dates" / day
    if (folder / "completed.json").exists():
        return check_completed(folder, identity)
    if folder.exists():
        raise ValueError("Preserve an incomplete learned-memory attempt and freeze its retry separately")
    folder.mkdir(parents=True)
    started = time.monotonic()
    write_json(folder / "attempt.json", {"date": day, "planned_models": list(MODEL_SPECS), "scores_sealed": True})
    parent = ROOT / protocol["baseline_run"] / "dates" / day
    check_completed(parent, json.loads((parent.parent.parent / "frozen_screen.json").read_text()))
    forecasts = {s: {} for s in SYMBOLS}
    for symbol in SYMBOLS:
        for name in BASELINES:
            with np.load(parent / "predictions" / f"{symbol}_{name}_registered.npz", allow_pickle=False) as saved:
                forecasts[symbol][name] = {k: saved[k].copy() for k in saved.files}
    metadata = {}
    for representation in ("observations", "combined"):
        write_json(output / "progress.json", {"date": day, "representation": representation, "stage": "historical_stream_preparation", "assessment_scores_sealed": True})
        source = ROOT / protocol["context_run" if representation == "observations" else "basis_run"] / "dates" / day / f"{representation}_neural"
        original = PooledForecaster.load(source)
        training, validation, vy, query, qy = prepare(day, representation, original, protocol, folder)
        qt = {s: query[s].times[query[s].selected] for s in SYMBOLS}
        for asset, symbol in enumerate(SYMBOLS):
            for saved in forecasts[symbol].values():
                np.testing.assert_array_equal(qy[symbol], saved["labels"])
                np.testing.assert_array_equal(qt[symbol], saved["decision_times"])
            np.testing.assert_array_equal(original.priors[asset], forecasts[symbol]["observations_tree"]["train_priors"])
            np.testing.assert_array_equal(query[symbol].base_probabilities, forecasts[symbol][f"{representation}_neural"]["probabilities"])
        for architecture in ARCHITECTURES:
            name = f"{representation}_{architecture}"
            begin = time.monotonic()
            write_json(output / "progress.json", {"date": day, "model": name, "stage": "historical_training", "assessment_scores_sealed": True})
            model, history = fit_learned_memory(original, training, validation, vy, architecture=architecture, device=protocol["selected_backend"])
            model.save(folder / name)
            write_json(folder / name / "training_history.json", history)
            restored = LearnedMemoryForecaster.load(folder / name, device=protocol["selected_backend"])
            checks = {}
            for asset, symbol in enumerate(SYMBOLS):
                seq = query[symbol]
                p, inference = seq.predict(model.adapter, original.priors[asset])
                checkpoint, _ = seq.predict(restored.adapter, restored.original.priors[asset])
                np.testing.assert_array_equal(p, checkpoint)
                # Market-query causality probe uses no outcomes. Keep the prefix
                # fixed while perturbing later normalized observations.
                prefix_end = int(np.flatnonzero(seq.selected)[63]) + 1
                prefix_mask = seq.selected[:prefix_end]
                unchanged, _ = predict_deltas(model.adapter, seq.matrix[:prefix_end], seq.times[:prefix_end], selected=prefix_mask)
                perturbed = seq.matrix[:prefix_end + 32].copy()
                perturbed[prefix_end:] = 1e4
                future_mask = np.r_[prefix_mask, np.zeros(len(perturbed) - prefix_end, dtype=bool)]
                future, _ = predict_deltas(model.adapter, perturbed, seq.times[:len(perturbed)], selected=future_mask)
                error = float(np.max(np.abs(unchanged - future)))
                if error > protocol["market_future_prefix_absolute_tolerance"]:
                    raise ValueError("Market learned-memory future-prefix invariance failed")
                common = forecasts[symbol]["observations_tree"]["train_priors"]
                forecasts[symbol][name] = {**forecasts[symbol]["observations_tree"], "probabilities": p, "decision_priors": common}
                checks[symbol] = {"checkpoint_forecast_exact": True, "future_prefix_max_absolute_error": error, "inference": inference}
            metadata[name] = {"representation": representation, "architecture": architecture, "best_epoch": history["best_epoch"],
                "backend": history["backend"], "optimizer_steps": history["optimizer_steps"], "trainable_parameters": history["trainable_parameters"],
                "original_parameters_frozen": history["original_parameters_frozen"], "training_rows": int(training.supervised.sum()),
                "checks": checks, "seconds": time.monotonic() - begin}
            print(f"learned_memory_fit={day}/{name} seconds={metadata[name]['seconds']:.1f}", flush=True)
            del model, restored
            gc.collect()
        del original, training, validation, vy, query
        gc.collect()
    for symbol in SYMBOLS:
        for name, (left, right) in BLENDS.items():
            a, b = forecasts[symbol][left], forecasts[symbol][right]
            np.testing.assert_array_equal(a["train_priors"], b["train_priors"])
            forecasts[symbol][name] = {**a, "probabilities": (a["probabilities"] + b["probabilities"]) / 2}
        with np.load(ROOT / protocol["prior_run"] / "priors" / day / f"{symbol}_original_reference.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["decision_times"], qt[symbol])
            initial = saved["labels_cumulative"][0].copy()
        for name, values in forecasts[symbol].items():
            for policy in protocol["decision_policies"]:
                pi = values["decision_priors"] if policy == "registered" else forecast_priors(values["probabilities"], qt[symbol], initial, half_life_seconds=3600)
                save_predictions(folder / "predictions" / f"{symbol}_{name}_{policy}.npz", values["probabilities"], qy[symbol], qt[symbol], values["train_priors"], pi)
                if name in BASELINES:
                    with np.load(parent / "predictions" / f"{symbol}_{name}_{policy}.npz", allow_pickle=False) as old:
                        np.testing.assert_array_equal(old["probabilities"], values["probabilities"])
                        np.testing.assert_array_equal(old["decision_priors"], pi)
    write_json(folder / "fit_metadata.json", metadata)
    record = {"identity": identity, "date": day, "model_fits": len(MODEL_SPECS), "seconds": time.monotonic() - started,
        "all_original_rows_labels_priors_and_saved_controls_exact": True, "original_checkpoint_predictions_exact": True,
        "artifact_hashes": {str(p.relative_to(folder)): sha256_file(p) for p in folder.rglob("*") if p.is_file()}}
    write_json(folder / "completed.json", record)
    return record


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
        raise ValueError("Incomplete learned-memory development family")
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
    import torch

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads(protocol_path.read_text())
    if protocol["evidence_status"] != "development_only_on_exposed_dates" or protocol["models"] != list(MODEL_SPECS) or protocol["decision_policies"] != ["registered", "forecast_3600"]:
        raise ValueError("Only the frozen learned-memory family is registered")
    if protocol["selected_backend"] not in ("cpu", "mps"):
        raise ValueError("A validated common backend is required")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen learned-memory input changed: {path}")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"], "python": platform.python_version(),
        "backend": protocol["selected_backend"], "dependencies": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "torch", "pyarrow", "scipy")}}
    frozen = output / "frozen_screen.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve this learned-memory experiment after an identity change")
    write_json(frozen, identity)
    for count, day in enumerate(protocol["dates"], 1):
        record = run_day(day, protocol, output, identity)
        print(f"learned_memory_dates_frozen={count}/{len(protocol['dates'])} seconds={record['seconds']:.1f}", flush=True)
    write_json(output / "summary.json", reveal(output, protocol, identity))
    print(f"learned_memory_development_complete {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.protocol.resolve(), args.output.resolve())
    else:
        print("Pass --run for the registered learned-memory development experiment.")
