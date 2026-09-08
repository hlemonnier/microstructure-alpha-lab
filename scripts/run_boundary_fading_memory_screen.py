"""Compare instantaneous, exponential and contractive recurrent market memory."""

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

import joblib
import numpy as np
import pandas as pd

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_adaptive_priors import forecast_priors
from lob_forge.boundary_combined_inputs import load_combined_inputs, original_combined_columns
from lob_forge.boundary_confirm_model import SYMBOLS, fit_confirm_neural
from lob_forge.boundary_event_inputs import utc_ms
from lob_forge.boundary_fading_memory import MECHANISMS, FadingMemoryMap, append_memory, observe_selected
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_pooled import PooledForecaster
from lob_forge.boundary_regime_coverage import calendar_stride_mask, history_cohorts
from lob_forge.boundary_weighted_boost import fit_weighted_boost
from run_boundary_confirmation import noon, save_predictions, write_json
from run_boundary_event_clock_screen import check_completed
from run_boundary_prequential_screen import BASELINES as OLD_BASELINES, BLENDS as OLD_BLENDS, MODEL_SPECS as OLD_MODELS

ROOT = Path(__file__).resolve().parents[1]
BASELINES = (*OLD_BASELINES, *OLD_MODELS, *OLD_BLENDS)
MODEL_SPECS = {f"{representation}_{mechanism}_{learner}": (representation, mechanism, learner)
    for representation in ("observations", "combined") for mechanism in MECHANISMS for learner in ("hgb", "neural")}
BLENDS = {f"{representation}_{mechanism}_blend": (f"{representation}_{mechanism}_hgb", f"{representation}_{mechanism}_neural")
    for representation in ("observations", "combined") for mechanism in MECHANISMS}
BLENDS.update({f"{name}_{suffix}": (name, "observations_neural" if suffix == "old_neural" else "deep500_hgb")
    for name in MODEL_SPECS for suffix in ("old_neural", "deep500_hgb")})


def prepare(day, representation, original, protocol, folder):
    started = time.monotonic()
    training_days = history_cohorts(day)["recent4_stride4"]["train_dates"]
    validation_day = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    manifest = json.loads((ROOT / protocol["manifest"]).read_text())
    generator = FadingMemoryMap.from_checkpoint(original)
    generator.save(folder / representation / "generator.joblib")
    restored = FadingMemoryMap.load(folder / representation / "generator.joblib")
    tx, ty, tm, vx, vy, vm, qx, qy, qt, qm, records = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, []
    for current in [*training_days, validation_day, day]:
        sessions = [r for r in manifest["sessions"] if r["session_date"] == current]
        if len(sessions) != 2 or {r["symbol"] for r in sessions} != set(SYMBOLS):
            raise ValueError("Both registered assets required for every memory source date")
        partition = folder / "inputs" / f"{current}.json"
        write_json(partition, {"sessions": sessions})
        x, y, times = load_combined_inputs(ROOT, partition, ROOT / protocol["depth_manifest"], ROOT / protocol["spot_manifest"])
        for asset, symbol in enumerate(SYMBOLS):
            key = symbol, current
            columns = original_combined_columns(x[key]) if representation == "observations" else list(x[key].columns)
            if columns != original.columns or len(columns) != protocol["representations"][representation]:
                raise ValueError("Original generator input schema changed")
            frame = x[key][columns]
            selected = calendar_stride_mask(times[key], utc_ms(current), stride_seconds=4) if current in training_days else noon(times[key], current)
            if current not in training_days and selected.sum() != 7070:
                raise ValueError("Every original noon row must remain eligible")
            raw = next(r for r in sessions if r["symbol"] == symbol)
            released = pd.read_parquet(ROOT / raw["features_path"], columns=["decision_time", "label", "future_event_time"]).set_index("decision_time").loc[times[key][selected]]
            np.testing.assert_array_equal(released.label.to_numpy(), y[key][selected])
            cutoff = utc_ms(validation_day if current in training_days else day)
            if current != day and (released.future_event_time.to_numpy() >= cutoff).any():
                raise ValueError("Readout training and validation require strictly past released labels")
            prefix = min(64, len(frame))
            np.testing.assert_array_equal(generator.input_matrix(frame.iloc[:prefix], asset), original.matrix(frame.iloc[:prefix], asset))
            a, _, _ = generator.observe(frame.iloc[:prefix], times[key][:prefix], asset)
            b, _, _ = restored.observe(frame.iloc[:prefix], times[key][:prefix], asset)
            for mechanism in MECHANISMS:
                np.testing.assert_array_equal(a[mechanism], b[mechanism])
            values, record = observe_selected(generator, frame, times[key], asset, selected)
            state_path = folder / representation / "states" / f"{symbol}_{current}.npz"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(state_path, decision_times=times[key][selected], **values)
            records.append({"symbol": symbol, "date": current, **record, "generator_reload_exact": True,
                "original_normalized_input_exact": True, "states_sha256": sha256_file(state_path)})
            if current in training_days:
                tx[key], ty[key], tm[key] = frame.loc[selected].reset_index(drop=True), y[key][selected].copy(), values
            elif current == validation_day:
                vx[symbol], vy[symbol], vm[symbol] = frame.loc[selected].reset_index(drop=True), y[key][selected].copy(), values
            else:
                qx[symbol], qy[symbol], qt[symbol], qm[symbol] = frame.loc[selected].reset_index(drop=True), y[key][selected].copy(), times[key][selected].copy(), values
        del x, y, times, frame, values, released
        gc.collect()
    tx = {s: pd.concat([tx[s, d] for d in training_days], ignore_index=True) for s in SYMBOLS}
    ty = {s: np.concatenate([ty[s, d] for d in training_days]) for s in SYMBOLS}
    tm = {s: {m: np.concatenate([tm[s, d][m] for d in training_days]) for m in MECHANISMS} for s in SYMBOLS}
    write_json(folder / representation / "preparation.json", {"training_dates": training_days, "validation_date": validation_day,
        "seconds": time.monotonic() - started, "sessions": records})
    return tx, ty, tm, vx, vy, vm, qx, qy, qt, qm


def run_day(day, protocol, output, identity):
    folder = output / "dates" / day
    if (folder / "completed.json").exists():
        return check_completed(folder, identity)
    if folder.exists():
        raise ValueError("Preserve an incomplete memory attempt and freeze its retry separately")
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
        write_json(output / "progress.json", {"date": day, "representation": representation, "stage": "past_only_memory", "assessment_scores_sealed": True})
        source = ROOT / protocol["context_run" if representation == "observations" else "basis_run"] / "dates" / day / f"{representation}_neural"
        original = PooledForecaster.load(source)
        tx, ty, tm, vx, vy, vm, qx, qy, qt, qm = prepare(day, representation, original, protocol, folder)
        for asset, symbol in enumerate(SYMBOLS):
            for saved in forecasts[symbol].values():
                np.testing.assert_array_equal(qy[symbol], saved["labels"])
                np.testing.assert_array_equal(qt[symbol], saved["decision_times"])
            expected = np.array([(ty[symbol] == k).mean() for k in (-1, 0, 1)])
            np.testing.assert_array_equal(expected, original.priors[asset])
            np.testing.assert_array_equal(expected, forecasts[symbol]["observations_tree"]["train_priors"])
            np.testing.assert_array_equal(original.predict_proba(qx[symbol], asset), forecasts[symbol][f"{representation}_neural"]["probabilities"])
        for mechanism in MECHANISMS:
            train = {s: append_memory(tx[s], tm[s][mechanism]) for s in SYMBOLS}
            validation = {s: append_memory(vx[s], vm[s][mechanism]) for s in SYMBOLS}
            query = {s: append_memory(qx[s], qm[s][mechanism]) for s in SYMBOLS}
            for learner in ("hgb", "neural"):
                name = f"{representation}_{mechanism}_{learner}"
                begin = time.monotonic()
                write_json(output / "progress.json", {"date": day, "model": name, "stage": "training", "assessment_scores_sealed": True})
                history = {}
                if learner == "hgb":
                    model = fit_weighted_boost(train, ty, leaves=7, class_balanced=True)
                    joblib.dump(model, folder / f"{name}.joblib")
                    restored = joblib.load(folder / f"{name}.joblib")
                else:
                    model, history = fit_confirm_neural(train, ty, validation, vy)
                    model.save(folder / name)
                    write_json(folder / name / "training_history.json", history)
                    restored = PooledForecaster.load(folder / name)
                for asset, symbol in enumerate(SYMBOLS):
                    identifier = symbol if learner == "hgb" else asset
                    p = model.predict_proba(query[symbol], identifier)
                    np.testing.assert_array_equal(p, restored.predict_proba(query[symbol], identifier))
                    common = forecasts[symbol]["observations_tree"]["train_priors"]
                    np.testing.assert_array_equal(model.priors[asset], common)
                    forecasts[symbol][name] = {**forecasts[symbol]["observations_tree"], "probabilities": p, "decision_priors": common}
                metadata[name] = {"representation": representation, "mechanism": mechanism, "learner": learner,
                    "columns": model.columns, "training_rows": {s: len(ty[s]) for s in SYMBOLS},
                    "seconds": time.monotonic() - begin, "best_epoch": history.get("best_epoch"), "checkpoint_parity": True}
                print(f"memory_fit={day}/{name} seconds={metadata[name]['seconds']:.1f}", flush=True)
                del model, restored
                gc.collect()
            del train, validation, query
        del tx, ty, tm, vx, vy, vm, qx, qm, original
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
        raise ValueError("Incomplete fixed fading-memory development family")
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
        raise ValueError("Only the frozen three-mechanism development family is registered")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen fading-memory input changed: {path}")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"], "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "torch", "pyarrow", "scipy")}}
    frozen = output / "frozen_screen.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve this fading-memory experiment after an identity change")
    write_json(frozen, identity)
    for count, day in enumerate(protocol["dates"], 1):
        record = run_day(day, protocol, output, identity)
        print(f"memory_dates_frozen={count}/{len(protocol['dates'])} seconds={record['seconds']:.1f}", flush=True)
    write_json(output / "summary.json", reveal(output, protocol, identity))
    print(f"memory_development_complete {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.protocol.resolve(), args.output.resolve())
    else:
        print("Pass --run for the frozen fading-memory development experiment.")
