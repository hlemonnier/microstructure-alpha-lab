"""Frozen within-session supervised adaptation with delayed model publication."""

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
from lob_forge.boundary_event_inputs import utc_ms
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_pooled import PooledForecaster
from lob_forge.boundary_prequential import SYMBOLS, VARIANTS, PrequentialState, decoded_release_clock, run_prequential, save_history
from lob_forge.boundary_regime_coverage import calendar_stride_mask, history_cohorts
from lob_forge.boundary_tabicl import select_balanced_context
from run_boundary_confirmation import noon, save_predictions, write_json
from run_boundary_event_clock_screen import check_completed
from run_boundary_tabicl_screen import BASELINES as OLD_BASELINES, BLENDS as OLD_BLENDS, MODEL_SPECS as OLD_MODELS

ROOT = Path(__file__).resolve().parents[1]
BASELINES = (*OLD_BASELINES, *OLD_MODELS, *OLD_BLENDS)
MODEL_SPECS = {f"{representation}_{variant}": (representation, variant)
    for representation in ("observations", "combined") for variant in VARIANTS}
BLENDS = {f"{name}_{suffix}": (name, "deep500_hgb" if suffix == "deep500_hgb" else
    "observations_tree" if representation == "observations" else "combined_hgb")
    for name, (representation, _) in MODEL_SPECS.items() for suffix in ("matched_hgb", "deep500_hgb")}


def prepare(day, protocol, folder):
    started = time.monotonic()
    training_days = history_cohorts(day)["recent4_stride4"]["train_dates"]
    cutoff = utc_ms((date.fromisoformat(day) - timedelta(days=1)).isoformat())
    manifest = json.loads((ROOT / protocol["manifest"]).read_text())
    history_x, history_y, history_t = {}, {}, {}
    stream_x, stream_y, stream_t, stream_release, query_x, query_y, query_t = {}, {}, {}, {}, {}, {}, {}
    for current in [*training_days, day]:
        sessions = [r for r in manifest["sessions"] if r["session_date"] == current]
        if len(sessions) != 2 or {r["symbol"] for r in sessions} != set(SYMBOLS):
            raise ValueError("Both registered assets required on every source date")
        partition = folder / "inputs" / f"{current}.json"
        write_json(partition, {"sessions": sessions})
        x, y, times = load_combined_inputs(ROOT, partition, ROOT / protocol["depth_manifest"], ROOT / protocol["spot_manifest"])
        for symbol in SYMBOLS:
            key = symbol, current
            raw = next(r for r in sessions if r["symbol"] == symbol)
            resolved = pd.read_parquet(ROOT / raw["features_path"], columns=["decision_time", "label", "future_event_time"]).set_index("decision_time").loc[times[key]]
            np.testing.assert_array_equal(resolved.label.to_numpy(), y[key])
            releases = decoded_release_clock(resolved.future_event_time.to_numpy())
            if current in training_days:
                mask = calendar_stride_mask(times[key], utc_ms(current), stride_seconds=4)
                if (releases[mask] >= cutoff).any():
                    raise ValueError("Historical replay labels require past-only releases")
                history_x[key], history_y[key], history_t[key] = x[key].loc[mask].copy(), y[key][mask].copy(), times[key][mask].copy()
            else:
                stream = (times[key] >= utc_ms(day, "10:30:00")) & (times[key] < utc_ms(day, "14:00:00"))
                stream_x[symbol], stream_y[symbol] = x[key].loc[stream].copy(), y[key][stream].copy()
                stream_t[symbol], stream_release[symbol] = times[key][stream].copy(), releases[stream].copy()
                test = noon(times[key], day)
                if test.sum() != 7070:
                    raise ValueError("Every original noon observation must remain eligible")
                query_x[symbol], query_y[symbol], query_t[symbol] = x[key].loc[test].copy(), y[key][test].copy(), times[key][test].copy()
        del x, y, times, resolved
        gc.collect()
    hx = {s: pd.concat([history_x[s, d] for d in training_days], ignore_index=True) for s in SYMBOLS}
    hy = {s: np.concatenate([history_y[s, d] for d in training_days]) for s in SYMBOLS}
    ht = {s: np.concatenate([history_t[s, d] for d in training_days]) for s in SYMBOLS}
    selected, population, sampled = select_balanced_context(hy, requested_rows=6144)
    replay_x, replay_y = {s: hx[s].iloc[selected[s]].copy() for s in SYMBOLS}, {s: hy[s][selected[s]].copy() for s in SYMBOLS}
    metadata = {"training_dates": training_days, "seconds": time.monotonic() - started, "assets": {s: {
        "historical_rows": len(hy[s]), "historical_class_prior": population[s].tolist(), "replay_rows": len(selected[s]),
        "replay_sampled_prior": sampled[s].tolist(), "replay_decision_times": ht[s][selected[s]].tolist(),
        "stream_rows": len(stream_y[s]), "assessment_rows": len(query_y[s])} for s in SYMBOLS}}
    write_json(folder / "preparation.json", metadata)
    return replay_x, replay_y, population, stream_x, stream_y, stream_t, stream_release, query_x, query_y, query_t


def run_day(day, protocol, output, identity):
    folder = output / "dates" / day
    if (folder / "completed.json").exists():
        return check_completed(folder, identity)
    if folder.exists():
        raise ValueError("Preserve any incomplete prequential attempt and freeze its retry separately")
    folder.mkdir(parents=True)
    started = time.monotonic()
    write_json(folder / "attempt.json", {"date": day, "planned_trajectories": list(MODEL_SPECS), "scores_sealed": True})
    parent = ROOT / protocol["baseline_run"] / "dates" / day
    check_completed(parent, json.loads((parent.parent.parent / "frozen_screen.json").read_text()))
    rx, ry, population, sx, sy, st, release, qx, qy, qt = prepare(day, protocol, folder)
    forecasts = {s: {} for s in SYMBOLS}
    for symbol in SYMBOLS:
        for name in BASELINES:
            with np.load(parent / "predictions" / f"{symbol}_{name}_registered.npz", allow_pickle=False) as saved:
                forecasts[symbol][name] = {k: saved[k].copy() for k in saved.files}
            np.testing.assert_array_equal(qy[symbol], forecasts[symbol][name]["labels"])
            np.testing.assert_array_equal(qt[symbol], forecasts[symbol][name]["decision_times"])
        np.testing.assert_array_equal(population[symbol], forecasts[symbol]["observations_tree"]["train_priors"])
    metadata = {}
    anchors = np.arange(utc_ms(day, "11:00:00"), utc_ms(day, "14:00:00"), 60000, dtype=np.int64)
    if len(anchors) != 180:
        raise ValueError("Exactly 180 fixed minute updates are registered")
    for representation in ("observations", "combined"):
        source = ROOT / protocol["context_run" if representation == "observations" else "basis_run"] / "dates" / day / f"{representation}_neural"
        original = PooledForecaster.load(source)
        columns = original_combined_columns(rx[SYMBOLS[0]]) if representation == "observations" else list(rx[SYMBOLS[0]].columns)
        if original.columns != columns or len(columns) != protocol["representations"][representation]:
            raise ValueError("Original neural feature schema changed")
        for asset, symbol in enumerate(SYMBOLS):
            np.testing.assert_array_equal(original.priors[asset], population[symbol])
            np.testing.assert_array_equal(original.predict_proba(qx[symbol][columns], asset), forecasts[symbol][f"{representation}_neural"]["probabilities"])
        replay = (np.concatenate([original.matrix(rx[s][columns], a) for a, s in enumerate(SYMBOLS)]),
                  np.concatenate([ry[s] for s in SYMBOLS]), np.concatenate([np.full(len(ry[s]), a, dtype=int) for a, s in enumerate(SYMBOLS)]))
        np.savez_compressed(folder / f"{representation}_replay.npz", matrix=replay[0], labels=replay[1], assets=replay[2])
        for variant in VARIANTS:
            name = f"{representation}_{variant}"
            begin = time.monotonic()
            write_json(output / "progress.json", {"date": day, "model": name, "stage": "causal_adaptation", "assessment_scores_sealed": True})
            p, state, history = run_prequential(original, variant, {s: sx[s][columns] for s in SYMBOLS}, sy, st, release,
                {s: qx[s][columns] for s in SYMBOLS}, qt, replay, anchors)
            state.save(folder / name)
            save_history(folder / name / "history.json", history)
            restored = PrequentialState.load(folder / name)
            for asset, symbol in enumerate(SYMBOLS):
                check_matrix = original.matrix(qx[symbol][columns].iloc[-64:], asset)
                np.testing.assert_array_equal(state.predict_matrix(check_matrix, asset), restored.predict_matrix(check_matrix, asset))
                if variant == "frozen":
                    np.testing.assert_array_equal(p[symbol], forecasts[symbol][f"{representation}_neural"]["probabilities"])
                common_prior = forecasts[symbol]["observations_tree"]["train_priors"]
                forecasts[symbol][name] = {"probabilities": p[symbol], "train_priors": common_prior, "decision_priors": common_prior}
            metadata[name] = {"variant": VARIANTS[variant], "representation": representation, "columns": columns,
                "seconds": time.monotonic() - begin, "optimizer_steps": history["optimizer_steps"],
                "max_update_seconds": history.get("max_update_seconds", 0), "last_checkpoint_forecast_parity": True,
                "zero_adaptation_exact": variant == "frozen", "unmodified_original_checkpoint": str(source.relative_to(ROOT))}
            print(f"prequential_trajectory={day}/{name} steps={history['optimizer_steps']} seconds={metadata[name]['seconds']:.1f}", flush=True)
            del state, restored
            gc.collect()
        del original, replay
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
    record = {"identity": identity, "date": day, "model_trajectories": len(MODEL_SPECS), "seconds": time.monotonic() - started,
        "optimizer_steps": sum(v["optimizer_steps"] for v in metadata.values()),
        "all_original_rows_labels_and_saved_controls_exact": True, "zero_adaptation_reproductions_exact": True,
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
        raise ValueError("Incomplete fixed prequential development family")
    board = []
    for name, policy in sorted({(r["model"], r["policy"]) for r in records}):
        rows = [r for r in records if (r["model"], r["policy"]) == (name, policy)]
        board.append({"model": name, "policy": policy, "asset_dates": len(rows),
            "means": {k: float(np.mean([r["metrics"][k] for r in rows])) for k in ("balanced_accuracy", "natural_accuracy", "log_loss", "macro_f1")},
            "balanced_accuracy_by_asset": {s: float(np.mean([r["metrics"]["balanced_accuracy"] for r in rows if r["symbol"] == s])) for s in SYMBOLS}})
    return {"identity": identity, "evidence_status": protocol["evidence_status"], "model_trajectories": len(MODEL_SPECS) * len(protocol["dates"]),
        "post_fit_panels": len(records), "substantial_gain_confirmed": False,
        "leaderboard": sorted(board, key=lambda r: r["means"]["balanced_accuracy"], reverse=True), "records": records}


def run(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if protocol["evidence_status"] != "development_only_on_exposed_dates" or protocol["trajectories"] != list(MODEL_SPECS) or protocol["variants"] != VARIANTS or protocol["decision_policies"] != ["registered", "forecast_3600"]:
        raise ValueError("Only the frozen delayed-feedback development family is registered")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen prequential input changed: {path}")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"], "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "torch", "pyarrow", "scipy")}}
    frozen = output / "frozen_screen.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve this prequential experiment after an identity change")
    write_json(frozen, identity)
    for count, day in enumerate(protocol["dates"], 1):
        record = run_day(day, protocol, output, identity)
        print(f"prequential_dates_frozen={count}/{len(protocol['dates'])} seconds={record['seconds']:.1f}", flush=True)
    write_json(output / "summary.json", reveal(output, protocol, identity))
    print(f"prequential_development_complete {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.protocol.resolve(), args.output.resolve())
    else:
        print("Pass --run for the frozen delayed-feedback adaptation experiment.")
