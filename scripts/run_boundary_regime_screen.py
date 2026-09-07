"""Compare historical diversity with recent-data and approximate row-budget controls."""

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
from threadpoolctl import threadpool_limits

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_adaptive_priors import forecast_priors
from lob_forge.boundary_confirm_model import SYMBOLS, fit_confirm_neural
from lob_forge.boundary_event_clock_inputs import load_clock_inputs
from lob_forge.boundary_event_inputs import utc_ms
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_pooled import PooledForecaster
from lob_forge.boundary_regime_coverage import calendar_stride_mask, history_cohorts
from lob_forge.boundary_weighted_boost import fit_weighted_boost
from run_boundary_confirmation import noon, save_predictions, write_json
from run_boundary_event_clock_screen import check_completed

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ("original_reference", "original_blend", "clock_tree_7", "clock_neural", "clock_neural_clock_tree_7")


def run_fold(day, protocol, output, identity):
    folder = output / "dates" / day
    if (folder / "completed.json").exists():
        return check_completed(folder, identity)
    started = time.monotonic()
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / f"attempt_{time.time_ns()}.json", {"day": day, "model_fits": 6})
    source = ROOT / protocol["event_run"] / "dates" / day
    check_completed(source, json.loads((source.parent.parent / "frozen_screen.json").read_text()))
    cohorts = history_cohorts(day)
    validation_day = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    manifest = json.loads((ROOT / protocol["manifest"]).read_text())
    selected_x, selected_y, validation_x, validation_y, test_x, test_y, test_times = {}, {}, {}, {}, {}, {}, {}
    required = sorted({d for cohort in cohorts.values() for d in cohort["train_dates"]} | {validation_day, day})
    # Build dense causal features one day at a time, then retain only training
    # clock samples. The wider study never holds all fourteen dense days in RAM.
    for current in required:
        partition = folder / "inputs" / f"{current}.json"
        sessions = [r for r in manifest["sessions"] if r["session_date"] == current]
        if len(sessions) != 2 or {r["symbol"] for r in sessions} != set(SYMBOLS):
            raise ValueError("Every registered historical day requires both assets")
        write_json(partition, {"sessions": sessions})
        x, y, times = load_clock_inputs(ROOT, partition)
        for symbol in SYMBOLS:
            if current == validation_day:
                mask = noon(times[symbol, current], current)
                validation_x[symbol], validation_y[symbol] = x[symbol, current].loc[mask].copy(), y[symbol, current][mask]
            elif current == day:
                mask = noon(times[symbol, current], current)
                test_x[symbol] = x[symbol, current].loc[mask].copy()
                test_y[symbol], test_times[symbol] = y[symbol, current][mask], times[symbol, current][mask]
            else:
                for stride in (4, 14):
                    mask = calendar_stride_mask(times[symbol, current], utc_ms(current), stride_seconds=stride)
                    selected_x[symbol, current, stride] = x[symbol, current].loc[mask].copy()
                    selected_y[symbol, current, stride] = y[symbol, current][mask]
        del x, y, times
        gc.collect()
    baseline = {}
    original_tree, original_neural = joblib.load(source / "clock_tree_7.joblib"), PooledForecaster.load(source / "clock_neural")
    for asset, symbol in enumerate(SYMBOLS):
        baseline[symbol] = {}
        for name in BASELINES:
            with np.load(source / "predictions" / f"{symbol}_{name}_registered.npz") as archive:
                baseline[symbol][name] = {k: archive[k].copy() for k in archive.files}
        np.testing.assert_array_equal(test_y[symbol], baseline[symbol]["original_blend"]["labels"])
        np.testing.assert_array_equal(test_times[symbol], baseline[symbol]["original_blend"]["decision_times"])
        np.testing.assert_array_equal(original_tree.predict_proba(test_x[symbol], symbol), baseline[symbol]["clock_tree_7"]["probabilities"])
        np.testing.assert_array_equal(original_neural.predict_proba(test_x[symbol], asset), baseline[symbol]["clock_neural"]["probabilities"])
    del original_tree, original_neural
    metadata = {}
    for cohort_name, cohort in cohorts.items():
        training_x = {s: pd.concat([selected_x[s, d, cohort["stride_seconds"]] for d in cohort["train_dates"]], ignore_index=True) for s in SYMBOLS}
        training_y = {s: np.concatenate([selected_y[s, d, cohort["stride_seconds"]] for d in cohort["train_dates"]]) for s in SYMBOLS}
        for kind in ("tree", "neural"):
            name = cohort_name + "_" + kind
            begin_fit = time.monotonic()
            write_json(output / "progress.json", {"day": day, "stage": "training", "model": name, "scores_sealed": True})
            if kind == "tree":
                with threadpool_limits(limits=2):
                    model = fit_weighted_boost(training_x, training_y, leaves=7, class_balanced=True)
                joblib.dump(model, folder / f"{name}.joblib")
                restored = joblib.load(folder / f"{name}.joblib")
            else:
                model, history = fit_confirm_neural(training_x, training_y, validation_x, validation_y)
                model.save(folder / name)
                write_json(folder / name / "training_history.json", history)
                restored = PooledForecaster.load(folder / name)
            for asset, symbol in enumerate(SYMBOLS):
                identifier = symbol if kind == "tree" else asset
                p = restored.predict_proba(test_x[symbol], identifier)
                np.testing.assert_array_equal(p, model.predict_proba(test_x[symbol], identifier))
                previous = np.array([(validation_y[symbol] == k).mean() for k in (-1, 0, 1)])
                baseline[symbol][name] = {**baseline[symbol]["original_blend"], "probabilities": p,
                    "train_priors": restored.priors[asset].copy(), "decision_priors": (restored.priors[asset] + previous) / 2}
            metadata[name] = {"cohort": cohort, "train_rows": {s: len(training_y[s]) for s in SYMBOLS}, "features": restored.columns,
                              "seconds": time.monotonic() - begin_fit, "checkpoint_parity": True}
            print(f"regime_fit={day}/{name} seconds={metadata[name]['seconds']:.1f}", flush=True)
            del model, restored
            gc.collect()
        for symbol in SYMBOLS:
            a, b = baseline[symbol][cohort_name + "_tree"], baseline[symbol][cohort_name + "_neural"]
            np.testing.assert_array_equal(a["train_priors"], b["train_priors"])
            baseline[symbol][cohort_name + "_blend"] = {**a, "probabilities": (a["probabilities"] + b["probabilities"]) / 2}
        del training_x, training_y
        gc.collect()
    for symbol in SYMBOLS:
        with np.load(ROOT / protocol["prior_run"] / "priors" / day / f"{symbol}_original_reference.npz") as archive:
            np.testing.assert_array_equal(archive["decision_times"], baseline[symbol]["original_blend"]["decision_times"])
            initial = archive["labels_cumulative"][0].copy()
        for name, values in baseline[symbol].items():
            for policy in ("registered", "forecast_3600"):
                pi = values["decision_priors"] if policy == "registered" else forecast_priors(values["probabilities"], values["decision_times"], initial, half_life_seconds=3600)
                save_predictions(folder / "predictions" / f"{symbol}_{name}_{policy}.npz", values["probabilities"], values["labels"], values["decision_times"], values["train_priors"], pi)
    write_json(folder / "fit_metadata.json", metadata)
    record = {"identity": identity, "date": day, "model_fits": 6, "seconds": time.monotonic() - started,
              "original_checkpoint_parity": True, "artifact_hashes": {str(p.relative_to(folder)): sha256_file(p) for p in folder.rglob("*") if p.is_file()}}
    write_json(folder / "completed.json", record)
    return record


def reveal(output, protocol, identity):
    records = []
    for day in protocol["dates"]:
        folder = output / "dates" / day
        check_completed(folder, identity)
        names = [*BASELINES, *(name + "_" + kind for name in history_cohorts(day) for kind in ("tree", "neural", "blend"))]
        for symbol in SYMBOLS:
            for name in names:
                for policy in ("registered", "forecast_3600"):
                    with np.load(folder / "predictions" / f"{symbol}_{name}_{policy}.npz") as archive:
                        metrics = classification_metrics(SimpleNamespace(priors=archive["decision_priors"]), archive["probabilities"], archive["labels"])
                    records.append({"date": day, "symbol": symbol, "model": name, "policy": policy, "metrics": metrics})
    if len(records) != protocol["post_fit_panels"]:
        raise ValueError("Incomplete history-breadth development family")
    index = {(r["date"], r["symbol"], r["model"], r["policy"]): r for r in records}
    leaderboard = []
    for name, policy in sorted({(r["model"], r["policy"]) for r in records}):
        rows = [r for r in records if (r["model"], r["policy"]) == (name, policy)]
        leaderboard.append({"model": name, "policy": policy, "asset_dates": len(rows),
            "means": {k: float(np.mean([r["metrics"][k] for r in rows])) for k in ("balanced_accuracy", "natural_accuracy", "log_loss", "macro_f1")},
            "delta_to_same_policy_recent_full_clock_blend": float(np.mean([r["metrics"]["balanced_accuracy"] - index[r["date"], r["symbol"], "clock_neural_clock_tree_7", policy]["metrics"]["balanced_accuracy"] for r in rows])),
            "balanced_accuracy_by_asset": {s: float(np.mean([r["metrics"]["balanced_accuracy"] for r in rows if r["symbol"] == s])) for s in SYMBOLS}})
    return {"identity": identity, "evidence_status": protocol["evidence_status"], "model_fits": 6 * len(protocol["dates"]), "post_fit_panels": len(records), "substantial_gain_confirmed": False,
            "leaderboard": sorted(leaderboard, key=lambda r: r["means"]["balanced_accuracy"], reverse=True), "records": records}


def run(protocol_path, output):
    import torch

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads(protocol_path.read_text())
    if protocol["evidence_status"] != "development_only_on_exposed_dates":
        raise ValueError("This fixed screen is development only")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Registered regime input changed: {path}")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"], "python": platform.python_version(),
                "dependencies": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "torch", "pyarrow")}}
    frozen = output / "frozen_screen.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve the existing history-breadth experiment after source changes")
    write_json(frozen, identity)
    for count, day in enumerate(protocol["dates"], 1):
        record = run_fold(day, protocol, output, identity)
        print(f"regime_dates_frozen={count}/{len(protocol['dates'])} date={day} seconds={record['seconds']:.1f}", flush=True)
        gc.collect()
    write_json(output / "summary.json", reveal(output, protocol, identity))
    print(f"regime_development_complete {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_regime_screen_20260907.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/boundary_regime_screen_20260907")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.protocol, args.output)
    else:
        print("Pass --run for the fixed exposed-date history-diversity experiment.")
