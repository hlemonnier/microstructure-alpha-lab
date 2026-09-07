"""Test same-day supervised adaptation with all updates fixed before noon."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_adaptive_priors import forecast_priors
from lob_forge.boundary_confirmation_integrity import verify_frozen_fold
from lob_forge.boundary_event_inputs import load_event_inputs, utc_ms
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_morning_adaptation import SYMBOLS, fit_morning_neural, morning_masks
from lob_forge.boundary_pooled import PooledForecaster
from lob_forge.boundary_weighted_boost import fit_weighted_boost
from run_boundary_confirmation import noon, write_json
from run_boundary_event_clock_screen import check_completed

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ("six_hours", "full_morning")
NEURAL_FITS = {f"{window}_neural_{rate_name}": (window, rate) for window in WINDOWS for rate_name, rate in (("1e4", 0.0001), ("3e4", 0.0003))}
TREE_FITS = {window + "_tree": window for window in WINDOWS}
FITS = [*TREE_FITS, *NEURAL_FITS]
BASELINES = ["original_reference", "matched_data_control", "original_blend", "pooled_blend", "old_neural", "old_tree"]
BLENDS = {
    **{name + "_old_tree_blend": (name, "old_tree") for name in NEURAL_FITS},
    **{name + "_morning_tree_blend": (name, window + "_tree") for name, (window, _) in NEURAL_FITS.items()},
}


def run_fold(day, protocol, output, identity):
    folder = output / "dates" / day
    if (folder / "completed.json").exists():
        return check_completed(folder, identity)
    started = time.monotonic()
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / f"attempt_{time.time_ns()}.json", {"date": day, "fits": FITS})
    primary, secondary = (ROOT / protocol[name] / "dates" / day for name in ("primary_run", "secondary_run"))
    for source in (primary, secondary):
        verify_frozen_fold(source, json.loads((source.parent.parent / "frozen_study.json").read_text()))
    source_manifest = json.loads((ROOT / protocol["manifest"]).read_text())
    sessions = [r for r in source_manifest["sessions"] if r["session_date"] == day]
    if {r["symbol"] for r in sessions} != set(SYMBOLS) or len(sessions) != 2:
        raise ValueError("The two exact-label sessions are required")
    write_json(folder / "partition_manifest.json", {"sessions": sessions})
    x, y, clocks = load_event_inputs(ROOT, folder / "partition_manifest.json")
    train_x, train_y, val_x, val_y, test_x = {}, {}, {}, {}, {}
    full_priors, partitions, forecasts = {}, {}, {}
    original_neural = PooledForecaster.load(primary / "neural")
    for asset, symbol in enumerate(SYMBOLS):
        raw = next(s for s in sessions if s["symbol"] == symbol)
        resolution = pd.read_parquet(ROOT / raw["features_path"], columns=["decision_time", "label", "future_event_time"]).set_index("decision_time").loc[clocks[symbol, day]]
        np.testing.assert_array_equal(resolution.label.to_numpy(), y[symbol, day])
        partitions[symbol] = {}
        for window in WINDOWS:
            training, validation = morning_masks(clocks[symbol, day], resolution.future_event_time.to_numpy(), utc_ms(day), window=window)
            if not training.any() or not validation.any():
                raise ValueError("Morning training and validation must both be observed")
            train_x[window, symbol], train_y[window, symbol] = x[symbol, day].loc[training], y[symbol, day][training]
            partitions[symbol][window] = {
                "training_rows": int(training.sum()), "validation_rows": int(validation.sum()),
                "last_training_release_ms": int(resolution.future_event_time.to_numpy()[training].max()),
                "last_validation_release_ms": int(resolution.future_event_time.to_numpy()[validation].max()),
            }
            if window == "full_morning":
                full_priors[symbol] = np.array([(train_y[window, symbol] == k).mean() for k in (-1, 0, 1)])
                val_x[symbol], val_y[symbol] = x[symbol, day].loc[validation], y[symbol, day][validation]
        test = noon(clocks[symbol, day], day)
        test_x[symbol] = x[symbol, day].loc[test]
        forecasts[symbol] = {}
        for name, location, kind in (
            ("original_reference", primary, "original_reference"), ("matched_data_control", primary, "matched_data_control"),
            ("original_blend", primary, "candidate"), ("pooled_blend", secondary, "candidate"),
            ("old_neural", primary, "neural_component"), ("old_tree", primary, "tree_component"),
        ):
            with np.load(location / "predictions" / f"{symbol}_{kind}.npz") as archive:
                forecasts[symbol][name] = {k: archive[k].copy() for k in archive.files}
        canonical = forecasts[symbol]["original_blend"]
        np.testing.assert_array_equal(canonical["labels"], y[symbol, day][test])
        np.testing.assert_array_equal(canonical["decision_times"], clocks[symbol, day][test])
        np.testing.assert_array_equal(original_neural.predict_proba(test_x[symbol], asset), forecasts[symbol]["old_neural"]["probabilities"])
    del x, y, original_neural
    gc.collect()
    write_json(folder / "available_partitions.json", {"asset_partitions": partitions, "full_morning_decision_priors": {s: p.tolist() for s, p in full_priors.items()}, "latest_validation_release_deadline_ms": utc_ms(day, "11:57:10"), "first_assessment_decision_ms": utc_ms(day, "12:02:00")})
    metadata = {}
    for name in FITS:
        fit_start = time.monotonic()
        window = TREE_FITS[name] if name in TREE_FITS else NEURAL_FITS[name][0]
        training_features, training_labels = {s: train_x[window, s] for s in SYMBOLS}, {s: train_y[window, s] for s in SYMBOLS}
        write_json(output / "progress.json", {"date": day, "stage": "training", "model": name, "development_scores_sealed": True})
        if name in TREE_FITS:
            with threadpool_limits(limits=2):
                model = fit_weighted_boost(training_features, training_labels, leaves=7, class_balanced=True)
            joblib.dump(model, folder / f"{name}.joblib")
            restored = joblib.load(folder / f"{name}.joblib")
            training_record = {"recent_training_priors": restored.priors.tolist(), "posterior_recovery_priors": restored.priors.tolist()}
        else:
            model, training_record = fit_morning_neural(primary / "neural", training_features, training_labels, val_x, val_y, full_priors, learning_rate=NEURAL_FITS[name][1])
            model.save(folder / name)
            restored = PooledForecaster.load(folder / name)
            write_json(folder / name / "training_history.json", training_record)
        ready_seconds = time.monotonic() - fit_start
        for asset, symbol in enumerate(SYMBOLS):
            identifier = symbol if name in TREE_FITS else asset
            p = restored.predict_proba(test_x[symbol], identifier)
            np.testing.assert_array_equal(p, model.predict_proba(test_x[symbol], identifier))
            forecasts[symbol][name] = {**forecasts[symbol]["original_blend"], "probabilities": p}
        metadata[name] = {
            "window": window, "training_and_checkpoint_seconds": ready_seconds, "within_290_second_computation_budget": ready_seconds <= 290,
            "checkpoint_parity": True, "training_record": training_record, "features": restored.columns,
        }
        print(f"morning_fit={day}/{name} ready_seconds={ready_seconds:.1f}", flush=True)
        del model, restored
        gc.collect()
    for symbol in SYMBOLS:
        for name, (left, right) in BLENDS.items():
            forecasts[symbol][name] = {**forecasts[symbol]["original_blend"], "probabilities": (forecasts[symbol][left]["probabilities"] + forecasts[symbol][right]["probabilities"]) / 2}
        with np.load(ROOT / protocol["prior_run"] / "priors" / day / f"{symbol}_original_reference.npz") as archive:
            np.testing.assert_array_equal(archive["decision_times"], forecasts[symbol]["original_blend"]["decision_times"])
            initial = archive["labels_cumulative"][0].copy()
        for name, values in forecasts[symbol].items():
            for policy in protocol["policies"]:
                priors = values["decision_priors"] if policy == "registered" else full_priors[symbol] if policy == "morning_full" else forecast_priors(values["probabilities"], values["decision_times"], initial, half_life_seconds=3600)
                destination = folder / "predictions" / f"{symbol}_{name}_{policy}.npz"
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination, probabilities=values["probabilities"], labels=values["labels"], decision_times=values["decision_times"], decision_priors=priors)
    write_json(folder / "fit_metadata.json", metadata)
    record = {"identity": identity, "date": day, "model_fits": len(FITS), "seconds": time.monotonic() - started,
              "original_checkpoint_and_row_parity": True, "artifact_hashes": {str(p.relative_to(folder)): sha256_file(p) for p in folder.rglob("*") if p.is_file()}}
    write_json(folder / "completed.json", record)
    return record


def reveal(output, protocol, identity):
    records = []
    names = [*BASELINES, *FITS, *BLENDS]
    for day in protocol["dates"]:
        folder = output / "dates" / day
        check_completed(folder, identity)
        for symbol in SYMBOLS:
            for name in names:
                for policy in protocol["policies"]:
                    with np.load(folder / "predictions" / f"{symbol}_{name}_{policy}.npz") as archive:
                        metrics = classification_metrics(SimpleNamespace(priors=archive["decision_priors"]), archive["probabilities"], archive["labels"])
                    records.append({"date": day, "symbol": symbol, "model": name, "policy": policy, "metrics": metrics})
    index = {(r["date"], r["symbol"], r["model"], r["policy"]): r for r in records}
    leaderboard = []
    for name in names:
        for policy in protocol["policies"]:
            rows = [r for r in records if (r["model"], r["policy"]) == (name, policy)]
            leaderboard.append({
                "model": name, "policy": policy, "asset_dates": len(rows),
                "means": {k: float(np.mean([r["metrics"][k] for r in rows])) for k in ("balanced_accuracy", "natural_accuracy", "log_loss", "macro_f1")},
                "balanced_accuracy_by_asset": {s: float(np.mean([r["metrics"]["balanced_accuracy"] for r in rows if r["symbol"] == s])) for s in SYMBOLS},
                "delta_to_original_registered_reference": float(np.mean([r["metrics"]["balanced_accuracy"] - index[r["date"], r["symbol"], "original_reference", "registered"]["metrics"]["balanced_accuracy"] for r in rows])),
                "delta_to_same_policy_original_blend": float(np.mean([r["metrics"]["balanced_accuracy"] - index[r["date"], r["symbol"], "original_blend", policy]["metrics"]["balanced_accuracy"] for r in rows])),
            })
    if len(records) != protocol["post_fit_asset_date_panels"]:
        raise ValueError("Incomplete fixed morning family")
    return {"identity": identity, "evidence_status": protocol["evidence_status"], "model_fits": len(FITS) * len(protocol["dates"]), "post_fit_panels": len(records), "substantial_gain_confirmed": False,
            "leaderboard": sorted(leaderboard, key=lambda r: r["means"]["balanced_accuracy"], reverse=True), "records": records}


def run(protocol_path, output):
    import torch

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads(protocol_path.read_text())
    if protocol["evidence_status"] != "development_only_on_exposed_dates" or protocol["fits"] != FITS or protocol["policies"] != ["registered", "morning_full", "forecast_3600"]:
        raise ValueError("Only the fixed development family may be run")
    for name, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / name) != checksum:
            raise ValueError(f"Registered development input changed: {name}")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"], "python": platform.python_version(),
                "dependencies": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "torch", "pyarrow")}}
    frozen = output / "frozen_screen.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve this development experiment after an identity change")
    write_json(frozen, identity)
    for count, day in enumerate(protocol["dates"], 1):
        record = run_fold(day, protocol, output, identity)
        print(f"morning_dates_frozen={count}/{len(protocol['dates'])} date={day} seconds={record['seconds']:.1f}", flush=True)
        gc.collect()
    write_json(output / "summary.json", reveal(output, protocol, identity))
    print(f"morning_development_complete {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_morning_screen_20260907.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/boundary_morning_screen_20260907")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.protocol, args.output)
    else:
        print("Pass --run for the fixed same-day morning development experiment.")
