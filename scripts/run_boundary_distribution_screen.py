"""Compare exact magnitude distribution objectives with matched forecast controls."""

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
from lob_forge.boundary_confirm_model import SYMBOLS
from lob_forge.boundary_distribution_model import DistributionForecaster, fit_distribution_neural
from lob_forge.boundary_distribution_targets import COARSE_CLASSES
from lob_forge.boundary_event_clock_inputs import load_clock_inputs
from lob_forge.boundary_event_inputs import utc_ms
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_pooled import PooledForecaster
from lob_forge.boundary_regime_coverage import calendar_stride_mask, history_cohorts
from run_boundary_confirmation import noon, save_predictions, write_json
from run_boundary_event_clock_screen import check_completed

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ("original_reference", "original_blend", "clock_neural_clock_tree_7",
             "recent4_stride4_tree", "recent4_stride4_neural", "recent4_stride4_blend")
OBJECTIVES = {"coarse_nine_head": (0, 0), "fine_quarter": (.25, 0), "fine_full": (1, 0), "fine_ranked": (1, 1)}


def run_fold(day, protocol, output, identity):
    folder = output / "dates" / day
    if (folder / "completed.json").exists():
        return check_completed(folder, identity)
    started = time.monotonic()
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / f"attempt_{time.time_ns()}.json", {"day": day, "model_fits": 4})
    source = ROOT / protocol["regime_run"] / "dates" / day
    check_completed(source, json.loads((source.parent.parent / "frozen_screen.json").read_text()))
    train_days = history_cohorts(day)["recent4_stride4"]["train_dates"]
    validation_day = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    manifest = json.loads((ROOT / protocol["manifest"]).read_text())
    targets = json.loads((ROOT / protocol["target_manifest"]).read_text())
    target_index = {(s["symbol"], s["session_date"]): s for s in targets["sessions"]}
    train_x, train_bins, validation_x, validation_y, test_x, test_y, test_times = {}, {}, {}, {}, {}, {}, {}
    for current in [*train_days, validation_day, day]:
        partition = folder / "inputs" / f"{current}.json"
        sessions = [r for r in manifest["sessions"] if r["session_date"] == current]
        if len(sessions) != 2 or {r["symbol"] for r in sessions} != set(SYMBOLS):
            raise ValueError("Every registered distribution day requires both assets")
        write_json(partition, {"sessions": sessions})
        x, y, times = load_clock_inputs(ROOT, partition)
        for session in sessions:
            symbol = session["symbol"]
            record = target_index[symbol, current]
            if record["source_features_sha256"] != session["sha256"] or sha256_file(ROOT / record["targets_path"]) != record["sha256"]:
                raise ValueError("Separate target and observation source identities must match")
            extra = pd.read_parquet(ROOT / record["targets_path"]).set_index("decision_time").loc[times[symbol, current]]
            fine = extra.move_bin.to_numpy()
            np.testing.assert_array_equal(COARSE_CLASSES[fine], y[symbol, current])
            np.testing.assert_array_equal(extra.label.to_numpy(), y[symbol, current])
            if current in train_days:
                mask = calendar_stride_mask(times[symbol, current], utc_ms(current), stride_seconds=4)
                train_x[symbol, current], train_bins[symbol, current] = x[symbol, current].loc[mask].copy(), fine[mask]
            else:
                mask = noon(times[symbol, current], current)
                if current == validation_day:
                    validation_x[symbol], validation_y[symbol] = x[symbol, current].loc[mask].copy(), y[symbol, current][mask]
                else:
                    test_x[symbol] = x[symbol, current].loc[mask].copy()
                    test_y[symbol], test_times[symbol] = y[symbol, current][mask], times[symbol, current][mask]
        del x, y, times, extra
        gc.collect()
    train_x = {s: pd.concat([train_x[s, d] for d in train_days], ignore_index=True) for s in SYMBOLS}
    train_bins = {s: np.concatenate([train_bins[s, d] for d in train_days]) for s in SYMBOLS}
    forecasts = {}
    old_neural = PooledForecaster.load(source / "recent4_stride4_neural")
    for asset, symbol in enumerate(SYMBOLS):
        forecasts[symbol] = {}
        for name in BASELINES:
            with np.load(source / "predictions" / f"{symbol}_{name}_registered.npz") as archive:
                forecasts[symbol][name] = {k: archive[k].copy() for k in archive.files}
        np.testing.assert_array_equal(test_y[symbol], forecasts[symbol]["original_blend"]["labels"])
        np.testing.assert_array_equal(test_times[symbol], forecasts[symbol]["original_blend"]["decision_times"])
        np.testing.assert_array_equal(old_neural.predict_proba(test_x[symbol], asset), forecasts[symbol]["recent4_stride4_neural"]["probabilities"])
    metadata = {}
    for name, (fine_weight, ranked_weight) in OBJECTIVES.items():
        begin = time.monotonic()
        write_json(output / "progress.json", {"day": day, "stage": "training", "model": name, "scores_sealed": True})
        model, history = fit_distribution_neural(train_x, train_bins, validation_x, validation_y, fine_weight=fine_weight, ranked_weight=ranked_weight)
        np.testing.assert_array_equal(model.active, old_neural.active)
        np.testing.assert_array_equal(model.normalizer.quantiles_, old_neural.normalizer.quantiles_)
        model.save(folder / name)
        write_json(folder / name / "training_history.json", history)
        restored = DistributionForecaster.load(folder / name)
        for asset, symbol in enumerate(SYMBOLS):
            p = restored.predict_proba(test_x[symbol], asset)
            np.testing.assert_array_equal(p, model.predict_proba(test_x[symbol], asset))
            np.testing.assert_array_equal(restored.priors[asset], forecasts[symbol]["recent4_stride4_tree"]["train_priors"])
            forecasts[symbol][name] = {**forecasts[symbol]["recent4_stride4_tree"], "probabilities": p}
            forecasts[symbol][name + "_tree_blend"] = {**forecasts[symbol][name], "probabilities": (p + forecasts[symbol]["recent4_stride4_tree"]["probabilities"]) / 2}
        metadata[name] = {"fine_weight": fine_weight, "ranked_weight": ranked_weight, "train_dates": train_days,
                          "train_rows": {s: len(train_bins[s]) for s in SYMBOLS}, "columns": model.columns,
                          "seconds": time.monotonic() - begin, "checkpoint_parity": True, "original_normalizer_parity": True}
        print(f"distribution_fit={day}/{name} seconds={metadata[name]['seconds']:.1f}", flush=True)
        del model, restored
        gc.collect()
    for symbol in SYMBOLS:
        with np.load(ROOT / protocol["prior_run"] / "priors" / day / f"{symbol}_original_reference.npz") as archive:
            np.testing.assert_array_equal(archive["decision_times"], test_times[symbol])
            initial = archive["labels_cumulative"][0].copy()
        for name, values in forecasts[symbol].items():
            for policy in ("registered", "forecast_3600"):
                pi = values["decision_priors"] if policy == "registered" else forecast_priors(values["probabilities"], test_times[symbol], initial, half_life_seconds=3600)
                save_predictions(folder / "predictions" / f"{symbol}_{name}_{policy}.npz", values["probabilities"], test_y[symbol], test_times[symbol], values["train_priors"], pi)
    write_json(folder / "fit_metadata.json", metadata)
    record = {"identity": identity, "date": day, "model_fits": 4, "seconds": time.monotonic() - started,
              "original_checkpoint_parity": True, "artifact_hashes": {str(p.relative_to(folder)): sha256_file(p) for p in folder.rglob("*") if p.is_file()}}
    write_json(folder / "completed.json", record)
    return record


def reveal(output, protocol, identity):
    records = []
    for day in protocol["dates"]:
        folder = output / "dates" / day
        check_completed(folder, identity)
        for symbol in SYMBOLS:
            for name in [*BASELINES, *OBJECTIVES, *(n + "_tree_blend" for n in OBJECTIVES)]:
                for policy in ("registered", "forecast_3600"):
                    with np.load(folder / "predictions" / f"{symbol}_{name}_{policy}.npz") as archive:
                        metrics = classification_metrics(SimpleNamespace(priors=archive["decision_priors"]), archive["probabilities"], archive["labels"])
                    records.append({"date": day, "symbol": symbol, "model": name, "policy": policy, "metrics": metrics})
    if len(records) != protocol["post_fit_panels"]:
        raise ValueError("Incomplete move-distribution development family")
    index = {(r["date"], r["symbol"], r["model"], r["policy"]): r for r in records}
    leaderboard = []
    for name, policy in sorted({(r["model"], r["policy"]) for r in records}):
        rows = [r for r in records if (r["model"], r["policy"]) == (name, policy)]
        leaderboard.append({"model": name, "policy": policy, "asset_dates": len(rows),
            "means": {k: float(np.mean([r["metrics"][k] for r in rows])) for k in ("balanced_accuracy", "natural_accuracy", "log_loss", "macro_f1")},
            "delta_to_same_policy_matched_clock_blend": float(np.mean([r["metrics"]["balanced_accuracy"] - index[r["date"], r["symbol"], "recent4_stride4_blend", policy]["metrics"]["balanced_accuracy"] for r in rows])),
            "balanced_accuracy_by_asset": {s: float(np.mean([r["metrics"]["balanced_accuracy"] for r in rows if r["symbol"] == s])) for s in SYMBOLS}})
    return {"identity": identity, "evidence_status": protocol["evidence_status"], "model_fits": 4 * len(protocol["dates"]), "post_fit_panels": len(records),
            "substantial_gain_confirmed": False, "leaderboard": sorted(leaderboard, key=lambda r: r["means"]["balanced_accuracy"], reverse=True), "records": records}


def run(protocol_path, output):
    import torch

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads(protocol_path.read_text())
    if protocol["evidence_status"] != "development_only_on_exposed_dates":
        raise ValueError("This fixed screen is development only")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Registered distribution input changed: {path}")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"], "python": platform.python_version(),
                "dependencies": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "torch", "pyarrow")}}
    frozen = output / "frozen_screen.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve the existing distribution experiment after source changes")
    write_json(frozen, identity)
    for count, day in enumerate(protocol["dates"], 1):
        record = run_fold(day, protocol, output, identity)
        print(f"distribution_dates_frozen={count}/{len(protocol['dates'])} date={day} seconds={record['seconds']:.1f}", flush=True)
        gc.collect()
    write_json(output / "summary.json", reveal(output, protocol, identity))
    print(f"distribution_development_complete {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_distribution_screen_20260907.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/boundary_distribution_screen_20260907")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.protocol.resolve(), args.output.resolve())
    else:
        print("Pass --run for the fixed exposed-date move-distribution experiment.")
