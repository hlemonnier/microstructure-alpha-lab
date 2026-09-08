"""Fixed three-date feature-embedding and shared-member accuracy screen."""

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
from lob_forge.boundary_confirm_model import SYMBOLS
from lob_forge.boundary_embedding import ARCHITECTURES, EmbeddingForecaster, fit_embedding_neural
from lob_forge.boundary_event_inputs import utc_ms
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_pooled import PooledForecaster
from lob_forge.boundary_regime_coverage import calendar_stride_mask, history_cohorts
from run_boundary_confirmation import noon, save_predictions, write_json
from run_boundary_event_clock_screen import check_completed

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ("original_reference", "observations_tree", "observations_neural", "observations_blend", "combined_hgb",
             "combined_neural", "combined_blend", "spot100_tree", "top1_100_tree_old_neural", "deep500_hgb", "deep500_hgb_old_neural")
MODEL_SPECS = {f"{representation}_{architecture}": (representation, architecture)
               for representation in ("observations", "combined") for architecture in ARCHITECTURES}
BLENDS = {f"{name}_{suffix}": (name, "deep500_hgb" if suffix == "deep500_hgb" else
           "observations_tree" if representation == "observations" else "combined_hgb")
          for name, (representation, _) in MODEL_SPECS.items() for suffix in ("matched_hgb", "deep500_hgb")}


def run_fold(day, protocol, output, identity):
    folder = output / "dates" / day
    if (folder / "completed.json").exists():
        return check_completed(folder, identity)
    started = time.monotonic()
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / f"attempt_{time.time_ns()}.json", {"date": day, "planned_model_fits": list(MODEL_SPECS)})
    sources = {key: ROOT / protocol[f"{key}_run"] / "dates" / day for key in ("context", "newton", "basis", "flow")}
    for source in sources.values():
        check_completed(source, json.loads((source.parent.parent / "frozen_screen.json").read_text()))
    training_days = history_cohorts(day)["recent4_stride4"]["train_dates"]
    validation_day = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    manifest = json.loads((ROOT / protocol["manifest"]).read_text())
    tx, ty, vx, vy, test_x, test_y, test_times = {}, {}, {}, {}, {}, {}, {}
    for current in [*training_days, validation_day, day]:
        partition = folder / "inputs" / f"{current}.json"
        sessions = [r for r in manifest["sessions"] if r["session_date"] == current]
        if len(sessions) != 2 or {r["symbol"] for r in sessions} != set(SYMBOLS):
            raise ValueError("Every fixed embedding date requires both assets")
        write_json(partition, {"sessions": sessions})
        x, y, times = load_combined_inputs(ROOT, partition, ROOT / protocol["depth_manifest"], ROOT / protocol["spot_manifest"])
        for symbol in SYMBOLS:
            key = symbol, current
            if current in training_days:
                mask = calendar_stride_mask(times[key], utc_ms(current), stride_seconds=4)
                tx[key], ty[key] = x[key].loc[mask].copy(), y[key][mask]
            else:
                mask = noon(times[key], current)
                if mask.sum() != 7070:
                    raise ValueError("The unchanged full noon assessment/validation universe is required")
                if current == validation_day:
                    vx[symbol], vy[symbol] = x[key].loc[mask].copy(), y[key][mask]
                else:
                    test_x[symbol], test_y[symbol], test_times[symbol] = x[key].loc[mask].copy(), y[key][mask], times[key][mask]
        del x, y, times
        gc.collect()
    tx = {s: pd.concat([tx[s, d] for d in training_days], ignore_index=True) for s in SYMBOLS}
    ty = {s: np.concatenate([ty[s, d] for d in training_days]) for s in SYMBOLS}
    forecasts = {s: {} for s in SYMBOLS}
    for symbol in SYMBOLS:
        for name in BASELINES:
            source = sources["context"] if name == "observations_tree" else sources["flow"]
            with np.load(source / "predictions" / f"{symbol}_{name}_registered.npz") as archive:
                forecasts[symbol][name] = {k: archive[k].copy() for k in archive.files}
            np.testing.assert_array_equal(test_y[symbol], forecasts[symbol][name]["labels"])
            np.testing.assert_array_equal(test_times[symbol], forecasts[symbol][name]["decision_times"])
    # Reproduce both matched tree and neural controls before training any challenger.
    for representation in ("observations", "combined"):
        tree_name = "observations_tree" if representation == "observations" else "combined_hgb"
        tree_source = sources["context"] if representation == "observations" else sources["newton"]
        neural_source = sources["context"] if representation == "observations" else sources["basis"]
        tree = joblib.load(tree_source / f"{tree_name}.joblib")
        neural = PooledForecaster.load(neural_source / f"{representation}_neural")
        columns = original_combined_columns(tx[SYMBOLS[0]]) if representation == "observations" else list(tx[SYMBOLS[0]].columns)
        if columns != tree.columns or columns != neural.columns or len(columns) != protocol["representations"][representation]:
            raise ValueError("Matched feature schema changed")
        for asset, symbol in enumerate(SYMBOLS):
            selected = test_x[symbol][columns]
            np.testing.assert_array_equal(tree.predict_proba(selected, symbol), forecasts[symbol][tree_name]["probabilities"])
            np.testing.assert_array_equal(neural.predict_proba(selected, asset), forecasts[symbol][f"{representation}_neural"]["probabilities"])
            expected = np.array([(ty[symbol] == k).mean() for k in (-1, 0, 1)])
            np.testing.assert_array_equal(expected, tree.priors[asset])
            np.testing.assert_array_equal(expected, neural.priors[asset])
        del tree, neural
    metadata = {}
    for name, (representation, architecture) in MODEL_SPECS.items():
        columns = original_combined_columns(tx[SYMBOLS[0]]) if representation == "observations" else list(tx[SYMBOLS[0]].columns)
        begin = time.monotonic()
        write_json(output / "progress.json", {"date": day, "stage": "training", "model": name, "assessment_scores_sealed": True})
        model, history = fit_embedding_neural({s: tx[s][columns] for s in SYMBOLS}, ty,
            {s: vx[s][columns] for s in SYMBOLS}, vy, architecture=architecture)
        model.save(folder / name)
        write_json(folder / name / "training_history.json", history)
        restored = EmbeddingForecaster.load(folder / name)
        tree_name = "observations_tree" if representation == "observations" else "combined_hgb"
        for asset, symbol in enumerate(SYMBOLS):
            p = restored.predict_proba(test_x[symbol][columns], asset)
            np.testing.assert_array_equal(p, model.predict_proba(test_x[symbol][columns], asset))
            np.testing.assert_array_equal(restored.priors[asset], forecasts[symbol][tree_name]["train_priors"])
            if architecture == "scalar1":
                np.testing.assert_array_equal(p, forecasts[symbol][f"{representation}_neural"]["probabilities"])
            forecasts[symbol][name] = {**forecasts[symbol][tree_name], "probabilities": p}
        metadata[name] = {"representation": representation, "architecture": architecture, "training_dates": training_days,
            "training_rows": {s: len(ty[s]) for s in SYMBOLS}, "columns": columns, "seconds": time.monotonic() - begin,
            "best_epoch": history["best_epoch"], "parameter_count": history["parameter_count"],
            "checkpoint_parity": True, "scalar_reproduction": architecture == "scalar1"}
        print(f"embedding_fit={day}/{name} seconds={metadata[name]['seconds']:.1f}", flush=True)
        del model, restored
        gc.collect()
    for symbol in SYMBOLS:
        for name, (left, right) in BLENDS.items():
            a, b = forecasts[symbol][left], forecasts[symbol][right]
            np.testing.assert_array_equal(a["train_priors"], b["train_priors"])
            forecasts[symbol][name] = {**a, "probabilities": (a["probabilities"] + b["probabilities"]) / 2}
        with np.load(ROOT / protocol["prior_run"] / "priors" / day / f"{symbol}_original_reference.npz") as archive:
            np.testing.assert_array_equal(archive["decision_times"], test_times[symbol])
            initial = archive["labels_cumulative"][0].copy()
        for name, values in forecasts[symbol].items():
            for policy in protocol["decision_policies"]:
                pi = values["decision_priors"] if policy == "registered" else forecast_priors(values["probabilities"], test_times[symbol], initial, half_life_seconds=3600)
                save_predictions(folder / "predictions" / f"{symbol}_{name}_{policy}.npz", values["probabilities"], test_y[symbol],
                    test_times[symbol], values["train_priors"], pi)
    write_json(folder / "fit_metadata.json", metadata)
    record = {"identity": identity, "date": day, "model_fits": len(MODEL_SPECS), "seconds": time.monotonic() - started,
        "original_checkpoint_parity": True, "scalar_reproduction_exact": True,
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
                    with np.load(folder / "predictions" / f"{symbol}_{name}_{policy}.npz") as archive:
                        metrics = classification_metrics(SimpleNamespace(priors=archive["decision_priors"]), archive["probabilities"], archive["labels"])
                    records.append({"date": day, "symbol": symbol, "model": name, "policy": policy, "metrics": metrics})
    if len(records) != protocol["post_fit_panels"]:
        raise ValueError("Incomplete frozen embedding development family")
    leaderboard = []
    for name, policy in sorted({(r["model"], r["policy"]) for r in records}):
        rows = [r for r in records if (r["model"], r["policy"]) == (name, policy)]
        leaderboard.append({"model": name, "policy": policy, "asset_dates": len(rows),
            "means": {k: float(np.mean([r["metrics"][k] for r in rows])) for k in ("balanced_accuracy", "natural_accuracy", "log_loss", "macro_f1")},
            "balanced_accuracy_by_asset": {s: float(np.mean([r["metrics"]["balanced_accuracy"] for r in rows if r["symbol"] == s])) for s in SYMBOLS}})
    return {"identity": identity, "evidence_status": protocol["evidence_status"], "model_fits": len(MODEL_SPECS) * len(protocol["dates"]),
        "post_fit_panels": len(records), "substantial_gain_confirmed": False,
        "leaderboard": sorted(leaderboard, key=lambda r: r["means"]["balanced_accuracy"], reverse=True), "records": records}


def run(protocol_path, output):
    import torch

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads(protocol_path.read_text())
    if protocol["evidence_status"] != "development_only_on_exposed_dates" or protocol["decision_policies"] != ["registered", "forecast_3600"]:
        raise ValueError("This fixed screen is development only with both registered policies")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Registered embedding input changed: {path}")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"], "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "torch", "pyarrow", "scipy")}}
    frozen = output / "frozen_screen.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve the existing embedding experiment after source changes")
    write_json(frozen, identity)
    for count, day in enumerate(protocol["dates"], 1):
        record = run_fold(day, protocol, output, identity)
        print(f"embedding_dates_frozen={count}/{len(protocol['dates'])} date={day} seconds={record['seconds']:.1f}", flush=True)
        gc.collect()
    write_json(output / "summary.json", reveal(output, protocol, identity))
    print(f"embedding_development_complete {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_embedding_screen_20260908.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/boundary_embedding_screen_20260908")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.protocol.resolve(), args.output.resolve())
    else:
        print("Pass --run for the fixed exposed-date feature-embedding experiment.")
