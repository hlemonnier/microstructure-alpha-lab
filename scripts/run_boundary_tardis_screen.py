"""Test target-venue depth against matched morning and historical predictors."""

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
from lob_forge.boundary_confirm_model import SYMBOLS, fit_confirm_neural
from lob_forge.boundary_confirmation_integrity import verify_frozen_fold
from lob_forge.boundary_event_inputs import utc_ms
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_long_context_inputs import context_columns, load_context_inputs
from lob_forge.boundary_morning_adaptation import morning_masks
from lob_forge.boundary_pooled import PooledForecaster
from lob_forge.boundary_regime_coverage import calendar_stride_mask, history_cohorts
from lob_forge.boundary_tardis_inputs import TARDIS_VARIANTS, load_tardis_inputs, select_tardis_variant
from lob_forge.boundary_weighted_boost import fit_weighted_boost
from run_boundary_confirmation import noon, write_json
from run_boundary_event_clock_screen import check_completed

ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIONS = ("observations", "combined", *TARDIS_VARIANTS)
BASELINES = ("original_reference", "original_blend", "pooled_blend")
FITS = {**{f"past_context_{kind}": ("past", kind) for kind in ("hgb", "neural")},
        **{f"{rep}_{kind}": (rep, kind) for rep in REPRESENTATIONS for kind in ("hgb", "neural")}}
BLENDS = {"past_context_blend": ("past_context_hgb", "past_context_neural")}
for _rep in REPRESENTATIONS:
    BLENDS[_rep + "_blend"] = (_rep + "_hgb", _rep + "_neural")
    BLENDS[_rep + "_hgb_past_neural"] = (_rep + "_hgb", "past_context_neural")
    BLENDS[_rep + "_neural_past_hgb"] = (_rep + "_neural", "past_context_hgb")


def _partition(folder, manifest, day):
    sessions = [r for r in manifest["sessions"] if r["session_date"] == day]
    if len(sessions) != 2 or {r["symbol"] for r in sessions} != set(SYMBOLS):
        raise ValueError("Both original assets are required in every partition")
    path = folder / "inputs" / f"{day}.json"
    write_json(path, {"sessions": sessions})
    return path, sessions


def run_fold(day, protocol, output, identity):
    folder = output / "dates" / day
    if (folder / "completed.json").exists():
        return check_completed(folder, identity)
    started = time.monotonic()
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / f"attempt_{time.time_ns()}.json", {"model_fits": len(FITS), "assessment_scores_sealed": True})
    manifest = json.loads((ROOT / protocol["manifest"]).read_text())
    train_dates = history_cohorts(day)["recent4_stride4"]["train_dates"]
    validation_day = protocol["historical_validation_day"]
    history_x, history_y, history_vx, history_vy = {}, {}, {}, {}
    for current in [*train_dates, validation_day]:
        partition, _ = _partition(folder, manifest, current)
        x, y, clocks = load_context_inputs(ROOT, partition)
        for symbol in SYMBOLS:
            features = x[symbol, current][context_columns(x[symbol, current], "observations")]
            if current in train_dates:
                selected = calendar_stride_mask(clocks[symbol, current], utc_ms(current), stride_seconds=4)
                history_x[symbol, current], history_y[symbol, current] = features.loc[selected].copy(), y[symbol, current][selected]
            else:
                selected = noon(clocks[symbol, current], current)
                history_vx[symbol], history_vy[symbol] = features.loc[selected].copy(), y[symbol, current][selected]
        del x, y, clocks
        gc.collect()
    history_x = {s: pd.concat([history_x[s, d] for d in train_dates], ignore_index=True) for s in SYMBOLS}
    history_y = {s: np.concatenate([history_y[s, d] for d in train_dates]) for s in SYMBOLS}
    partition, sessions = _partition(folder, manifest, day)
    x, y, clocks = load_tardis_inputs(ROOT, partition, ROOT / protocol["native_manifest"],
                                     ROOT / protocol["bybit_manifest"], ROOT / protocol["spot_manifest"])
    train_x, train_y, val_x, val_y, test_x, test_y, test_times, morning_priors = ({} for _ in range(8))
    partitions = {}
    for symbol in SYMBOLS:
        record = next(r for r in sessions if r["symbol"] == symbol)
        released = pd.read_parquet(ROOT / record["features_path"], columns=["decision_time", "label", "future_event_time"]).set_index("decision_time").loc[clocks[symbol, day]]
        np.testing.assert_array_equal(released.label.to_numpy(), y[symbol, day])
        training, validation = morning_masks(clocks[symbol, day], released.future_event_time.to_numpy(), utc_ms(day), window="full_morning")
        selected = noon(clocks[symbol, day], day)
        if int(selected.sum()) != 7070 or not training.any() or not validation.any():
            raise ValueError("Exact original noon universe and nonempty purged morning partitions required")
        train_x[symbol], train_y[symbol] = x[symbol, day].loc[training].copy(), y[symbol, day][training]
        val_x[symbol], val_y[symbol] = x[symbol, day].loc[validation].copy(), y[symbol, day][validation]
        test_x[symbol], test_y[symbol], test_times[symbol] = x[symbol, day].loc[selected].copy(), y[symbol, day][selected], clocks[symbol, day][selected]
        morning_priors[symbol] = np.array([(train_y[symbol] == k).mean() for k in (-1, 0, 1)])
        partitions[symbol] = {"training_rows": int(training.sum()), "validation_rows": int(validation.sum()),
            "assessment_rows": int(selected.sum()), "last_training_release_ms": int(released.future_event_time.to_numpy()[training].max()),
            "last_validation_release_ms": int(released.future_event_time.to_numpy()[validation].max())}
    del x, y, clocks
    gc.collect()
    write_json(folder / "available_partitions.json", {"partitions": partitions, "past_training_dates": train_dates,
        "past_validation_day": validation_day, "latest_morning_validation_release_ms": utc_ms(day, "11:57:10"),
        "first_noon_decision_ms": utc_ms(day, "12:02:00"), "morning_priors": {s: p.tolist() for s, p in morning_priors.items()}})
    primary, secondary = (ROOT / protocol[k] / "dates" / day for k in ("primary_run", "secondary_run"))
    for source in (primary, secondary):
        verify_frozen_fold(source, json.loads((source.parent.parent / "frozen_study.json").read_text()))
    forecasts = {s: {} for s in SYMBOLS}
    old_neural = PooledForecaster.load(primary / "neural")
    for asset, symbol in enumerate(SYMBOLS):
        for name, source, label in (("original_reference", primary, "original_reference"), ("original_blend", primary, "candidate"),
                                    ("pooled_blend", secondary, "candidate")):
            with np.load(source / "predictions" / f"{symbol}_{label}.npz", allow_pickle=False) as values:
                np.testing.assert_array_equal(values["labels"], test_y[symbol])
                np.testing.assert_array_equal(values["decision_times"], test_times[symbol])
                forecasts[symbol][name] = {"p": values["probabilities"].copy(), "components": [name]}
        with np.load(primary / "predictions" / f"{symbol}_neural_component.npz", allow_pickle=False) as values:
            np.testing.assert_array_equal(old_neural.predict_proba(test_x[symbol][old_neural.columns], asset), values["probabilities"])
    del old_neural
    metadata = {}
    for name, (representation, kind) in FITS.items():
        if representation == "past":
            tx, ty, vx, vy = history_x, history_y, history_vx, history_vy
            assessment = {s: select_tardis_variant(test_x[s], "observations") for s in SYMBOLS}
        else:
            tx = {s: select_tardis_variant(train_x[s], representation) for s in SYMBOLS}
            vx = {s: select_tardis_variant(val_x[s], representation) for s in SYMBOLS}
            assessment = {s: select_tardis_variant(test_x[s], representation) for s in SYMBOLS}
            ty, vy = train_y, val_y
        expected_width = protocol["feature_counts"]["observations" if representation == "past" else representation]
        if any(len(v.columns) != expected_width for v in [*tx.values(), *vx.values(), *assessment.values()]):
            raise ValueError("Registered native-depth feature schema changed")
        write_json(output / "progress.json", {"stage": "training", "model": name, "assessment_scores_sealed": True})
        begin = time.monotonic()
        if kind == "hgb":
            with threadpool_limits(limits=2):
                model = fit_weighted_boost(tx, ty, leaves=7, class_balanced=True)
            joblib.dump(model, folder / f"{name}.joblib")
            restored = joblib.load(folder / f"{name}.joblib")
        else:
            model, history = fit_confirm_neural(tx, ty, vx, vy)
            model.save(folder / name)
            write_json(folder / name / "training_history.json", history)
            restored = PooledForecaster.load(folder / name)
        seconds = time.monotonic() - begin
        if representation != "past" and seconds > 290:
            raise ValueError("Morning fit and checkpoint exceeded its declared computation window")
        priors = {}
        for asset, symbol in enumerate(SYMBOLS):
            identifier = symbol if kind == "hgb" else asset
            p = restored.predict_proba(assessment[symbol], identifier)
            np.testing.assert_array_equal(p, model.predict_proba(assessment[symbol], identifier))
            expected = np.array([(ty[symbol] == k).mean() for k in (-1, 0, 1)])
            np.testing.assert_array_equal(restored.priors[asset], expected)
            forecasts[symbol][name] = {"p": p, "components": [name]}
            priors[symbol] = expected.tolist()
        metadata[name] = {"representation": representation, "kind": kind, "columns": list(tx[SYMBOLS[0]].columns),
            "training_rows": {s: len(ty[s]) for s in SYMBOLS}, "training_and_checkpoint_seconds": seconds,
            "posterior_recovery_priors": priors, "checkpoint_parity": True}
        print(f"target_depth_fit={name} seconds={seconds:.1f}", flush=True)
        del model, restored
        gc.collect()
    for name, components in BLENDS.items():
        seconds = sum(metadata[c]["training_and_checkpoint_seconds"] for c in components if not c.startswith("past_"))
        if seconds > 290:
            raise ValueError("Combined morning component fits exceed their causal computation window")
    for symbol in SYMBOLS:
        for name, components in BLENDS.items():
            forecasts[symbol][name] = {"p": sum(forecasts[symbol][c]["p"] for c in components) / len(components), "components": list(components)}
        with np.load(ROOT / protocol["prior_run"] / "priors" / day / f"{symbol}_original_reference.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["decision_times"], test_times[symbol])
            initial = saved["labels_cumulative"][0].copy()
        for name, values in forecasts[symbol].items():
            for policy in protocol["policies"]:
                pi = morning_priors[symbol] if policy == "morning_full" else forecast_priors(values["p"], test_times[symbol], initial, half_life_seconds=3600)
                path = folder / "predictions" / f"{symbol}_{name}_{policy}.npz"
                path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path, probabilities=values["p"], labels=test_y[symbol], decision_times=test_times[symbol],
                    decision_priors=pi, components=np.asarray(values["components"]))
                with np.load(path, allow_pickle=False) as restored:
                    np.testing.assert_array_equal(restored["probabilities"], values["p"])
                    np.testing.assert_array_equal(restored["decision_priors"], pi)
    write_json(folder / "fit_metadata.json", metadata)
    record = {"identity": identity, "model_fits": len(FITS), "original_checkpoint_and_row_parity": True,
        "seconds": time.monotonic() - started, "artifact_hashes": {str(p.relative_to(folder)): sha256_file(p) for p in folder.rglob("*") if p.is_file()}}
    write_json(folder / "completed.json", record)
    return record


def reveal(output, protocol, identity):
    folder = output / "dates" / protocol["date"]
    check_completed(folder, identity)
    rows = []
    for symbol in SYMBOLS:
        for model in [*BASELINES, *FITS, *BLENDS]:
            for policy in protocol["policies"]:
                with np.load(folder / "predictions" / f"{symbol}_{model}_{policy}.npz", allow_pickle=False) as values:
                    metric = classification_metrics(SimpleNamespace(priors=values["decision_priors"]), values["probabilities"], values["labels"])
                rows.append({"date": protocol["date"], "symbol": symbol, "model": model, "policy": policy, "metrics": metric})
    if len(rows) != protocol["post_fit_panels"]:
        raise ValueError("Incomplete target-venue depth assessment family")
    board = []
    for model, policy in sorted({(r["model"], r["policy"]) for r in rows}):
        chosen = [r for r in rows if (r["model"], r["policy"]) == (model, policy)]
        board.append({"model": model, "policy": policy, "means": {k: float(np.mean([r["metrics"][k] for r in chosen]))
            for k in ("balanced_accuracy", "natural_accuracy", "log_loss", "macro_f1")},
            "balanced_accuracy_by_asset": {r["symbol"]: r["metrics"]["balanced_accuracy"] for r in chosen}})
    return {"identity": identity, "evidence_status": protocol["evidence_status"], "model_fits": len(FITS),
        "post_fit_panels": len(rows), "substantial_gain_confirmed": False,
        "leaderboard": sorted(board, key=lambda r: r["means"]["balanced_accuracy"], reverse=True), "records": rows}


def run(protocol_path, output):
    import torch

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads(protocol_path.read_text())
    if protocol["evidence_status"] != "development_only_on_exposed_date" or protocol["policies"] != ["morning_full", "forecast_3600"]:
        raise ValueError("Fixed development scope and common decision policies required")
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen target-depth input changed: {path}")
    identity = {"protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"], "python": platform.python_version(),
        "dependencies": {n: version(n) for n in ("numpy", "pandas", "scikit-learn", "torch", "pyarrow")}}
    frozen = output / "frozen_screen.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve the existing native-depth screen after source changes")
    write_json(frozen, identity)
    run_fold(protocol["date"], protocol, output, identity)
    write_json(output / "summary.json", reveal(output, protocol, identity))
    print(f"target_depth_development_complete {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.protocol.resolve(), args.output.resolve())
    else:
        print("Pass --run for the fixed target-depth development experiment.")
