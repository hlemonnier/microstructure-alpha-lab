"""Run registered supervised-model development trials on previously exposed dates."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_forecasts import classification_metrics, feature_frame, fit_classifier
from lob_forge.features import FEATURE_SEMANTICS_VERSION

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def time_ms(day, clock):
    return int(datetime.fromisoformat(f"{day}T{clock}").replace(tzinfo=timezone.utc).timestamp() * 1000)


def load_frames(protocol):
    manifest = ROOT / protocol["data"]["manifest"]
    if sha256_file(manifest) != protocol["data"]["manifest_sha256"]:
        raise ValueError("development dataset identity changed")
    data = json.loads(manifest.read_text())
    raw = {}
    for session in data["sessions"]:
        if session["session_date"] in protocol["data"]["unseen_reserved"]:
            raise ValueError("reserved data cannot enter development screen")
        path = Path(session["features_path"])
        if sha256_file(path) != session["feature"]["sha256"]:
            raise ValueError("feature artifact hash mismatch")
        frame = pd.read_csv(path)
        if not (frame["feature_semantics_version"] == FEATURE_SEMANTICS_VERSION).all():
            raise ValueError("wrong feature semantics")
        raw[session["symbol"], session["session_date"]] = frame
    features, targets, times = {}, {}, {}
    for (symbol, day), frame in raw.items():
        peer = raw["ETHUSDT" if symbol == "BTCUSDT" else "BTCUSDT", day]
        clock = frame["decision_time"].to_numpy()
        eligible = (
            (clock >= time_ms(day, protocol["data"]["decision_window_utc"][0]))
            & (clock < time_ms(day, protocol["data"]["decision_window_utc"][1]))
            & (frame["quote_age_ms"].to_numpy() <= 1000)
        )
        targets[symbol, day] = frame.loc[eligible, "label"].to_numpy(dtype=int)
        times[symbol, day] = frame.loc[eligible, "decision_time"].to_numpy(dtype=np.int64)
        for representation in protocol["first_tabular_grid"]["representations"]:
            transformed = feature_frame(frame, peer, representation)
            features[symbol, day, representation] = transformed.loc[eligible].reset_index(drop=True)
    return features, targets, times


def summarize(records):
    groups = {}
    for r in records:
        groups.setdefault((r["representation"], r["model"]), []).append(r)
    leaderboard = []
    for (representation, model), rows in groups.items():
        leaderboard.append(
            {
                "representation": representation,
                "model": model,
                "completed_asset_folds": len(rows),
                "mean_validation_balanced_accuracy": float(
                    np.mean([r["validation"]["balanced_accuracy"] for r in rows])
                ),
                "mean_development_balanced_accuracy": float(
                    np.mean([r["assessment"]["balanced_accuracy"] for r in rows])
                ),
                "mean_development_log_loss": float(np.mean([r["assessment"]["log_loss"] for r in rows])),
                "by_asset": {
                    symbol: float(
                        np.mean([r["assessment"]["balanced_accuracy"] for r in rows if r["symbol"] == symbol])
                    )
                    for symbol in {r["symbol"] for r in rows}
                },
            }
        )
    leaderboard.sort(key=lambda r: r["mean_development_balanced_accuracy"], reverse=True)
    selected = []
    for symbol, day in sorted({(r["symbol"], r["assessment_date"]) for r in records}):
        fold = [r for r in records if r["symbol"] == symbol and r["assessment_date"] == day]
        if len(fold) != 18:
            continue
        references = [r for r in fold if r["representation"] == "base" and r["model"].startswith(("logistic_", "hgb_"))]
        challengers = [r for r in fold if r["representation"] != "base"]
        def ranking(r):
            return (r["validation"]["balanced_accuracy"], -r["validation"]["log_loss"], -r["feature_count"])
        baseline = max(references, key=ranking)
        candidate = max(challengers, key=ranking)
        selected.append(
            {
                "symbol": symbol,
                "assessment_date": day,
                "reference": {k: baseline[k] for k in ["representation", "model", "validation", "assessment"]},
                "challenger": {k: candidate[k] for k in ["representation", "model", "validation", "assessment"]},
                "balanced_accuracy_difference": candidate["assessment"]["balanced_accuracy"]
                - baseline["assessment"]["balanced_accuracy"],
            }
        )
    return {
        "leaderboard": leaderboard,
        "validation_selected_comparison": selected,
        "evidence_status": "development_only_previously_exposed_dates",
        "substantial_gain_confirmed": False,
    }


def run(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    identity = {
        "protocol_sha256": sha256_file(protocol_path),
        "dataset_sha256": protocol["data"]["manifest_sha256"],
        "code_hashes": {
            name: sha256_file(ROOT / name)
            for name in ["scripts/run_boundary_predictive_screen.py", "src/lob_forge/boundary_forecasts.py"]
        },
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy", "joblib"]},
    }
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / "frozen_screen.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("keep previous run and use a new output directory after source/protocol changes")
    write_json(frozen, identity)
    features, targets, times = load_frames(protocol)
    grid = protocol["first_tabular_grid"]
    model_names = (
        [f"logistic_{value}" for value in grid["logistic_C"]]
        + [f"hgb_{value}" for value in grid["hgb_leaves"]]
        + ["extra_trees"]
    )
    dates = sorted({day for symbol, day in targets})
    records = []
    print(f"loaded sessions={len(targets)} rows={sum(len(v) for v in targets.values())} planned_fits=108", flush=True)
    with threadpool_limits(limits=2):
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            for test_index in range(5, len(dates)):
                train_dates = dates[test_index - 5 : test_index - 1]
                validation_date = dates[test_index - 1]
                assessment_date = dates[test_index]
                assert time_ms(train_dates[-1], "14:00:30") < time_ms(validation_date, "00:00:00")
                for representation in grid["representations"]:
                    train_x = pd.concat(
                        [features[symbol, day, representation] for day in train_dates], ignore_index=True
                    )
                    train_y = np.concatenate([targets[symbol, day] for day in train_dates])
                    validation_x = features[symbol, validation_date, representation]
                    assessment_x = features[symbol, assessment_date, representation]
                    for model_name in model_names:
                        job = output / symbol / assessment_date / representation / model_name
                        job.mkdir(parents=True, exist_ok=True)
                        completed = job / "completed.json"
                        if completed.exists():
                            saved = json.loads(completed.read_text())
                            if saved["run_identity"] != identity_hash(identity):
                                raise ValueError("model job identity mismatch")
                            for filename, sha in saved["artifact_hashes"].items():
                                if sha256_file(job / filename) != sha:
                                    raise ValueError("model job artifact mismatch")
                            records.append(saved["record"])
                            continue
                        started = time.monotonic()
                        write_json(
                            job / "attempt_started.json",
                            {
                                "identity": identity,
                                "train_dates": train_dates,
                                "validation_date": validation_date,
                                "assessment_date": assessment_date,
                                "model": model_name,
                                "representation": representation,
                                "status": "fit_started",
                            },
                        )
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            learner = fit_classifier(model_name, train_x, train_y, seed=grid["seed"])
                        joblib.dump(learner, job / "model.joblib")
                        restored = joblib.load(job / "model.joblib")
                        if not np.allclose(
                            restored.predict_proba(validation_x.iloc[:128]),
                            learner.predict_proba(validation_x.iloc[:128]),
                            rtol=0,
                            atol=1e-14,
                        ):
                            raise ValueError("checkpoint prediction parity failed")
                        probabilities = {}
                        metrics = {}
                        for name, day, x in [
                            ("validation", validation_date, validation_x),
                            ("assessment", assessment_date, assessment_x),
                        ]:
                            probabilities[name] = restored.predict_proba(x)
                            metrics[name] = classification_metrics(restored, probabilities[name], targets[symbol, day])
                            np.savez_compressed(
                                job / f"{name}_predictions.npz",
                                probabilities=probabilities[name],
                                labels=targets[symbol, day],
                                decision_times=times[symbol, day],
                                train_priors=restored.priors,
                            )
                        record = {
                            "symbol": symbol,
                            "assessment_date": assessment_date,
                            "validation_date": validation_date,
                            "train_dates": train_dates,
                            "representation": representation,
                            "model": model_name,
                            "feature_count": train_x.shape[1],
                            "train_rows": len(train_y),
                            "train_class_priors": learner.priors.tolist(),
                            **metrics,
                            "fit_warnings": [str(w.message) for w in caught],
                            "seconds": time.monotonic() - started,
                            "checkpoint_parity": True,
                        }
                        write_json(
                            completed,
                            {
                                "run_identity": identity_hash(identity),
                                "record": record,
                                "artifact_hashes": {
                                    f.name: sha256_file(f)
                                    for f in job.iterdir()
                                    if f.is_file() and f.name != "completed.json"
                                },
                            },
                        )
                        records.append(record)
                        write_json(output / "progress.json", {"completed_fits": len(records), **summarize(records)})
                        print(
                            f"fit={len(records)}/108 {symbol} {assessment_date} {representation}/{model_name} val_BA={metrics['validation']['balanced_accuracy']:.4f} dev_BA={metrics['assessment']['balanced_accuracy']:.4f} seconds={record['seconds']:.1f}",
                            flush=True,
                        )
    result = {"identity": identity, "completed_fits": len(records), "records": records, **summarize(records)}
    if len(records) != 108:
        raise ValueError("incomplete registered grid")
    write_json(output / "summary.json", result)
    print(f"completed_screen {output / 'summary.json'}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "docs/research/boundary_predictive_screen_20260907.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results/boundary_predictive_screen_20260907")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.protocol, args.output)
    else:
        print(json.dumps({"protocol": str(args.protocol), "output": str(args.output), "requires": "--run"}))


if __name__ == "__main__":
    main()
