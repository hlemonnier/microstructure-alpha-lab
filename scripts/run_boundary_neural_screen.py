"""Run registered causal sequence development models with frozen checkpoints."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_neural import NeuralForecaster, SequenceNormalizer, causal_window_indices
from lob_forge.ml_models import build_torch_sequence_classifier

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("structural_screen", ROOT / "scripts/run_boundary_structural_screen.py")
structural = importlib.util.module_from_spec(spec)
spec.loader.exec_module(structural)
screen = structural.screen


def selected_comparison(records, references):
    comparison = []
    for reference in references:
        rows = [
            r
            for r in records
            if (r["symbol"], r["assessment_date"]) == (reference["symbol"], reference["assessment_date"])
        ]
        if len(rows) != 3:
            continue
        best = max(rows, key=lambda r: (r["validation"]["balanced_accuracy"], -r["validation"]["log_loss"]))
        comparison.append(
            {
                "symbol": best["symbol"],
                "assessment_date": best["assessment_date"],
                "reference": reference["reference"],
                "challenger": {key: best[key] for key in ["model", "validation", "assessment"]},
                "balanced_accuracy_difference": best["assessment"]["balanced_accuracy"]
                - reference["reference"]["assessment"]["balanced_accuracy"],
            }
        )
    summary = structural.summarize(records, [])
    summary["validation_selected_comparison"] = comparison
    return summary


def run(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    source_protocol = ROOT / "docs/research/boundary_predictive_screen_20260907.json"
    reference_path = ROOT / "results/boundary_predictive_screen_20260907/summary.json"
    references = json.loads(reference_path.read_text())["validation_selected_comparison"]
    identity = {
        "protocol_sha256": sha256_file(protocol_path),
        "source_protocol_sha256": sha256_file(source_protocol),
        "reference_sha256": sha256_file(reference_path),
        "code_hashes": {
            name: sha256_file(ROOT / name)
            for name in [
                "scripts/run_boundary_neural_screen.py",
                "scripts/run_boundary_predictive_screen.py",
                "scripts/run_boundary_structural_screen.py",
                "src/lob_forge/boundary_forecasts.py",
                "src/lob_forge/boundary_neural.py",
                "src/lob_forge/ml_models.py",
            ]
        },
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ["torch", "numpy", "pandas", "scikit-learn", "scipy"]},
    }
    frozen = output / "frozen_screen.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Source changed: preserve this experiment and use a new output directory")
    screen.write_json(frozen, identity)
    data_protocol = json.loads(source_protocol.read_text())
    data_protocol["data"]["decision_window_utc"][0] = "12:00:00"
    data_protocol["first_tabular_grid"]["representations"] = ["temporal_cross"]
    frames, labels, times = screen.load_frames(data_protocol)
    columns = list(next(iter(frames.values())).columns[:26])
    dates = sorted({day for symbol, day in labels})
    records = []
    torch.set_num_threads(protocol["threads"])
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            for date_index in range(5, len(dates)):
                train_dates, val_date, dev_date = (
                    dates[date_index - 5 : date_index - 1],
                    dates[date_index - 1],
                    dates[date_index],
                )
                train_x = pd.concat([frames[symbol, day, "temporal_cross"][columns] for day in train_dates]).to_numpy()
                train_times = np.concatenate([times[symbol, day] for day in train_dates])
                train_keep = np.concatenate(
                    [times[symbol, day] >= screen.time_ms(day, "12:02:00") for day in train_dates]
                )
                train_y = np.concatenate([labels[symbol, day] for day in train_dates])[train_keep]
                normalizer = SequenceNormalizer.fit(train_x)
                normalized_train = normalizer.transform(train_x)
                priors = np.array([(train_y == label).mean() for label in [-1, 0, 1]])
                for variant in protocol["variants"]:
                    job = output / symbol / dev_date / variant["name"]
                    completed = job / "completed.json"
                    if completed.exists():
                        saved = json.loads(completed.read_text())
                        if saved["run_identity"] != screen.identity_hash(identity):
                            raise ValueError("Run identity mismatch")
                        for filename, checksum in saved["artifact_hashes"].items():
                            if sha256_file(job / filename) != checksum:
                                raise ValueError("Artifact checksum mismatch")
                        records.append(saved["record"])
                        continue
                    screen.write_json(
                        job / "attempt_started.json",
                        {
                            "identity": identity,
                            "variant": variant,
                            "train_dates": train_dates,
                            "validation_date": val_date,
                            "assessment_date": dev_date,
                        },
                    )
                    started = time.monotonic()
                    torch.manual_seed(protocol["seed"])
                    rng = np.random.default_rng(protocol["seed"])
                    network = build_torch_sequence_classifier(
                        window=variant["window"],
                        feature_count=len(columns),
                        model_name=variant["architecture"],
                        hidden_size=protocol["hidden_size"],
                    )
                    model = NeuralForecaster(
                        network,
                        normalizer,
                        columns,
                        priors,
                        variant["architecture"],
                        variant["window"],
                        protocol["hidden_size"],
                    )
                    train_indices = causal_window_indices(train_times, variant["window"])[train_keep]
                    partitions = {}
                    for partition, day in [("validation", val_date), ("assessment", dev_date)]:
                        keep = times[symbol, day] >= screen.time_ms(day, "12:02:00")
                        partitions[partition] = (
                            frames[symbol, day, "temporal_cross"][columns].to_numpy(),
                            causal_window_indices(times[symbol, day], variant["window"])[keep],
                            labels[symbol, day][keep],
                            times[symbol, day][keep],
                        )
                    vx, vi, vy, vt = partitions["validation"]
                    optimizer = torch.optim.AdamW(
                        network.parameters(), lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"]
                    )
                    weights = torch.tensor(1 / (3 * priors), dtype=torch.float32)
                    target = torch.tensor(train_y + 1, dtype=torch.long)
                    best_key, best_state, best_epoch, history = (-np.inf, -np.inf), None, None, []
                    for epoch in range(1, protocol["epochs"] + 1):
                        network.train()
                        order = rng.permutation(len(train_y))
                        losses = []
                        for start in range(0, len(order), protocol["batch_size"]):
                            rows = order[start : start + protocol["batch_size"]]
                            x = torch.from_numpy(normalized_train[train_indices[rows]])
                            logits = network(x)
                            loss = (
                                torch.nn.functional.cross_entropy(logits, target[rows], reduction="none")
                                * weights[target[rows]]
                            ).mean()
                            if not torch.isfinite(loss):
                                raise ValueError("Nonfinite neural training loss")
                            optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(network.parameters(), protocol["gradient_norm_limit"])
                            optimizer.step()
                            losses.append(float(loss.detach()))
                        validation = classification_metrics(model, model.predict_proba(vx, vi), vy)
                        key = (validation["balanced_accuracy"], -validation["log_loss"])
                        if key > best_key:
                            best_key, best_state, best_epoch = key, copy.deepcopy(network.state_dict()), epoch
                        history.append(
                            {"epoch": epoch, "training_loss": float(np.mean(losses)), "validation": validation}
                        )
                        screen.write_json(job / "training_progress.json", {"epochs": history, "best_epoch": best_epoch})
                        print(
                            f"{symbol} {dev_date} {variant['name']} epoch={epoch}/{protocol['epochs']} "
                            f"validation_BA={validation['balanced_accuracy']:.4f}",
                            flush=True,
                        )
                    network.load_state_dict(best_state)
                    model.save(job / "model.pt")
                    restored = NeuralForecaster.load(job / "model.pt")
                    np.testing.assert_allclose(
                        model.predict_proba(vx, vi[:128]), restored.predict_proba(vx, vi[:128]), rtol=0, atol=0
                    )
                    metrics = {}
                    for partition, (x, indices, y, clock) in partitions.items():
                        p = restored.predict_proba(x, indices)
                        metrics[partition] = classification_metrics(restored, p, y)
                        np.savez_compressed(
                            job / f"{partition}_predictions.npz",
                            probabilities=p,
                            labels=y,
                            decision_times=clock,
                            train_priors=priors,
                        )
                        day = val_date if partition == "validation" else dev_date
                        baseline_file = (
                            ROOT
                            / "results/boundary_predictive_screen_20260907"
                            / symbol
                            / dev_date
                            / "base/logistic_0.01"
                            / f"{partition}_predictions.npz"
                        )
                        baseline = np.load(baseline_file)
                        np.testing.assert_array_equal(clock, baseline["decision_times"])
                        np.testing.assert_array_equal(y, baseline["labels"])
                    record = {
                        "symbol": symbol,
                        "model": variant["name"],
                        "train_dates": train_dates,
                        "validation_date": val_date,
                        "assessment_date": dev_date,
                        "train_rows": len(train_y),
                        "train_class_priors": priors.tolist(),
                        "checkpoint_parity": True,
                        "best_epoch": best_epoch,
                        "epoch_history": history,
                        **metrics,
                        "seconds": time.monotonic() - started,
                    }
                    screen.write_json(
                        completed,
                        {
                            "run_identity": screen.identity_hash(identity),
                            "record": record,
                            "artifact_hashes": {
                                p.name: sha256_file(p) for p in job.iterdir() if p.is_file() and p != completed
                            },
                        },
                    )
                    records.append(record)
                    screen.write_json(
                        output / "progress.json",
                        {"completed_fits": len(records), **selected_comparison(records, references)},
                    )
                    print(
                        f"fit={len(records)}/18 {symbol} {dev_date} {variant['name']} "
                        f"dev_BA={metrics['assessment']['balanced_accuracy']:.4f} seconds={record['seconds']:.1f}",
                        flush=True,
                    )
    if len(records) != 18:
        raise ValueError("Incomplete registered experiment")
    screen.write_json(
        output / "summary.json",
        {
            "identity": identity,
            "completed_fits": len(records),
            "records": records,
            **selected_comparison(records, references),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_neural_screen_20260907.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/boundary_neural_screen_20260907")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.protocol, args.output)
    else:
        print("Pass --run to execute the registered 18-fit neural development screen.")
