"""Run registered cross-asset neural and parameter-sharing ensemble experiments."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer
from threadpoolctl import threadpool_limits

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_event_inputs import load_event_inputs, utc_ms
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_pooled import PooledForecaster, balanced_asset_weights, build_member_network

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ["BTCUSDT", "ETHUSDT"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def summarize(records, references):
    leaderboard, comparison = [], []
    for name in sorted({r["model"] for r in records}):
        rows = [r for r in records if r["model"] == name]
        leaderboard.append(
            {
                "model": name,
                "asset_folds": len(rows),
                **{
                    f"mean_development_{metric}": float(np.mean([r["assessment"][metric] for r in rows]))
                    for metric in ["balanced_accuracy", "natural_accuracy", "log_loss"]
                },
            }
        )

    def rank(record):
        return (record["validation"]["balanced_accuracy"], -record["validation"]["log_loss"])

    for ref in references:
        rows = [r for r in records if (r["symbol"], r["assessment_date"]) == (ref["symbol"], ref["assessment_date"])]
        if len(rows) != 3:
            continue
        best = max(rows, key=rank)
        comparison.append(
            {
                "symbol": ref["symbol"],
                "assessment_date": ref["assessment_date"],
                "reference": ref["reference"],
                "challenger": {k: best[k] for k in ["model", "validation", "assessment"]},
                "balanced_accuracy_difference": best["assessment"]["balanced_accuracy"]
                - ref["reference"]["assessment"]["balanced_accuracy"],
            }
        )
    return {
        "leaderboard": sorted(leaderboard, key=lambda r: r["mean_development_balanced_accuracy"], reverse=True),
        "validation_selected_comparison": comparison,
        "evidence_status": "development_only_previously_exposed_dates",
        "substantial_gain_confirmed": False,
    }


def run(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    manifest = ROOT / "data/research/boundary_event_data_20260907/dataset_manifest.json"
    reference_path = ROOT / "results/boundary_predictive_screen_20260907/summary.json"
    references = json.loads(reference_path.read_text())["validation_selected_comparison"]
    identity = {
        "protocol_sha256": sha256_file(protocol_path),
        "manifest_sha256": sha256_file(manifest),
        "reference_sha256": sha256_file(reference_path),
        "code_hashes": {
            name: sha256_file(ROOT / name)
            for name in [
                "scripts/run_boundary_pooled_neural.py",
                "src/lob_forge/boundary_pooled.py",
                "src/lob_forge/boundary_neural.py",
                "src/lob_forge/boundary_event_inputs.py",
                "src/lob_forge/boundary_forecasts.py",
                "src/lob_forge/boundary_events.py",
            ]
        },
        "python": platform.python_version(),
        "dependencies": {
            name: version(name) for name in ["torch", "numpy", "pandas", "scikit-learn", "scipy", "joblib", "pyarrow"]
        },
    }
    frozen = output / "frozen_screen.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve this experiment and use a new output directory after source changes")
    write_json(frozen, identity)
    features, labels, times = load_event_inputs(ROOT, manifest)
    dates = sorted({day for symbol, day in labels})
    records, completed_fits = [], 0
    torch.set_num_threads(protocol["threads"])
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        for index in range(5, len(dates)):
            train_dates, val_day, dev_day = dates[index - 5 : index - 1], dates[index - 1], dates[index]
            for variant in protocol["variants"]:
                groups = [SYMBOLS] if variant["pooled"] else [[symbol] for symbol in SYMBOLS]
                for symbols in groups:
                    job = output / dev_day / variant["name"] / "_".join(symbols)
                    completed = job / "completed.json"
                    if completed.exists():
                        saved = json.loads(completed.read_text())
                        if saved["identity"] != identity:
                            raise ValueError("Run identity mismatch")
                        for name, checksum in saved["artifact_hashes"].items():
                            if sha256_file(job / name) != checksum:
                                raise ValueError("Checkpoint artifact changed")
                        records.extend(saved["records"])
                        completed_fits += 1
                        continue
                    started = time.monotonic()
                    write_json(
                        job / "attempt_started.json",
                        {
                            "identity": identity,
                            "variant": variant,
                            "train_dates": train_dates,
                            "symbols": symbols,
                            "validation_date": val_day,
                            "assessment_date": dev_day,
                        },
                    )
                    torch.manual_seed(protocol["seed"])
                    rng = np.random.default_rng(protocol["seed"])
                    columns = list(features[symbols[0], train_dates[0]].columns)
                    train_x = pd.concat(
                        [features[symbol, day] for symbol in symbols for day in train_dates], ignore_index=True
                    )
                    train_y = np.concatenate([labels[symbol, day] for symbol in symbols for day in train_dates])
                    assets = np.concatenate(
                        [
                            np.full(len(labels[symbol, day]), SYMBOLS.index(symbol), dtype=int)
                            for symbol in symbols
                            for day in train_dates
                        ]
                    )
                    priors, weights = balanced_asset_weights(train_y, assets)
                    raw_matrix = train_x.to_numpy(dtype=float)
                    active = np.ptp(raw_matrix, axis=0) > 0
                    normalizer = QuantileTransformer(
                        n_quantiles=min(1024, len(train_x)),
                        output_distribution="normal",
                        subsample=min(100000, len(train_x)),
                        random_state=protocol["seed"],
                    )
                    normalizer.fit(raw_matrix[:, active])
                    normalized = np.column_stack([normalizer.transform(raw_matrix[:, active]), 2 * assets - 1]).astype(
                        np.float32
                    )
                    network = build_member_network(
                        normalized.shape[1], members=variant["members"], hidden_size=protocol["hidden_size"]
                    )
                    learner = PooledForecaster(
                        network, normalizer, active, columns, priors, variant["members"], protocol["hidden_size"]
                    )
                    partitions = {}
                    for symbol in symbols:
                        for partition, day in [("validation", val_day), ("assessment", dev_day)]:
                            clock = times[symbol, day]
                            keep = (clock >= utc_ms(day, "12:02:00")) & (clock < utc_ms(day, "13:59:50"))
                            baseline = np.load(
                                ROOT
                                / "results/boundary_predictive_screen_20260907"
                                / symbol
                                / dev_day
                                / "base/logistic_0.01"
                                / f"{partition}_predictions.npz"
                            )
                            np.testing.assert_array_equal(clock[keep], baseline["decision_times"])
                            np.testing.assert_array_equal(labels[symbol, day][keep], baseline["labels"])
                            partitions[symbol, partition] = (
                                features[symbol, day].loc[keep],
                                labels[symbol, day][keep],
                                clock[keep],
                            )
                    optimizer = torch.optim.AdamW(
                        network.parameters(), lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"]
                    )
                    target, sample_weights = torch.tensor(train_y + 1, dtype=torch.long), torch.from_numpy(weights)
                    best_key, best_state, best_epoch, history = (-np.inf, -np.inf), None, None, []
                    for epoch in range(1, protocol["epochs"] + 1):
                        network.train()
                        order = rng.permutation(len(train_y))
                        losses = []
                        for start in range(0, len(order), protocol["batch_size"]):
                            rows = order[start : start + protocol["batch_size"]]
                            logits = network(torch.from_numpy(normalized[rows]))
                            repeated_y = target[rows, None].expand(-1, variant["members"]).reshape(-1)
                            member_losses = torch.nn.functional.cross_entropy(
                                logits.reshape(-1, 3), repeated_y, reduction="none"
                            ).reshape(len(rows), -1)
                            loss = (member_losses.mean(dim=1) * sample_weights[rows]).mean()
                            if not torch.isfinite(loss):
                                raise ValueError("Nonfinite ensemble training loss")
                            optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(network.parameters(), protocol["gradient_norm_limit"])
                            optimizer.step()
                            losses.append(float(loss.detach()))
                        validation = {}
                        for symbol in symbols:
                            x, y, _ = partitions[symbol, "validation"]
                            asset = SYMBOLS.index(symbol)
                            validation[symbol] = classification_metrics(
                                SimpleNamespace(priors=priors[asset]), learner.predict_proba(x, asset), y
                            )
                        key = (
                            float(np.mean([m["balanced_accuracy"] for m in validation.values()])),
                            -float(np.mean([m["log_loss"] for m in validation.values()])),
                        )
                        if key > best_key:
                            best_key, best_state, best_epoch = key, copy.deepcopy(network.state_dict()), epoch
                        history.append(
                            {"epoch": epoch, "mean_training_loss": float(np.mean(losses)), "validation": validation}
                        )
                        write_json(job / "training_progress.json", {"history": history, "best_epoch": best_epoch})
                        print(
                            f"{dev_day} {variant['name']} {'/'.join(symbols)} epoch={epoch}/{protocol['epochs']} "
                            f"mean_validation_BA={key[0]:.4f}",
                            flush=True,
                        )
                    network.load_state_dict(best_state)
                    learner.save(job)
                    restored = PooledForecaster.load(job)
                    job_records = []
                    for symbol in symbols:
                        asset = SYMBOLS.index(symbol)
                        vx = partitions[symbol, "validation"][0].iloc[:128]
                        np.testing.assert_array_equal(
                            restored.predict_proba(vx, asset), learner.predict_proba(vx, asset)
                        )
                        metrics = {}
                        for partition in ["validation", "assessment"]:
                            x, y, clock = partitions[symbol, partition]
                            p = restored.predict_proba(x, asset)
                            metrics[partition] = classification_metrics(SimpleNamespace(priors=priors[asset]), p, y)
                            np.savez_compressed(
                                job / f"{symbol}_{partition}_predictions.npz",
                                probabilities=p,
                                labels=y,
                                decision_times=clock,
                                train_priors=priors[asset],
                            )
                        job_records.append(
                            {
                                "symbol": symbol,
                                "model": variant["name"],
                                "assessment_date": dev_day,
                                "validation_date": val_day,
                                "train_dates": train_dates,
                                "training_assets": symbols,
                                "train_rows": len(train_y),
                                "active_features": int(active.sum()),
                                "best_epoch": best_epoch,
                                "checkpoint_parity": True,
                                "model_parameters": sum(p.numel() for p in network.parameters()),
                                "train_class_priors": priors[asset].tolist(),
                                "seconds": time.monotonic() - started,
                                **metrics,
                            }
                        )
                    write_json(
                        completed,
                        {
                            "identity": identity,
                            "records": job_records,
                            "artifact_hashes": {
                                path.name: sha256_file(path)
                                for path in job.iterdir()
                                if path.is_file() and path != completed
                            },
                        },
                    )
                    records.extend(job_records)
                    completed_fits += 1
                    write_json(
                        output / "progress.json", {"completed_fits": completed_fits, **summarize(records, references)}
                    )
                    print(f"completed_fits={completed_fits}/12 seconds={time.monotonic() - started:.1f}", flush=True)
    if completed_fits != protocol["planned_model_fits"] or len(records) != 18:
        raise ValueError("Incomplete registered neural comparison")
    write_json(
        output / "summary.json",
        {"identity": identity, "completed_fits": completed_fits, "records": records, **summarize(records, references)},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_pooled_neural_20260907.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/boundary_pooled_neural_20260907")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.protocol, args.output)
    else:
        print("Pass --run for 12 registered full-day neural model fits.")
