"""Development-only same-day prior experiments on already exposed predictions."""

from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_adaptive_priors import POLICIES, all_policies, label_priors
from lob_forge.boundary_confirmation_integrity import verify_frozen_fold
from lob_forge.boundary_event_inputs import utc_ms
from lob_forge.boundary_forecasts import classification_metrics

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def summarize(records):
    reference = {(r["symbol"], r["date"]): r for r in records if (r["source"], r["policy"]) == ("original_reference", "registered")}
    registered = {(r["symbol"], r["date"], r["source"]): r for r in records if r["policy"] == "registered"}
    controls = {(r["symbol"], r["date"], r["policy"]): r for r in records if r["source"] == "matched_data_control"}
    leaderboard = []
    for source, policy in sorted({(r["source"], r["policy"]) for r in records}):
        rows = [r for r in records if (r["source"], r["policy"]) == (source, policy)]
        paired = []
        for day in sorted({r["date"] for r in rows}):
            panels = [r for r in rows if r["date"] == day]
            paired.append({
                "date": day,
                "balanced_accuracy": float(np.mean([r["metrics"]["balanced_accuracy"] for r in panels])),
                "delta_to_registered_reference": float(np.mean([
                    r["metrics"]["balanced_accuracy"] - reference[r["symbol"], day]["metrics"]["balanced_accuracy"] for r in panels
                ])),
            })
        leaderboard.append({
            "source": source, "policy": policy, "asset_dates": len(rows),
            "means": {k: float(np.mean([r["metrics"][k] for r in rows])) for k in ("balanced_accuracy", "natural_accuracy", "log_loss", "macro_f1")},
            "balanced_accuracy_by_asset": {s: float(np.mean([r["metrics"]["balanced_accuracy"] for r in rows if r["symbol"] == s])) for s in sorted({r["symbol"] for r in rows})},
            "mean_delta_to_registered_reference": float(np.mean([r["delta_to_registered_reference"] for r in paired])),
            "mean_delta_to_own_registered_policy": float(np.mean([
                r["metrics"]["balanced_accuracy"] - registered[r["symbol"], r["date"], source]["metrics"]["balanced_accuracy"] for r in rows
            ])),
            "mean_delta_to_same_policy_matched_control": float(np.mean([
                r["metrics"]["balanced_accuracy"] - controls[r["symbol"], r["date"], policy]["metrics"]["balanced_accuracy"] for r in rows
            ])),
            "paired_dates": paired,
        })
    return sorted(leaderboard, key=lambda r: r["means"]["balanced_accuracy"], reverse=True)


def run(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    if protocol["evidence_status"] != "development_only_on_exposed_confirmation_dates" or protocol["policies"] != list(POLICIES):
        raise ValueError("This is the registered development family, not an independent confirmation")
    for name, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / name) != checksum:
            raise ValueError(f"Registered source changed: {name}")
    identity = {
        "protocol_sha256": sha256_file(protocol_path), "input_hashes": protocol["input_hashes"],
        "python": platform.python_version(), "dependencies": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "pyarrow")},
    }
    frozen = output / "frozen_development.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve this development run after an identity change")
    write_json(frozen, identity)
    manifest = json.loads((ROOT / protocol["manifest"]).read_text())
    sessions = {(r["symbol"], r["session_date"]): r for r in manifest["sessions"]}
    records = []
    artifacts = {}
    for day in protocol["dates"]:
        verified = {}
        for source in protocol["sources"]:
            base = ROOT / source["run"]
            if source["run"] not in verified:
                verify_frozen_fold(base / "dates" / day, json.loads((base / "frozen_study.json").read_text()))
                verified[source["run"]] = True
        for symbol in protocol["symbols"]:
            session = sessions[symbol, day]
            path = ROOT / session["features_path"]
            if sha256_file(path) != session["sha256"]:
                raise ValueError("Exact-label history artifact changed")
            history = pd.read_parquet(path, columns=["decision_time", "label", "future_event_time", "quote_age_ms"])
            # Eligibility uses observed quote age and a fixed two-minute day-start
            # warm-up. Availability below, never an outcome value, gates updates.
            history = history.loc[
                (history.decision_time >= utc_ms(day, "00:02:00")) & (history.quote_age_ms <= 1000)
                & history.label.notna() & history.future_event_time.notna()
            ].set_index("decision_time")
            estimates, canonical = None, None
            for source in protocol["sources"]:
                source_path = ROOT / source["run"] / "dates" / day / "predictions" / f"{symbol}_{source['prediction']}.npz"
                with np.load(source_path) as archive:
                    p, y, times, training, registered = (archive[k].copy() for k in ("probabilities", "labels", "decision_times", "train_priors", "decision_priors"))
                np.testing.assert_array_equal(history.loc[times, "label"].to_numpy(), y)
                if canonical is None:
                    canonical = (times.copy(), y.copy())
                    estimates = {
                        "labels_cumulative" if half_life is None else f"labels_{half_life}": label_priors(
                            history.label.to_numpy(), history.index.to_numpy(), history.future_event_time.to_numpy(), times,
                            half_life_seconds=half_life,
                        ) for half_life in (None, 300, 900, 3600)
                    }
                else:
                    np.testing.assert_array_equal(times, canonical[0])
                    np.testing.assert_array_equal(y, canonical[1])
                policies = all_policies(p, times, training, registered, estimates)
                original = classification_metrics(SimpleNamespace(priors=registered), p, y)
                destination = output / "priors" / day / f"{symbol}_{source['name']}.npz"
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination, decision_times=times, **policies)
                for policy, priors in policies.items():
                    metrics = classification_metrics(SimpleNamespace(priors=priors), p, y)
                    for key in ("natural_accuracy", "log_loss"):
                        if metrics[key] != original[key]:
                            raise ValueError("Decision-only rules must preserve natural posterior metrics exactly")
                    records.append({
                        "date": day, "symbol": symbol, "source": source["name"], "policy": policy, "metrics": metrics,
                        "source_prediction": str(source_path.relative_to(ROOT)), "source_sha256": sha256_file(source_path),
                        "history_sha256": session["sha256"], "priors_artifact": str(destination.relative_to(output)),
                    })
                artifacts[str(destination.relative_to(output))] = sha256_file(destination)
            write_json(output / "progress.json", {"completed_asset_dates": len(records) // (len(POLICIES) * len(protocol["sources"])), "completed_post_fit_trials": len(records), "evidence_status": protocol["evidence_status"]})
            print(f"development_prior_panel={symbol}/{day} completed_trials={len(records)}", flush=True)
    if len(records) != protocol["post_fit_trials"]:
        raise ValueError("Incomplete registered development family")
    write_json(output / "summary.json", {
        "identity": identity, "evidence_status": protocol["evidence_status"], "post_fit_trials": len(records), "model_fits": 0,
        "substantial_gain_confirmed": False, "natural_posterior_metrics_unchanged": True,
        "leaderboard": summarize(records), "records": records, "artifact_hashes": artifacts,
    })
    print(f"development_prior_screen_complete {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_same_day_priors_20260907.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/boundary_same_day_priors_20260907")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.protocol, args.output)
    else:
        print("Pass --run for the development-only prior family on the explicitly exposed dates.")
