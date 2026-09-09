"""Verify and report every frozen counted-depth predictive comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_forecasts import classification_metrics
from run_boundary_confirmation import write_json
from run_boundary_event_clock_screen import check_completed
from run_boundary_okx_counted_screen import BASELINES, BLENDS, MODEL_SPECS, SYMBOLS

ROOT = Path(__file__).resolve().parents[1]


def paired_comparison(records, candidate, reference):
    left = {(r["date"], r["symbol"]): r["metrics"] for r in records
            if r["model"] == candidate and r["policy"] == "forecast_3600"}
    right = {(r["date"], r["symbol"]): r["metrics"] for r in records
             if r["model"] == reference and r["policy"] == "forecast_3600"}
    if set(left) != set(right) or len(left) != 6:
        raise ValueError("Every comparison requires the same six frozen asset/date panels")
    deltas = {k: 100 * (left[k]["balanced_accuracy"] - right[k]["balanced_accuracy"]) for k in left}
    return {"candidate": candidate, "reference": reference,
        "balanced_accuracy_gain_pp": float(np.mean(list(deltas.values()))),
        "gain_pp_by_asset": {s: float(np.mean([v for (_, asset), v in deltas.items() if asset == s])) for s in SYMBOLS},
        "gain_pp_by_date": {d: float(np.mean([v for (day, _), v in deltas.items() if day == d])) for d in sorted({d for d, _ in deltas})},
        "log_loss_ratio": float(np.mean([v["log_loss"] for v in left.values()]) / np.mean([v["log_loss"] for v in right.values()]))}


def assess(protocol_path, output, evidence_path, report_path):
    if evidence_path.exists() or report_path.exists():
        raise ValueError("Preserve previous counted-model evidence and reporting attempts")
    protocol = json.loads(protocol_path.read_text())
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    identity = json.loads((output / "frozen_screen.json").read_text())
    if (identity["protocol_sha256"] != sha256_file(protocol_path) or summary["identity"] != identity
        or protocol["model_specs"] != MODEL_SPECS or summary["trainable_model_fits"] != 66
        or summary["post_fit_panels"] != 744 or summary["substantial_gain_confirmed"]):
        raise ValueError("The exact complete development study is required")
    for path, digest in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError(f"Frozen counted-model input changed before reporting: {path}")
    names = (*BASELINES, *MODEL_SPECS, *BLENDS)
    expected = {(d, s, n, p) for d in protocol["dates"] for s in SYMBOLS for n in names for p in ("registered", "forecast_3600")}
    indexed = {(r["date"], r["symbol"], r["model"], r["policy"]): r for r in summary["records"]}
    if set(indexed) != expected or len(summary["records"]) != len(expected):
        raise ValueError("Every fixed reporting panel must occur exactly once")
    baseline = ROOT / protocol["baseline_run"]
    old = json.loads((baseline / "summary.json").read_text())
    original_metrics = {(r["date"], r["symbol"], r["model"], r["policy"]): r["metrics"] for r in old["records"]}
    resources, errors, column_counts, train_rows = [], [], {}, {}
    parity_panels = 0
    for day in protocol["dates"]:
        folder = output / "dates" / day
        check_completed(folder, identity)
        for name in MODEL_SPECS:
            worker = json.loads((folder / "workers" / name / "worker.json").read_text())
            supervision = json.loads((folder / f"{name}.supervision.json").read_text())
            if (supervision["returncode"] != 0 or supervision["failure"] is not None
                or worker["assessment_labels_decoded"] or worker["specification"] != MODEL_SPECS[name]):
                raise ValueError("Every worker must pass its frozen execution and label-access checks")
            for symbol in SYMBOLS:
                check = worker["checks"][symbol]
                if not check["checkpoint_probabilities_exact"] or not check["training_priors_exact"]:
                    raise ValueError("All checkpoints and training priors must reproduce exactly")
                errors.append(check["errors"])
            resources.append({"date": day, "model": name, "fit_save_reload_seconds": worker["fit_save_reload_seconds"],
                "supervised_seconds": supervision["seconds"], **worker["memory"]})
            column_counts.setdefault(name, []).append(len(worker["columns"]))
            train_rows.setdefault(day, worker["train_rows"])
            if train_rows[day] != worker["train_rows"]:
                raise ValueError("Matched models must retain identical training row universes")
        for symbol in SYMBOLS:
            for name in names:
                for policy in ("registered", "forecast_3600"):
                    key = day, symbol, name, policy
                    file = folder / "predictions" / f"{symbol}_{name}_{policy}.npz"
                    with np.load(file, allow_pickle=False) as saved:
                        if len(saved["labels"]) != 7070:
                            raise ValueError("Every procedure must preserve all original noon rows")
                        recomputed = classification_metrics(SimpleNamespace(priors=saved["decision_priors"]), saved["probabilities"], saved["labels"])
                        if recomputed != indexed[key]["metrics"]:
                            raise ValueError("Metrics must reproduce exactly from frozen forecasts")
                        if name in BASELINES:
                            with np.load(baseline / "dates" / day / "predictions" / file.name, allow_pickle=False) as retained:
                                for field in saved.files:
                                    np.testing.assert_array_equal(saved[field], retained[field])
                            if recomputed != original_metrics[key]:
                                raise ValueError("Every retained metric must reproduce exactly")
                            parity_panels += 1
    maximum_errors = {k: max(r[k] for r in errors) for k in errors[0]}
    if (maximum_errors["future_prefix"] > protocol["future_prefix_absolute_tolerance"]
        or max(maximum_errors.values()) > protocol["query_partition_absolute_tolerance"]):
        raise ValueError("Frozen query-independence tolerances must pass")
    expert = json.loads((ROOT / "results/boundary_expert_feedback_screen_20260908/summary.json").read_text())
    expert_name = "confidence_context_eta10_h300"
    expert_records = [{**r, "model": r["case"]} for r in expert["records"]
                      if r["case"] == expert_name and r["policy"] == "forecast_3600"]
    records = [*summary["records"], *expert_records]
    novel = (*MODEL_SPECS, *BLENDS)
    rule = protocol["development_advancement"]
    comparisons = [paired_comparison(records, n, rule["reference"]) for n in novel]
    for row in comparisons:
        row["development_expansion_gate_passes"] = (row["balanced_accuracy_gain_pp"] >= rule["minimum_mean_gain_pp"]
            and all(gain > 0 for gain in row["gain_pp_by_asset"].values()) and row["log_loss_ratio"] <= rule["maximum_log_loss_ratio"])
    comparisons.sort(key=lambda r: r["balanced_accuracy_gain_pp"], reverse=True)
    count_pairs = []
    for name in novel:
        tokens = name.split("_")
        count_tokens = [i for i, token in enumerate(tokens) if token in ("n25", "n100")]
        if count_tokens:
            if len(count_tokens) != 1:
                raise ValueError("Each counted procedure requires one unambiguous quantity-only counterpart")
            i = count_tokens[0]
            tokens[i] = "q" + tokens[i][1:]
            reference = "_".join(tokens)
            if reference not in novel:
                raise ValueError("Every counted procedure needs its exact matched quantity control")
            count_pairs.append(paired_comparison(records, name, reference))
    if len(count_pairs) != 27:
        raise ValueError("All 27 matched quantity/count procedure pairs are required")
    primary = sorted([r for r in summary["leaderboard"] if r["policy"] == "forecast_3600"],
                     key=lambda r: r["means"]["balanced_accuracy"], reverse=True)
    evidence = {"status": "complete_exposed_development_only", "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": sha256_file(protocol_path), "summary": str(summary_path.relative_to(ROOT)),
        "summary_sha256": sha256_file(summary_path), "input_hashes_verified": len(protocol["input_hashes"]),
        "trainable_model_fits": 66, "reporting_panels_recomputed_exactly": 744, "retained_panels_exact": parity_panels,
        "assessment_rows_per_source": 42420, "resources": resources, "input_columns_by_model_and_date": column_counts,
        "training_rows_by_date": train_rows, "maximum_query_errors": maximum_errors,
        "all_saved_checkpoint_and_prior_checks_passed": True, "assessment_labels_decoded_by_training_workers": False,
        "primary_leaderboard": primary, "candidate_vs_complete_control": comparisons,
        "count_minus_quantity_pairs": count_pairs,
        "candidate_vs_expert": [paired_comparison(records, n, expert_name) for n in novel],
        "development_expansion_candidates": [r["candidate"] for r in comparisons if r["development_expansion_gate_passes"]],
        "substantial_gain_confirmed": False, "independent_confirmation_rounds_consumed": 0}
    write_json(evidence_path, evidence)
    best = comparisons[0]
    score = next(r for r in primary if r["model"] == best["candidate"])
    lines = ["# Matched OKX quantity and order-count predictive screen", "",
        f"The best new procedure is `{best['candidate']}`: **{100*score['means']['balanced_accuracy']:.6f}% balanced accuracy**, "
        f"a **{best['balanced_accuracy_gain_pp']:+.6f} percentage-point** difference from the retained complete procedure. "
        f"Its natural log-loss ratio is {best['log_loss_ratio']:.9f}. These are three repeatedly exposed development dates, not independent confirmation.", "",
        "The frozen study fitted all 66 models, reported all 744 panels, and retained all 42,420 assessment rows per forecast source. "
        "Every reported panel was recomputed from its saved forecasts; all 96 retained panels and forecast arrays reproduced exactly. "
        "All checkpoint, training-prior and bounded query-independence checks passed.", "",
        "| Procedure | Balanced accuracy | Log loss | Gain vs complete control (pp) |", "| --- | ---: | ---: | ---: |"]
    differences = {r["candidate"]: r["balanced_accuracy_gain_pp"] for r in comparisons}
    for row in primary:
        gain = differences.get(row["model"])
        formatted = f"{gain:+.6f}" if gain is not None else "retained control"
        lines.append(f"| `{row['model']}` | {100*row['means']['balanced_accuracy']:.6f}% | {row['means']['log_loss']:.9f} | {formatted} |")
    lines.extend(["", "Matched count-minus-quantity comparisons:", "",
        "| Count procedure | Mean gain (pp) | BTC gain (pp) | ETH gain (pp) | Log-loss ratio |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in count_pairs:
        lines.append(f"| `{row['candidate']}` | {row['balanced_accuracy_gain_pp']:+.6f} | {row['gain_pp_by_asset']['BTCUSDT']:+.6f} | "
                     f"{row['gain_pp_by_asset']['ETHUSDT']:+.6f} | {row['log_loss_ratio']:.9f} |")
    lines.extend(["", f"Candidates passing the prospectively fixed development expansion rule: {evidence['development_expansion_candidates'] or 'none'}.", "",
        "The rule requires at least 0.5 points above the retained complete procedure, improvement on both assets, and a log-loss ratio at most 1.01. "
        "Passing this rule only justifies broader evaluation. The unchanged substantial-gain requirement still needs at least five points above the original registered reference "
        "and the frozen uncertainty, per-asset and twenty-independent-date requirements. This experiment consumes no independent confirmation round.", "",
        "Historical publisher-time sampling does not certify live source arrival or execution latency. Published quantities are not converted across venues. "
        "Count-specific conclusions require the matched quantity-control comparison; all candidates were selected from exposed development data.", "",
        f"Protocol SHA256: `{sha256_file(protocol_path)}`. Full result SHA256: `{sha256_file(summary_path)}`. "
        f"Tracked evidence: `{evidence_path.relative_to(ROOT)}`, SHA256 `{sha256_file(evidence_path)}`."])
    report_path.write_text("\n".join(lines) + "\n")
    print(json.dumps({"best_new": best, "count_pairs": len(count_pairs), "expansion_candidates": evidence["development_expansion_candidates"],
        "summary_sha256": sha256_file(summary_path), "evidence_sha256": sha256_file(evidence_path)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    assess(args.protocol.resolve(), args.output.resolve(), args.evidence.resolve(), args.report.resolve())
