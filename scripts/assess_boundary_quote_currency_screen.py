"""Verify every fixed native-source, conversion-control and retained comparison."""

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
from run_boundary_quote_currency_screen import BASELINES, BLENDS, MODEL_SPECS, SYMBOLS
from assess_boundary_okx_counted_screen import paired_comparison

ROOT = Path(__file__).resolve().parents[1]


def assess(protocol_path, output, evidence_path, report_path):
    if evidence_path.exists() or report_path.exists():
        raise ValueError("Preserve previous quote-source model evidence and reporting attempts")
    protocol = json.loads(protocol_path.read_text())
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    identity = json.loads((output / "frozen_screen.json").read_text())
    if (identity["protocol_sha256"] != sha256_file(protocol_path) or summary["identity"] != identity
        or protocol["model_specs"] != MODEL_SPECS or summary["trainable_model_fits"] != 36
        or summary["post_fit_panels"] != 456 or summary["substantial_gain_confirmed"]):
        raise ValueError("The exact complete development study is required")
    for path, digest in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError(f"Frozen quote-source model input changed before reporting: {path}")
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
    expert_spec = protocol["expert_reference"]
    expert = json.loads((ROOT / expert_spec["summary"]).read_text())
    expert_name = f"expert::{expert_spec['case']}"
    expert_records = [{**r, "model": expert_name} for r in expert["records"]
                      if r["case"] == expert_spec["case"] and r["policy"] == "forecast_3600"]
    counted_spec = protocol["counted_reference"]
    counted_path = ROOT / counted_spec["run"]
    counted = json.loads((counted_path / "summary.json").read_text())
    counted_name = f"counted::{counted_spec['model']}"
    counted_records = [{**r, "model": counted_name} for r in counted["records"]
                       if r["model"] == counted_spec["model"] and r["policy"] == "forecast_3600"]
    for row in counted_records:
        path = counted_path / "dates" / row["date"] / "predictions" / f"{row['symbol']}_{counted_spec['model']}_forecast_3600.npz"
        with np.load(path, allow_pickle=False) as saved:
            recomputed = classification_metrics(SimpleNamespace(priors=saved["decision_priors"]), saved["probabilities"], saved["labels"])
            if recomputed != row["metrics"]:
                raise ValueError("The strongest completed counted-depth reference must retain exact metric replay")
    records = [*summary["records"], *expert_records, *counted_records]
    novel = (*MODEL_SPECS, *BLENDS)
    rule = protocol["development_advancement"]
    comparisons = {n: paired_comparison(records, n, rule["reference"]) for n in novel}
    extra_names = (counted_name, expert_name)
    stronger_names = [n for n in extra_names if paired_comparison(records, n, rule["reference"])["balanced_accuracy_gain_pp"] > 0]
    extra_comparisons = {n: {reference: paired_comparison(records, n, reference) for reference in extra_names} for n in novel}
    native_pairs, both_minus_btc, advancing = [], [], []

    def passes(comparison, minimum_gain):
        return (comparison["balanced_accuracy_gain_pp"] >= minimum_gain
            and all(gain > 0 for gain in comparison["gain_pp_by_asset"].values())
            and comparison["log_loss_ratio"] <= rule["maximum_log_loss_ratio"])

    for name in novel:
        tokens = name.split("_")
        variant = tokens[1]
        if variant not in ("btc100", "both100"):
            continue
        fx = "_".join([tokens[0], "fx100", *tokens[2:]])
        if fx not in novel:
            raise ValueError("Every native procedure requires its exact FX-only counterpart")
        native = paired_comparison(records, name, fx)
        native["development_expansion_gate_passes"] = (
            passes(comparisons[name], rule["minimum_mean_gain_pp"])
            and passes(native, rule["minimum_native_vs_fx_gain_pp"])
            and all(passes(extra_comparisons[name][reference], rule["minimum_mean_gain_pp"]) for reference in stronger_names))
        native_pairs.append(native)
        if native["development_expansion_gate_passes"]:
            advancing.append(name)
        if variant == "both100":
            btc = "_".join([tokens[0], "btc100", *tokens[2:]])
            both_minus_btc.append(paired_comparison(records, name, btc))
    if len(native_pairs) != 20 or len(both_minus_btc) != 10:
        raise ValueError("Every fixed native-source attribution comparison must be present")
    ordered = sorted(comparisons.values(), key=lambda r: r["balanced_accuracy_gain_pp"], reverse=True)
    primary = sorted([r for r in summary["leaderboard"] if r["policy"] == "forecast_3600"],
                     key=lambda r: r["means"]["balanced_accuracy"], reverse=True)
    evidence = {"status": "complete_exposed_development_only", "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": sha256_file(protocol_path), "summary": str(summary_path.relative_to(ROOT)),
        "summary_sha256": sha256_file(summary_path), "input_hashes_verified": len(protocol["input_hashes"]),
        "trainable_model_fits": 36, "reporting_panels_recomputed_exactly": 456, "retained_panels_exact": parity_panels,
        "assessment_rows_per_source": 42420, "resources": resources, "input_columns_by_model_and_date": column_counts,
        "training_rows_by_date": train_rows, "maximum_query_errors": maximum_errors,
        "all_saved_checkpoint_and_prior_checks_passed": True, "assessment_labels_decoded_by_training_workers": False,
        "primary_leaderboard": primary, "candidate_vs_complete_control": ordered,
        "native_minus_fx_pairs": native_pairs, "both_minus_btc_pairs": both_minus_btc,
        "candidate_vs_additional_controls": extra_comparisons, "stronger_additional_controls": stronger_names,
        "development_expansion_candidates": advancing, "substantial_gain_confirmed": False,
        "independent_confirmation_rounds_consumed": 0}
    write_json(evidence_path, evidence)
    best = ordered[0]
    score = next(r for r in primary if r["model"] == best["candidate"])
    lines = ["# Alternative quote-currency predictive screen", "",
        f"The best new procedure is `{best['candidate']}`: **{100*score['means']['balanced_accuracy']:.6f}% balanced accuracy**, "
        f"a **{best['balanced_accuracy_gain_pp']:+.6f} percentage-point** difference from the retained complete procedure. "
        f"Its natural log-loss ratio is {best['log_loss_ratio']:.9f}. These are three repeatedly exposed development dates, not independent confirmation.", "",
        "All 36 fits and 456 reporting panels completed. Every source retained all 42,420 assessment rows. "
        "All panels were recomputed exactly from saved forecasts; all 96 retained panels and forecast arrays reproduced exactly. "
        "All checkpoint, training-prior and query-independence checks passed. These verification results do not themselves establish a predictive gain.", "",
        "| Procedure | Balanced accuracy | Log loss | Gain vs complete control (pp) |", "| --- | ---: | ---: | ---: |"]
    for row in primary:
        comparison = comparisons.get(row["model"])
        formatted = f"{comparison['balanced_accuracy_gain_pp']:+.6f}" if comparison else "retained control"
        lines.append(f"| `{row['model']}` | {100*row['means']['balanced_accuracy']:.6f}% | {row['means']['log_loss']:.9f} | {formatted} |")
    for title, pairs in (("Native-source minus matched FX-only procedure", native_pairs), ("Joint-source minus matched BTC-only procedure", both_minus_btc)):
        lines.extend(["", title + ":", "", "| Procedure | Mean gain (pp) | BTC gain (pp) | ETH gain (pp) | Log-loss ratio |",
                      "| --- | ---: | ---: | ---: | ---: |"])
        for row in pairs:
            lines.append(f"| `{row['candidate']}` | {row['balanced_accuracy_gain_pp']:+.6f} | {row['gain_pp_by_asset']['BTCUSDT']:+.6f} | "
                         f"{row['gain_pp_by_asset']['ETHUSDT']:+.6f} | {row['log_loss_ratio']:.9f} |")
    lines.extend(["", f"Native procedures passing all frozen development expansion gates: {advancing or 'none'}.", "",
        "A native procedure must gain at least 0.5 points over the complete control and any stronger measured counted-depth or expert reference, "
        "and at least 0.25 points over its exact FX-only counterpart. Both assets must improve and each corresponding log-loss ratio must be at most 1.01. "
        "Passing justifies broader development evaluation. The unchanged substantial-gain requirement still needs five points above the original frozen reference "
        "and its uncertainty, per-asset and twenty-independent-date gates. No independent confirmation round was consumed.", "",
        "Source activity, publisher-time causality and exact software checks do not establish live source arrival, executable fills or economic returns. "
        "All candidate selection here uses exposed development data.", "",
        f"Protocol SHA256: `{sha256_file(protocol_path)}`. Result SHA256: `{sha256_file(summary_path)}`. "
        f"Evidence SHA256: `{sha256_file(evidence_path)}`."])
    report_path.write_text("\n".join(lines) + "\n")
    print(json.dumps({"best_new": best, "native_pairs": len(native_pairs), "expansion_candidates": advancing,
        "summary_sha256": sha256_file(summary_path), "evidence_sha256": sha256_file(evidence_path)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    assess(args.protocol.resolve(), args.output.resolve(), args.evidence.resolve(), args.report.resolve())
