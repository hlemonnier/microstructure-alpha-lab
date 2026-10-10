"""Source-shaped, synthetic-only quote-renewal CLI E2E; written before its learner.

The scalar endpoint oracle and probability oracle do not import the learner.
Every attempt keeps its sources, command receipts, failures and hash inventory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src/lob_forge/boundary_quote_renewal.py"
DESIGN = ROOT / "docs/research/boundary_quote_renewal_design_20261011.md"
COLUMNS = ["update_id", "best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty", "transaction_time", "event_time"]
THREAD_VARS = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]
FAILURES = [
    "Same-time feature-last or endpoint-first selection is changed.",
    "A quote published exactly at decision time leaks into features.",
    "The exit moves from d+5100 to realized entry time+5000.",
    "Initial age is omitted or survival G>=a has an off-by-one exponent.",
    "One block spanning both stops becomes directional.",
    "Multi-block reversals or entry-spread randomness are omitted.",
    "Exact-decimal ties/off-grid prices are rounded or overflow.",
    "IDs/timestamps regress, duplicate, or quoted prices/quantities are invalid; locked quotes fail the whole attempt.",
    "Selected source-window start/end creates an extra transition or decodes a quote at exclusive end.",
    "File boundaries become transitions or censor survival is off by one.",
    "Sparse states drop rows instead of disclosed pooled backoff.",
    "Regularized joint EM objective decreases or probabilities lose mass.",
    "Simulation truncates dwell tails, event counts, jumps, or invalid-price mass.",
    "Reflection is omitted, applied after an endpoint, or loses path mass; initial bid violates its floor.",
    "Future perturbation, batching, serialization, or query order changes forecasts.",
    "Finite path bias or resource violations are hidden.",
]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def quote(t, uid, bid="30000", ask="30000.2", bq="7", aq="5"):
    return dict(zip(COLUMNS, [str(uid), str(bid), str(bq), str(ask), str(aq), str(t - 1), str(t)]))


def archive(path, rows):
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    info = zipfile.ZipInfo(path.stem + ".csv", date_time=(2026, 10, 11, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr(info, text.getvalue())
    return path


def decimal(value):
    # Canonical observation semantics are shortest decimal of the parsed float.
    return Fraction(str(float(value)))


def oracle(rows, decision, minimum_tick="0.1"):
    previous = [r for r in rows if int(r["event_time"]) < decision]
    entry = next((r for r in rows if int(r["event_time"]) >= decision + 100), None)
    future = next((r for r in rows if int(r["event_time"]) >= decision + 5100), None)
    require(previous and entry is not None and future is not None, "Oracle tape must resolve all three selections")
    known = previous[-1]
    eb, ea = decimal(entry["best_bid_price"]), decimal(entry["best_ask_price"])
    fb, fa = decimal(future["best_bid_price"]), decimal(future["best_ask_price"])
    delta = fb + fa - eb - ea
    threshold = max(ea - eb, 2 * decimal(minimum_tick))
    return {
        "decision_time": decision,
        "feature_update_id": int(known["update_id"]),
        "feature_event_time": int(known["event_time"]),
        "entry_update_id": int(entry["update_id"]),
        "entry_event_time": int(entry["event_time"]),
        "exit_update_id": int(future["update_id"]),
        "exit_event_time": int(future["event_time"]),
        "delta_exact": str(delta),
        "threshold_exact": str(threshold),
        "label": 1 if delta > threshold else -1 if delta < -threshold else 0,
    }


def source_record(path, rows, end):
    return {"path": str(path), "sha256": digest(path), "coverage_end_exclusive": end,
            "expected_blocks": len({int(r["event_time"]) for r in rows})}


def manifest(path, sources, tick="0.1"):
    dump(path, {"data_kind": "synthetic", "asset": "BTCUSDT", "minimum_tick": tick, "sources": sources})
    return path


def reflection_oracles(out):
    # Direct S <- max(S+increment,spread+2) is independent of the C/R cache.
    cases = {
        "inactive_tie": (Fraction(1000), 2, [(100, 0, 2, 3, 2), (5100, 1, 2, 50, 4)]),
        "pre_entry_push": (Fraction(10), 2, [(20, -20, 2, -21, 2), (100, 5, 2, 6, 2), (5100, 0, 2, 1, 2)]),
        "post_entry_push": (Fraction(10), 2, [(100, 0, 2, 0, 2), (200, -20, 2, -22, 2), (5100, 4, 2, 5, 2)]),
        "varying_spread": (Fraction(10), 2, [(100, 0, 2, 0, 2), (200, -7, 8, -3, 4), (5100, 0, 2, 30, 2)]),
        "same_stop": (Fraction(10), 2, [(7000, -80, 4, 0, 2)]),
        "off_grid_tie": (Fraction(7, 2), 1, [(100, -1, 1, -1, 1), (5100, 4, 1, 20, 2)]),
    }
    inputs, expected = [], []
    for name, (origin, known_spread, values) in cases.items():
        require((origin-known_spread)/2 >= 1, "Known synthetic bid must satisfy floor")
        unreflected, regulator, current = 0, 0, origin
        entry, future, ledger = None, None, []
        blocks = []
        for timestamp, first_delta, first_spread, last_delta, last_spread in values:
            blocks.append({"time": timestamp, "first_delta": first_delta, "first_spread": first_spread,
                           "last_delta": last_delta, "last_spread": last_spread, "next_state": 0})
            for kind, increment, spread in (("first", first_delta, first_spread),
                                             ("last", last_delta-first_delta, last_spread)):
                current = max(current + increment, Fraction(spread+2))
                unreflected += increment
                regulator = max(regulator, spread+2-unreflected)
                reconstructed = origin+unreflected+max(Fraction(0), Fraction(regulator)-origin)
                require(current == reconstructed and (current-spread)/2 >= 1, "Independent reflection forms differ")
                ledger.append({"time": timestamp, "kind": kind, "sum_units": str(current),
                               "spread_units": spread, "c": unreflected, "r": regulator})
                if kind == "first" and entry is None and timestamp >= 100:
                    entry = (current, spread, unreflected, regulator)
                if kind == "first" and timestamp >= 5100:
                    future = (current, spread, unreflected, regulator)
                    break
            if future is not None:
                break
        require(entry is not None and future is not None, "Reflection oracle unresolved")
        delta = future[0]-entry[0]
        threshold = max(entry[1], 4)
        label = 1 if delta > threshold else -1 if delta < -threshold else 0
        expected.append({"case": name, "origin_sum_units": str(origin), "entry_c": entry[2],
                         "exit_c": future[2], "entry_r": entry[3], "exit_r": future[3],
                         "entry_spread": entry[1], "exit_spread": future[1], "delta_exact": str(delta),
                         "threshold_units": threshold, "label": label, "ledger": ledger})
        inputs.append({"case": name, "origin_sum_units": str(origin), "initial_spread_units": known_spread,
                       "blocks": blocks})
    require(expected[-1]["delta_exact"] == "4" and expected[-1]["label"] == 0, "Off-grid tie oracle changed")
    require(expected[4]["entry_c"] == expected[4]["exit_c"] and expected[4]["label"] == 0, "Same-stop reflection differs")
    dump(out / "reflection_oracle.json", expected)
    dump(out / "fixtures/reflection_inputs.json", {"data_kind": "synthetic", "minimum_tick_units": 2,
                                                 "entry_delay_ms": 100, "exit_delay_ms": 5100, "paths": inputs})


def prepare(out):
    fixtures = out / "fixtures"
    fixtures.mkdir()
    dump(out / "failure_contract.json", {"written_before_implementation": not MODULE.exists(),
        "initial_failure_contract_precedes_implementation": True,
        "pre_code_contract_artifact": "artifacts/quote-renewal-e2e-20261010T233927157597Z/evidence.json",
        "failures": FAILURES})
    reflection_oracles(out)
    cases = {
        "ties": [quote(900, 1), quote(900, 2, bq="10"), quote(1000, 3, "31000", "31000.2"),
                 quote(1100, 4), quote(1100, 5, "30000.5", "30000.7"),
                 quote(6100, 6, "30000.1", "30000.3"), quote(6100, 7, "30001", "30001.2")],
        "off_grid_above_tie": [quote(900, 1), quote(1100, 2),
                               quote(6100, 3, "30000.1", "30000.300000000003")],
        "reversal": [quote(900, 1), quote(1100, 2), quote(1300, 3, "30000.8", "30001"),
                     quote(3100, 4, "29999.4", "29999.6"), quote(6100, 5)],
        "random_entry_spread": [quote(900, 1), quote(1100, 2, "29999.8", "30000.6"),
                                quote(6100, 3, "30000.3", "30000.5")],
        "delayed_entry_fixed_exit": [quote(900, 1), quote(1500, 2),
                                     quote(6100, 3, "30000.3", "30000.5"),
                                     quote(6500, 4, "29999", "29999.2")],
        "both_stops_one_block": [quote(900, 1), quote(7000, 2, "30001", "30001.2"),
                                 quote(7000, 3, "30002", "30002.2")],
        "zero_depth": [quote(900, 1, bq="0", aq="0"), quote(1100, 2), quote(6100, 3)],
    }
    eval_sources, endpoint_rows, queries = [], [], []
    for name, rows in cases.items():
        path = archive(fixtures / (name + ".zip"), rows)
        eval_sources.append(source_record(path, rows, int(rows[-1]["event_time"]) + 101))
        result = oracle(rows, 1000)
        result.update({"case": name, "archive": str(path)})
        endpoint_rows.append(result)
        queries.append({"archive": str(path), "decision_time": 1000})
    expected_labels = {"ties": 0, "off_grid_above_tie": 1, "reversal": 0,
                       "random_entry_spread": 0, "delayed_entry_fixed_exit": 1,
                       "both_stops_one_block": 0, "zero_depth": 0}
    require({r["case"]: r["label"] for r in endpoint_rows} == expected_labels, "Independent fixture labels changed")
    require(endpoint_rows[0]["feature_update_id"] == 2 and endpoint_rows[0]["entry_update_id"] == 4
            and endpoint_rows[0]["exit_update_id"] == 6, "Fixture first/last ledger is wrong")
    dump(out / "oracle_ledger.json", endpoint_rows)
    with (out / "oracle_ledger.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(endpoint_rows[0]))
        writer.writeheader()
        writer.writerows(endpoint_rows)
    dump(fixtures / "queries.json", {"asset": "BTCUSDT", "queries": queries})
    manifest(fixtures / "inspection_manifest.json", eval_sources)
    train_sources = []
    for session in range(4):
        rows, timestamp, level, uid = [], 100000 + session * 200000, 30000, 1
        for block in range(144):
            if block:
                timestamp += [15, 70, 100, 260, 400, 620, 1500][(block + session) % 7]
            level += [0, 0.1, -0.1, 0.2, -0.2, 0][block % 6]
            spread = [0.2, 0.4, 0.6, 0.2][block % 4]
            rows.append(quote(timestamp, uid, f"{level:.2f}", f"{level + spread:.2f}",
                              1 + block % 9, 1 + (block * 3) % 11))
            uid += 1
            rows.append(quote(timestamp, uid, f"{level + 0.05:.2f}", f"{level + 0.05 + spread:.2f}",
                              1 + (block * 2) % 13, 1 + (block * 5) % 7))
            uid += 1
        path = archive(fixtures / f"train_{session}.zip", rows)
        train_sources.append(source_record(path, rows, timestamp + 37 + session))
    manifest(fixtures / "train_manifest.json", train_sources)
    age_rows = [quote(0, 1), quote(0, 2, bq="9", aq="1"), quote(9000, 3)]
    age_archive = archive(fixtures / "age.zip", age_rows)
    age_queries = [{"archive": str(age_archive), "decision_time": value} for value in (1, 2, 101, 1001)]
    dump(fixtures / "age_queries.json", {"asset": "BTCUSDT", "queries": age_queries})
    changed = [dict(r) for r in age_rows]
    changed[-1].update(best_bid_price="45000", best_ask_price="45001", best_bid_qty="100")
    changed_archive = archive(fixtures / "future_changed.zip", changed)
    dump(fixtures / "future_queries.json", {"asset": "BTCUSDT", "queries": [
        {"archive": str(changed_archive), "decision_time": r["decision_time"]} for r in age_queries]})
    dump(fixtures / "partition_a.json", {"asset": "BTCUSDT", "queries": age_queries[:2]})
    dump(fixtures / "partition_b.json", {"asset": "BTCUSDT", "queries": age_queries[2:]})
    dump(fixtures / "reversed_queries.json", {"asset": "BTCUSDT", "queries": age_queries[::-1]})
    at_decision = archive(fixtures / "at_decision.zip", [age_rows[0], age_rows[1],
        quote(1, 3, "45000", "45001", bq="100", aq="1"), quote(9000, 4)])
    dump(fixtures / "at_decision_queries.json", {"asset": "BTCUSDT", "queries": [
        {"archive": str(at_decision), "decision_time": 1}]})
    low_archive = archive(fixtures / "low_origin.zip", [quote(0, 1, "0.1", "0.3", bq="9", aq="1"), quote(9000, 2)])
    dump(fixtures / "low_queries.json", {"asset": "BTCUSDT", "queries": [{"archive": str(low_archive), "decision_time": 1}]})
    fractional_archive = archive(fixtures / "fractional_origin.zip", [quote(0, 1, "0.101", "0.301", bq="9", aq="1"), quote(9000, 2)])
    dump(fixtures / "fractional_queries.json", {"asset": "BTCUSDT", "queries": [{"archive": str(fractional_archive), "decision_time": 1}]})
    at_floor = archive(fixtures / "at_floor_origin.zip", [quote(0, 1, "0.025", "0.225", bq="9", aq="1"), quote(9000, 2)])
    dump(fixtures / "at_floor_queries.json", {"asset": "BTCUSDT", "queries": [{"archive": str(at_floor), "decision_time": 1}]})
    below_floor = archive(fixtures / "below_floor.zip", [quote(0, 1, "0.0001", "0.2001"), quote(9000, 2)])
    dump(fixtures / "below_floor_queries.json", {"asset": "BTCUSDT", "queries": [{"archive": str(below_floor), "decision_time": 1}]})
    selected_rows = [quote(99, 1, "20000", "20000.2"), quote(100, 2), quote(100, 3, "30000.5", "30000.7"),
                     quote(200, 4, "30000.1", "30000.3"), quote(300, 5, "nan", "30000.2"), quote(1000, 6)]
    selected_archive = archive(fixtures / "selected_window.zip", selected_rows)
    selected = source_record(selected_archive, selected_rows, 300)
    selected.update(window_start_inclusive=100, expected_blocks=2)
    manifest(fixtures / "selected_window_manifest.json", [selected])
    malformed = {
        "duplicate_id": [quote(0, 1), quote(100, 1)],
        "regressing_timestamp": [quote(100, 1), quote(99, 2)],
        "crossed_price": [quote(0, 1), quote(100, 2, "30001", "30000")],
        "locked_price": [quote(0, 1), quote(100, 2, "30000", "30000")],
        "nonfinite_price": [quote(0, 1), quote(100, 2, "nan", "30000.2")],
        "negative_quantity": [quote(0, 1), quote(100, 2, bq="-1")],
    }
    for name, rows in malformed.items():
        path = archive(fixtures / (name + ".zip"), rows)
        manifest(fixtures / (name + "_manifest.json"), [source_record(path, rows, 1000)])
    invalid_end = dict(train_sources[0])
    invalid_end["coverage_end_exclusive"] -= 10000
    manifest(fixtures / "invalid_boundary_manifest.json", [invalid_end])
    corrupted = dict(train_sources[0])
    corrupted["sha256"] = "0" * 64
    manifest(fixtures / "invalid_hash_manifest.json", [corrupted])
    giant = [quote(0, 1, "1e120", "2e120"), quote(100, 2, "2e120", "3e120")]
    giant_path = archive(fixtures / "giant.zip", giant)
    manifest(fixtures / "giant_manifest.json", [source_record(giant_path, giant, 200)])
    # Hand-constructed homogeneous kernel: an extra block moves 3 integer sum
    # units, strictly beyond its 2-unit barrier. Neutral iff no block occurs
    # in [d+100,d+5100), giving exactly (1-p)^5000, including endpoint equality.
    p = 0.0002
    small_model = {
        "format_version": "quote_block_reflected_renewal_v1_synthetic", "data_kind": "synthetic",
        "asset": "BTCUSDT", "scale": 20, "minimum_tick_units": 2,
        "floor": {"numerator": 1, "denominator": 20},
        "spread_edges_units": [], "entry_delay_ms": 100, "exit_delay_ms": 5100,
        "seed": 913, "simulation_paths": 4096,
        "state_kernel_map": {str(i): "pooled" for i in range(32)},
        "kernels": {"pooled": {"weights": [0.4, 0.6], "p": [p, p],
            "marks": [{"first_delta": 6, "first_spread": 4, "last_delta": 6,
                       "last_spread": 4, "next_state": 0}], "emissions": [[1.0], [1.0]],
            "objective_trace": [], "complete_transitions": 0, "censored_intervals": 0}},
        "cache": {}, "fit_summary": {"hand_constructed_oracle": True},
        "assumptions": ["Synthetic probability oracle; not a fitted market model."],
    }
    dump(fixtures / "probability_oracle_model.json", small_model)
    underflow_model = json.loads(json.dumps(small_model))
    underflow_model["kernels"]["pooled"]["p"] = [0.5, p]
    underflow_model["simulation_paths"] = 64
    dump(fixtures / "survival_underflow_model.json", underflow_model)
    stale = archive(fixtures / "stale_prefix.zip", [quote(0, 1, bq="9", aq="1")])
    dump(fixtures / "survival_underflow_queries.json", {"asset": "BTCUSDT", "queries": [
        {"archive": str(stale), "decision_time": 1}, {"archive": str(stale), "decision_time": 10001}]})
    wrong_state_model = json.loads(json.dumps(small_model))
    wrong_state_model["spread_edges_units"] = [3]
    dump(fixtures / "wrong_mark_state_model.json", wrong_state_model)
    dump(out / "probability_oracle.json", {"down": 0.0, "neutral": (1 - p) ** 5000,
        "up": 1 - (1 - p) ** 5000, "formula": "neutral=(1-p)^5000", "p": p,
        "fixed_paths": 4096, "absolute_tolerance": 0.04})
    return fixtures


def prediction_values(report):
    return [r["natural_probabilities"] for r in report["predictions"]]


def run(out, fixtures, receipt):
    attempts = out / "attempts"
    attempts.mkdir()
    env = os.environ.copy()
    env.update({name: "2" for name in THREAD_VARS})
    env["PYTHONPATH"] = str(ROOT / "src")

    def cli(name, args, rejection=False):
        command = [sys.executable, "-m", "lob_forge.boundary_quote_renewal", *map(str, args)]
        started = time.monotonic()
        timed_out = False
        try:
            completed = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=120)
            code, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as error:
            timed_out, code = True, None
            stdout, stderr = error.stdout or "", error.stderr or ""
            stdout = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
            stderr = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
        (attempts / (name + ".stdout.txt")).write_text(stdout)
        (attempts / (name + ".stderr.txt")).write_text(stderr)
        record = {"name": name, "command": command, "exit_code": code, "timed_out": timed_out,
                  "wall_seconds": time.monotonic() - started, "expected_rejection": rejection,
                  "thread_environment": {key: env[key] for key in THREAD_VARS}}
        dump(attempts / (name + ".json"), record)
        receipt["commands"].append(record)
        require(not timed_out, f"CLI {name} timed out; partial stdout/stderr and command receipt are preserved")
        require((code != 0) if rejection else (code == 0), f"CLI {name} returned {code}: {stderr[-2000:]}")
        receipt["passed_steps"].append(name)

    cli("replay_reflection_oracle", ["replay-paths", "--input", fixtures / "reflection_inputs.json",
        "--output", out / "reflection_actual.json"])
    reflected = json.loads((out / "reflection_actual.json").read_text())["paths"]
    expected_reflection = json.loads((out / "reflection_oracle.json").read_text())
    require(reflected == expected_reflection, "Direct sequential reflection versus cache formula differs")
    receipt["passed_steps"].append("independent_reflection_ledger_exact")
    cli("selected_source_window", ["inspect", "--manifest", fixtures / "selected_window_manifest.json",
        "--output", out / "selected_window.json"])
    selected = json.loads((out / "selected_window.json").read_text())["sessions"][0]
    require(selected["complete_transitions"] == 1 and selected["censor_duration_at_least"] == 100,
            "Selected window fabricated a transition or censor")
    require([(b["first_id"], b["last_id"]) for b in selected["blocks"]] == [(2, 3), (4, 4)],
            "Window start or exclusive end admitted a quote")
    receipt["passed_steps"].append("inclusive_start_exclusive_end_exact")
    cli("inspect_exact_endpoints", ["inspect", "--manifest", fixtures / "inspection_manifest.json",
        "--queries", fixtures / "queries.json", "--output", out / "inspected.json"])
    inspected = json.loads((out / "inspected.json").read_text())
    expected = json.loads((out / "oracle_ledger.json").read_text())
    for actual, reference in zip(inspected["endpoint_ledger"], expected):
        require(all(actual[key] == reference[key] for key in reference if key != "case"), "Endpoint ledger mismatch")
    require(len(inspected["endpoint_ledger"]) == len(expected), "Endpoint count mismatch")
    receipt["passed_steps"].append("independent_endpoint_ledger_exact")
    for name in ("duplicate_id", "regressing_timestamp", "crossed_price", "locked_price", "nonfinite_price", "negative_quantity",
                 "invalid_boundary", "invalid_hash"):
        cli("reject_" + name, ["inspect", "--manifest", fixtures / (name + "_manifest.json"),
                              "--output", out / (name + ".json")], rejection=True)
    model_path = out / "model.json"
    cli("fit_joint_em_and_cache", ["fit", "--manifest", fixtures / "train_manifest.json", "--model", model_path,
        "--iterations", 30, "--min-state-transitions", 12, "--paths", 256, "--seed", 914])
    model = json.loads(model_path.read_text())
    require(model["fit_summary"]["complete_transitions"] == 4 * 143, "Cross-file transition fabricated")
    require(model["fit_summary"]["censored_intervals"] == 4, "File censor count mismatch")
    require(model["fit_summary"]["censor_duration_at_least"] == [37, 38, 39, 40], "Censor boundary off by one")
    require(model["fit_summary"]["backoff_states"], "Synthetic sparse-state backoff not exercised")
    for kernel in model["kernels"].values():
        trace = kernel["objective_trace"]
        require(all(b >= a - 1e-7 for a, b in zip(trace, trace[1:])), "Regularized EM objective decreased")
        require(abs(sum(kernel["weights"]) - 1) < 1e-12, "Mixture weight mass lost")
        require(all(0 < value < 1 for value in kernel["p"]), "Illegal duration probability")
        require(all(abs(sum(row) - 1) < 1e-12 for row in kernel["emissions"]), "Emission mass lost")
    receipt["passed_steps"].append("objective_censor_and_backoff_receipts")
    for name, queries in [("regular", "queries.json"), ("age", "age_queries.json"),
                          ("age_repeat", "age_queries.json"), ("future", "future_queries.json"),
                          ("partition_a", "partition_a.json"), ("partition_b", "partition_b.json"),
                          ("reversed", "reversed_queries.json")]:
        cli("predict_" + name, ["predict", "--model", model_path, "--queries", fixtures / queries,
                               "--output", out / (name + ".json")])
    reports = {name: json.loads((out / (name + ".json")).read_text()) for name in
               ("regular", "age", "age_repeat", "future", "partition_a", "partition_b", "reversed")}
    values = prediction_values(reports["age"])
    require(values == prediction_values(reports["age_repeat"]), "Reload forecast changed")
    require(values == prediction_values(reports["future"]), "Future data changed a forecast")
    require(values == prediction_values(reports["partition_a"]) + prediction_values(reports["partition_b"]), "Query partition changed")
    require(values == prediction_values(reports["reversed"])[::-1], "Query order changed")
    cli("predict_exact_decision_excluded", ["predict", "--model", model_path,
        "--queries", fixtures / "at_decision_queries.json", "--output", out / "at_decision.json"])
    at_decision = json.loads((out / "at_decision.json").read_text())
    require(prediction_values(at_decision) == values[:1], "Quote exactly at decision changed forecast")
    require(at_decision["predictions"][0]["feature_update_id"] == 2
            and reports["age"]["predictions"][0]["feature_update_id"] == 2,
            "Prediction features did not exclude the quote at decision")
    receipt["passed_steps"].append("prediction_quote_at_decision_excluded")
    for report in reports.values():
        for row in report["predictions"]:
            require(all(math.isfinite(p) and p >= 0 for p in row["natural_probabilities"])
                    and abs(sum(row["natural_probabilities"]) - 1) < 1e-12, "Invalid natural posterior")
            require(row["positive_support_certified"] and row["physical_violation_mass"] == 0, "Physical-price support uncertified")
    for row in reports["age"]["predictions"]:
        kernel = model["kernels"][row["kernel"]]
        cache = model["cache"][row["kernel"]]
        logs = [math.log(w) + (row["initial_age_ms"] - 1) * math.log1p(-p)
                for w, p in zip(kernel["weights"], kernel["p"])]
        unnormalized = [math.exp(log - max(logs)) for log in logs]
        weights = [w / sum(unnormalized) for w in unnormalized]
        posterior = [sum(weights[r] * cache["component_probabilities"][r][c] for r in range(2)) for c in range(3)]
        require(max(abs(a - b) for a, b in zip(posterior, row["natural_probabilities"])) < 2e-15, "Initial-age survival law is wrong")
    receipt["passed_steps"].append("save_load_prefix_partition_order_and_age_exact")
    for name in ("low", "fractional", "at_floor"):
        cli("predict_reflected_"+name, ["predict", "--model", model_path, "--queries", fixtures / (name+"_queries.json"),
            "--output", out / (name+"_reflected.json")])
        low = json.loads((out / (name+"_reflected.json")).read_text())
        require(all(row["positive_support_certified"] and row["physical_violation_mass"] == 0
                    and abs(sum(row["natural_probabilities"])-1) < 1e-12 for row in low["predictions"]),
                "Reflected path mass or positivity lost")
    require(json.loads((out / "low_reflected.json").read_text())["predictions"][0]["reflected_fraction"] > 0,
            "Synthetic boundary reflection was not exercised")
    cli("reject_initial_bid_floor", ["predict", "--model", model_path, "--queries", fixtures / "below_floor_queries.json",
        "--output", out / "below_floor_failure.json"], rejection=True)
    low = json.loads((out / "below_floor_failure.json").read_text())
    require(low["status"] == "failed" and "floor" in low["error"].lower(), "Floor rejection not preserved")
    receipt["passed_steps"].append("reflection_full_mass_off_grid_and_initial_floor")
    cli("reject_survival_underflow", ["predict", "--model", fixtures / "survival_underflow_model.json",
        "--queries", fixtures / "survival_underflow_queries.json", "--output", out / "survival_underflow_failure.json"], rejection=True)
    underflow = json.loads((out / "survival_underflow_failure.json").read_text())
    require(underflow["status"] == "failed" and "log_weights=" in underflow["error"]
            and "predictions" not in underflow, "Survival underflow lost its detail or retained a partial query batch")
    cli("reject_joint_mark_state_mismatch", ["predict", "--model", fixtures / "wrong_mark_state_model.json",
        "--queries", fixtures / "age_queries.json", "--output", out / "wrong_mark_state_failure.json"], rejection=True)
    wrong_state = json.loads((out / "wrong_mark_state_failure.json").read_text())
    require(wrong_state["status"] == "failed" and "last-spread" in wrong_state["error"], "Invalid next-state bin not rejected")
    receipt["passed_steps"].append("numerical_survival_limit_and_exact_mark_state_contract")
    cli("reject_simulation_overflow", ["fit", "--manifest", fixtures / "giant_manifest.json",
        "--model", out / "overflow_failure.json", "--iterations", 3, "--paths", 64], rejection=True)
    cli("independent_probability_oracle", ["predict", "--model", fixtures / "probability_oracle_model.json",
        "--queries", fixtures / "age_queries.json", "--output", out / "oracle_predictions.json"])
    predictions = json.loads((out / "oracle_predictions.json").read_text())
    known = json.loads((out / "probability_oracle.json").read_text())
    for row in predictions["predictions"]:
        require(row["natural_probabilities"][0] == 0, "Positive-jump oracle generated down mass")
        require(abs(row["natural_probabilities"][1] - known["neutral"]) < known["absolute_tolerance"], "Analytic geometric oracle differs")
    receipt["passed_steps"].append("independent_geometric_neutral_probability")
    cli("independent_seed", ["predict", "--model", model_path, "--queries", fixtures / "age_queries.json",
        "--output", out / "independent_seed.json", "--seed", 915, "--paths", 256])
    cli("doubled_paths", ["predict", "--model", model_path, "--queries", fixtures / "age_queries.json",
        "--output", out / "doubled_paths.json", "--seed", 914, "--paths", 512])
    comparisons = {}
    for name in ("independent_seed", "doubled_paths"):
        other = prediction_values(json.loads((out / (name + ".json")).read_text()))
        comparisons[name] = {"maximum_probability_difference": max(abs(a - b) for x, y in zip(values, other) for a, b in zip(x, y)),
                             "changed_argmax_decisions": sum(max(range(3), key=x.__getitem__) != max(range(3), key=y.__getitem__)
                                                             for x, y in zip(values, other))}
    dump(out / "simulation_convergence.json", {"prototype_paths": 256, "comparisons": comparisons,
         "market_preflight_passed": False, "interpretation": "Measured finite simulation variation; no market tolerance or accuracy claim."})
    receipt["passed_steps"].append("finite_simulation_variation_reported")
    original_model_hash = digest(model_path)
    cli("reject_overwrite", ["fit", "--manifest", fixtures / "train_manifest.json", "--model", model_path,
        "--iterations", 2, "--paths", 64], rejection=True)
    require(digest(model_path) == original_model_hash, "Rejected overwrite changed the original model")
    receipt["passed_steps"].append("rejected_overwrite_preserves_model_hash")
    lint_command = [str(Path(sys.executable).parent / "ruff"), "check", str(MODULE), str(Path(__file__).resolve())]
    lint = subprocess.run(lint_command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=60)
    (attempts / "scoped_ruff.stdout.txt").write_text(lint.stdout)
    (attempts / "scoped_ruff.stderr.txt").write_text(lint.stderr)
    dump(attempts / "scoped_ruff.json", {"command": lint_command, "exit_code": lint.returncode})
    require(lint.returncode == 0, "Scoped Ruff failed: " + lint.stdout + lint.stderr)
    receipt["passed_steps"].append("scoped_ruff")
    receipt["maximum_probability_variation"] = max(r["maximum_probability_difference"] for r in comparisons.values())
    receipt["synthetic_only"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "full"), default="full")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = (args.output or ROOT / "artifacts" / ("quote-renewal-e2e-" + timestamp)).resolve()
    require(not out.exists(), "Preserve old attempts: E2E output directory already exists")
    require(out.parent == ROOT / "artifacts" and out.name.startswith("quote-renewal-e2e-"), "Output must remain in owned artifact scope")
    out.mkdir(parents=True)
    started = time.monotonic()
    receipt = {"status": "running", "started_at_utc": timestamp, "stage": args.stage,
               "commands": [], "passed_steps": [], "python": sys.executable, "platform": platform.platform(),
               "implementation_existed_before_preparation": MODULE.exists(), "synthetic_only": True}
    try:
        fixtures = prepare(out)
        receipt["passed_steps"].append("source_zip_and_independent_oracles_prepared")
        if args.stage == "full":
            require(MODULE.exists(), "Learner missing; pre-code fixtures are preserved")
            run(out, fixtures, receipt)
            receipt["status"] = "passed"
        else:
            receipt["status"] = "prepared_after_implementation" if MODULE.exists() else "prepared_before_implementation"
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
    finally:
        snapshots = out / "source"
        snapshots.mkdir()
        for source in (Path(__file__).resolve(), MODULE, DESIGN):
            if source.exists():
                (snapshots / source.name).write_bytes(source.read_bytes())
        receipt["wall_seconds"] = time.monotonic() - started
        receipt["files"] = [{"path": str(path.relative_to(out)), "bytes": path.stat().st_size, "sha256": digest(path)}
                            for path in sorted(out.rglob("*")) if path.is_file() and path.name != "evidence.json"]
        dump(out / "evidence.json", receipt)
        print(json.dumps({"status": receipt["status"], "artifact": str(out / "evidence.json"),
                          "passed_steps": len(receipt["passed_steps"]), "error": receipt.get("error")}))
    return 1 if receipt["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
