"""Verify the complete alternative-quote feature window without reading targets."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from lob_forge.binance_vision import sha256_file

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ("BTCUSDT", "ETHUSDT")


def assess(protocol_path, manifest_path, evidence_path, report_path):
    if evidence_path.exists() or report_path.exists():
        raise ValueError("Preserve previous feature-preparation assessments")
    protocol = json.loads(protocol_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    verified = {}

    def verify(path, digest=None):
        path = Path(path)
        absolute = path if path.is_absolute() else ROOT / path
        key = str(absolute.relative_to(ROOT))
        actual = sha256_file(absolute)
        if digest is not None and digest != actual:
            raise ValueError(f"Feature-preparation evidence changed: {key}")
        if key in verified and verified[key] != actual:
            raise ValueError("An artifact changed during the feature audit")
        verified[key] = actual
        return actual

    protocol_sha = verify(protocol_path)
    manifest_sha = verify(manifest_path)
    for path, digest in protocol["input_hashes"].items():
        verify(path, digest)
    expected_identity = {"protocol_sha256": protocol_sha, "input_hashes": protocol["input_hashes"]}
    if manifest["identity"] != expected_identity:
        raise ValueError("The complete-window feature identity changed")
    frozen = manifest_path.parent / "frozen_preparation.json"
    verify(frozen)
    if json.loads(frozen.read_text()) != expected_identity:
        raise ValueError("The original feature-preparation freeze changed")

    def no_assessment(record):
        if (not record["complete"] or record["model_fits"] != 0 or record["target_labels_decoded"]
            or record["predictive_metrics_computed"] or record["independent_confirmation_data_opened"]):
            raise ValueError("This report only covers completed source preparation without target assessments")

    no_assessment(manifest)
    days = [(date(2023, 5, 29) + timedelta(days=i)).isoformat() for i in range(14)]
    sessions = manifest["sessions"]
    universe = {(s, d) for s in SYMBOLS for d in days}
    if len(sessions) != 28 or {(r["symbol"], r["date"]) for r in sessions} != universe:
        raise ValueError("All fixed 28 asset/date sessions must occur exactly once")
    day_specs = {r["date"]: r for r in protocol["days"]}
    day_evidence = {r["date"]: r for r in manifest["day_evidence"]}
    if set(day_specs) != set(days) or set(day_evidence) != set(days) or len(manifest["day_evidence"]) != 14:
        raise ValueError("The full fixed source-day universe is required")
    summaries, records = {}, []
    source_totals = {s: 0 for s in ("BTCTUSD", "TUSDUSDT", "ETHBTC", "BTCUSDT")}
    for day in days:
        pair = [r for r in sessions if r["date"] == day]
        if len({r["preparation_summary"] for r in pair}) != 1:
            raise ValueError("Both target assets must share the same original daily preparation")
        summary_path = ROOT / pair[0]["preparation_summary"]
        verify(summary_path, pair[0]["preparation_summary_sha256"])
        summary = json.loads(summary_path.read_text())
        no_assessment(summary)
        summaries[day] = summary
        reused = "reuse_summary" in day_specs[day]
        daily_protocol = (ROOT / "docs/research/boundary_quote_currency_feature_preflight_20260909.json"
                          if reused else manifest_path.parent / "definitions" / f"{day}.json")
        verify(daily_protocol, summary["protocol_sha256"])
        if not reused:
            status_path = manifest_path.parent / "supervision" / f"{day}.json"
            verify(status_path)
            status = json.loads(status_path.read_text())
            if status["returncode"] != 0 or status["failure"] is not None or status["seconds"] > protocol["per_day_wall_seconds"]:
                raise ValueError("Every new daily worker must pass the original frozen execution budget")
            verify(manifest_path.parent / "logs" / f"{day}.log", status["log_sha256"])
        if (day_evidence[day]["reused"] != reused or day_evidence[day]["source_audits"] != summary["source_audits"]
            or day_evidence[day]["memory"] != summary["memory"] or day_evidence[day]["seconds"] != summary["seconds"]):
            raise ValueError("Daily source and resource evidence must agree with the complete-window manifest")
        if summary["original_read_columns"] != ["decision_time", "bid", "ask", "bid_qty", "ask_qty"]:
            raise ValueError("Original source reads must exclude target and execution-outcome fields")
        rss = summary["memory"]["maximum_sampled_rss_bytes"]
        if not 0 < rss <= protocol["daily_settings"]["limits"]["worker_rss_bytes"]:
            raise ValueError("Every data worker requires a working monitor within its frozen resource budget")
        for path, digest in summary["artifact_hashes"].items():
            verify(summary_path.parent / path, digest)
        if set(summary["source_audits"]) != set(source_totals):
            raise ValueError("All four original native and conversion trade sources are required")
        start_ms = int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000)
        for symbol, audit in summary["source_audits"].items():
            if (audit["aggregate_id_gaps"] != 0 or audit["rows"] <= 0
                or not 0 <= audit["first_publisher_time"] - start_ms <= 300000
                or not 0 < start_ms + 86400000 - audit["last_publisher_time"] <= 300000):
                raise ValueError("Every source must retain contiguous aggregate IDs and complete day boundaries")
            source_totals[symbol] += audit["rows"]
        if len(summary["records"]) != 4:
            raise ValueError("Each original daily result must contain four feature groups")
        for session in pair:
            verify(session["original_features_path"], session["original_features_sha256"])
            verify(summary_path, session["preparation_summary_sha256"])
            if set(session["groups"]) != {"100", "500"} or session["reused_verified_training_preflight"] != reused:
                raise ValueError("Both exact delay caches and original reuse lineage must be preserved")
            for delay, group in session["groups"].items():
                original = [r for r in summary["records"] if r["symbol"] == session["symbol"] and r["delay_ms"] == int(delay)]
                if len(original) != 1 or group != {**original[0], "features_path": original[0]["feature_path"]}:
                    raise ValueError("Complete-window groups must exactly reference their original daily records")
                if (group["rows"] != session["decision_rows"] or len(group["columns"]) != 79
                    or len(set(group["columns"])) != 79
                    or not all(group[k] for k in ("original_native_fields_exact", "identity_conversion_replay_exact",
                                                 "truncated_source_prefix_exact", "saved_features_exact"))):
                    raise ValueError("Every feature group must retain all rows and all exact replay checks")
                verify(group["features_path"], group["sha256"])
                records.append({"symbol": session["symbol"], "date": day, "delay_ms": int(delay),
                    "rows": group["rows"], "conversion_available_rows": group["conversion_available_rows"],
                    "converted_basis_available_rows": group["converted_basis_available_rows"],
                    "features_path": group["features_path"], "sha256": group["sha256"]})
    if sum(r["reused"] for r in day_evidence.values()) != 1:
        raise ValueError("Only the exact original June 1 training preflight is reused")
    evidence = {"status": "complete_exposed_development_features_without_model_assessment",
        "protocol": str(protocol_path.relative_to(ROOT)), "protocol_sha256": protocol_sha,
        "feature_manifest": str(manifest_path.relative_to(ROOT)), "feature_manifest_sha256": manifest_sha,
        "pinned_preparation_inputs_verified": len(protocol["input_hashes"]), "verified_artifact_hashes": verified,
        "target_sessions": 28, "feature_groups": len(records), "feature_columns_including_clock": 79,
        "decision_rows": sum(r["decision_rows"] for r in sessions), "source_aggregate_trade_rows": source_totals,
        "seconds": manifest["seconds"], "reused_days": 1,
        "maximum_sampled_worker_rss_bytes": max(r["memory"]["maximum_sampled_rss_bytes"] for r in summaries.values()),
        "minimum_conversion_available_fraction": min(r["conversion_available_rows"] / r["rows"] for r in records),
        "records": records, "model_fits": 0, "target_labels_decoded": False, "predictive_metrics_computed": False,
        "substantial_gain_confirmed": False, "independent_confirmation_data_opened": False}
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    lines = ["# Alternative quote-currency feature preparation", "",
        "All 14 exposed source dates (May 29 through June 11, 2023) passed the fixed source and conversion checks. "
        "The preparation retained 28 target sessions, 56 feature groups at 100/500 ms delays, and "
        f"{evidence['decision_rows']:,} original decision rows. No model was fitted and no predictive score was computed.", "",
        "BTC/TUSD is converted using observed TUSD/USDT trades; ETH/BTC uses observed BTC/USDT trades. "
        "Rates must be strictly available before the decision after the stated delay and no more than 60 seconds old. "
        "Invalid converted basis observations cannot update the basis smoothing state. Native trade flows and returns retain their native-market meaning.", "",
        f"All {len(protocol['input_hashes'])} pinned preparation inputs and each daily artifact inventory were verified. "
        "Each source had contiguous aggregate IDs and its first/last trade within five minutes of the day boundaries. "
        "Every cache passed exact native-field replay, identity-currency conversion, truncated-source prefix and Parquet reload checks. "
        "The original June 1 training preflight was reused exactly; thirteen additional daily workers completed within their fixed resource budgets.", "",
        f"Preparation elapsed time: {manifest['seconds']:.6f} seconds. Maximum sampled worker RSS: "
        f"{evidence['maximum_sampled_worker_rss_bytes']:,} bytes. Lowest conversion availability among the 56 groups: "
        f"{100*evidence['minimum_conversion_available_fraction']:.6f}%.", "",
        "| Native or conversion source | Aggregate-trade records |", "| --- | ---: |"]
    lines.extend(f"| {s} | {n:,} |" for s, n in source_totals.items())
    lines.extend(["", "These are source-integrity results, not predictive gains. Zero-fee trading volume may include uninformative activity. "
        "The next predictive study must compare conversion-only controls with BTC/TUSD and ETH/BTC additions on the same rows. "
        "Historical publisher timestamps do not certify live arrival latency, and last-trade availability does not certify a fresh executable quote.", "",
        "The twenty reserved independent confirmation dates remain unopened for market analysis. "
        "This preparation does not change the substantial-gain requirement or consume a confirmation round.", "",
        f"Feature manifest SHA256: `{manifest_sha}`. Evidence SHA256: `{sha256_file(evidence_path)}`."])
    report_path.write_text("\n".join(lines) + "\n")
    print(json.dumps({k: evidence[k] for k in ("target_sessions", "feature_groups", "decision_rows",
        "source_aggregate_trade_rows", "maximum_sampled_worker_rss_bytes", "minimum_conversion_available_fraction")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    assess(*(p.resolve() for p in (args.protocol, args.manifest, args.evidence, args.report)))
