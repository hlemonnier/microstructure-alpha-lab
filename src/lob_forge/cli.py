from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from lob_forge.alpha_factory import (
    AcceptanceCriteria,
    audit_result_artifact,
    correct_p_values,
    evaluate_acceptance,
    format_p_value_corrections,
    format_result_audit_csv,
    format_result_audit_markdown,
    read_p_value_records,
)
from lob_forge.baselines import (
    compute_metrics,
    evaluate_rule_file,
    format_baseline_results,
    format_calendar_walk_forward_results,
    format_conditional_walk_forward_results,
    format_fee_sweep,
    format_regime_analysis,
    format_rule_evaluation,
    format_walk_forward_results,
    run_calendar_walk_forward_thresholds,
    run_conditional_walk_forward_thresholds,
    run_regime_analysis,
    run_fee_sweep,
    run_threshold_baselines,
    run_walk_forward_thresholds,
    predict_feature_threshold,
)
from lob_forge.capacity import format_capacity_diagnostics, run_capacity_diagnostics
from lob_forge.archive import sample_zip_csv
from lob_forge.binance_vision import (
    archive_key,
    dataset_prefix,
    download_archive,
    infer_date_from_key,
    iter_dates,
    list_objects,
)
from lob_forge.edge_model import (
    format_edge_walk_forward_results,
    run_edge_walk_forward,
    run_edge_walk_forward_streaming,
    write_edge_shadow_decisions_streaming,
)
from lob_forge.evidence_gates import (
    evaluate_remaining_evidence_gates,
    format_evidence_gate_report,
    write_evidence_gate_report,
)
from lob_forge.experiments import build_daily_feature_range
from lob_forge.features import build_quote_trade_dataset, summarize_feature_csv
from lob_forge.fill_diagnostics import (
    format_fill_diagnostics,
    format_fill_regime_diagnostics,
    run_fill_diagnostics,
    run_fill_regime_diagnostics,
)
from lob_forge.holdout import (
    build_holdout_manifest,
    canonical_json_sha256,
    holdout_manifest_source_root,
    read_holdout_manifest,
    read_holdout_rows,
    verify_holdout_manifest_file,
    write_development_csv,
    write_final_holdout_result,
    write_holdout_manifest,
)
from lob_forge.data_sources import format_schema_validation, validate_bybit_orderbook_data_zip, validate_l2_csv_schema
from lob_forge.l2_ingest import (
    build_historical_l2_manifest,
    download_historical_l2_manifest,
    import_historical_l2_file,
    import_historical_l2_manifest,
    resolve_bybit_historical_l2_manifest,
    resolve_okx_historical_l2_manifest,
    write_historical_l2_manifest,
)
from lob_forge.live_collectors import capture_live_l2
from lob_forge.live_validation import (
    SUPPORTED_OBSERVED_FILL_PROVIDERS,
    format_observed_fill_merge_report,
    format_observed_fill_normalization_report,
    format_shadow_fill_validation_report,
    merge_observed_fills_into_shadow_decisions,
    normalize_observed_fills,
    simulate_shadow_fills_to_file,
    validate_shadow_fill_predictions,
    write_observed_fill_template,
    write_market_events_from_feature_csv,
)
from lob_forge.observed_fill_fetch import (
    SUPPORTED_OBSERVED_FILL_FETCH_PROVIDERS,
    fetch_observed_fill_export,
    format_observed_fill_fetch_report,
)
from lob_forge.local_api_sources import (
    LOCAL_EVIDENCE_GATES,
    PAPER_FILL_GAP,
    QUOTE_TRADE_GAP,
    TRUE_L2_SMOKE_GAP,
    format_local_api_sources_csv,
    format_local_api_sources_json,
    format_local_api_sources_markdown,
    list_local_api_sources,
)
from lob_forge.logistic import format_logistic_walk_forward_results, run_logistic_walk_forward
from lob_forge.memory_guard import apply_process_memory_limit, assert_csv_load_budget
from lob_forge.ml_models import (
    evaluate_model_readiness,
    format_l2_masked_pretraining_report,
    format_l2_sequence_experiment_report,
    format_model_readiness_report,
    run_l2_masked_pretraining_smoke,
    run_l2_torch_sequence_experiment,
)
from lob_forge.portfolio import evaluate_oos_variance_stability, format_variance_stability_report
from lob_forge.execution_sim import (
    LatencyAssumptions,
    MarketEvent,
    OrderConstraints,
    QueueAssumptions,
    SignalEvent,
    StatefulExecutionConfig,
    simulate_stateful_execution,
)


def main(argv: list[str] | None = None) -> int:
    apply_process_memory_limit()

    parser = argparse.ArgumentParser(prog="lob-forge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("datasets", help="Print the v1 Binance Vision dataset scope.")

    list_parser = subparsers.add_parser("list", help="List Binance Vision archives.")
    _add_archive_args(list_parser, require_date=False)
    list_parser.add_argument("--limit", type=int, default=20)

    download_parser = subparsers.add_parser("download", help="Download one archive.")
    _add_archive_args(download_parser, require_date=True)
    download_parser.add_argument("--root", default="data/raw")
    download_parser.add_argument("--overwrite", action="store_true")
    download_parser.add_argument("--no-verify", action="store_true")

    range_parser = subparsers.add_parser("download-range", help="Download daily archives over a date range.")
    _add_archive_args(range_parser, require_date=False)
    range_parser.add_argument("--start", required=True)
    range_parser.add_argument("--end", required=True)
    range_parser.add_argument("--root", default="data/raw")
    range_parser.add_argument("--overwrite", action="store_true")
    range_parser.add_argument("--no-verify", action="store_true")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect the first CSV inside a ZIP archive.")
    inspect_parser.add_argument("path")
    inspect_parser.add_argument("--rows", type=int, default=5)

    build_parser = subparsers.add_parser(
        "build-sample",
        help="Build a bucketed quote/trade feature CSV from downloaded Binance Vision archives.",
    )
    build_parser.add_argument("--book-ticker-zip", required=True)
    build_parser.add_argument("--agg-trades-zip")
    build_parser.add_argument("--book-depth-zip")
    build_parser.add_argument("--output", required=True)
    build_parser.add_argument("--bucket-ms", type=int, default=1000)
    build_parser.add_argument("--horizon-ms", type=int, default=1000)
    build_parser.add_argument("--execution-latency-ms", type=int, default=0)
    build_parser.add_argument(
        "--execution-quote-resolution",
        choices=["raw", "bucket"],
        default="raw",
        help="Resolve entry/exit prices from first raw quote event or from retained bucket quotes.",
    )
    build_parser.add_argument(
        "--threshold",
        choices=["half_spread", "one_tick", "zero"],
        default="half_spread",
    )
    build_parser.add_argument("--min-tick", type=float, default=0.0)
    build_parser.add_argument("--large-trade-notional", type=float, default=10_000.0)
    build_parser.add_argument("--max-quote-buckets", type=int)
    build_parser.add_argument(
        "--max-feature-build-memory-gb",
        type=float,
        default=0.0,
        help="Refuse daily feature construction when compressed input-size estimate exceeds this budget. 0 disables.",
    )
    build_parser.add_argument(
        "--feature-memory-estimate-multiplier",
        type=float,
        default=12.0,
        help="Multiplier applied to compressed ZIP sizes for feature-build memory preflight.",
    )

    build_range_parser = subparsers.add_parser(
        "build-range",
        help="Download missing daily bookTicker/aggTrades archives and build feature CSVs.",
    )
    build_range_parser.add_argument("--market", default="futures/um")
    build_range_parser.add_argument("--symbol", required=True)
    build_range_parser.add_argument("--start", required=True)
    build_range_parser.add_argument("--end", required=True)
    build_range_parser.add_argument("--raw-root", default="data/raw")
    build_range_parser.add_argument("--output-dir", default="data/processed")
    build_range_parser.add_argument("--combined-output")
    build_range_parser.add_argument("--bucket-ms", type=int, default=1000)
    build_range_parser.add_argument("--horizon-ms", type=int, default=1000)
    build_range_parser.add_argument("--execution-latency-ms", type=int, default=0)
    build_range_parser.add_argument(
        "--execution-quote-resolution",
        choices=["raw", "bucket"],
        default="raw",
        help="Resolve entry/exit prices from first raw quote event or from retained bucket quotes.",
    )
    build_range_parser.add_argument(
        "--threshold",
        choices=["half_spread", "one_tick", "zero"],
        default="half_spread",
    )
    build_range_parser.add_argument("--min-tick", type=float, default=0.0)
    build_range_parser.add_argument("--large-trade-notional", type=float, default=10_000.0)
    build_range_parser.add_argument("--max-quote-buckets", type=int)
    build_range_parser.add_argument("--with-book-depth", action="store_true")
    build_range_parser.add_argument(
        "--max-feature-build-memory-gb",
        type=float,
        default=0.0,
        help="Refuse each daily feature build when compressed input-size estimate exceeds this budget. 0 disables.",
    )
    build_range_parser.add_argument(
        "--feature-memory-estimate-multiplier",
        type=float,
        default=12.0,
        help="Multiplier applied to compressed ZIP sizes for feature-build memory preflight.",
    )
    build_range_parser.add_argument("--overwrite-download", action="store_true")
    build_range_parser.add_argument("--no-verify", action="store_true")

    describe_parser = subparsers.add_parser(
        "describe-features",
        help="Summarize a feature CSV generated by build-sample.",
    )
    describe_parser.add_argument("path")

    create_holdout_parser = subparsers.add_parser(
        "create-holdout-manifest",
        help="Create a hash-locked calendar/source holdout manifest for a feature CSV.",
    )
    create_holdout_parser.add_argument("path")
    create_holdout_parser.add_argument("--output", required=True)
    create_holdout_parser.add_argument("--split-column", default="source_date")
    create_holdout_parser.add_argument("--holdout-values", required=True)
    create_holdout_parser.add_argument("--feature-version", default="feature_v1")
    create_holdout_parser.add_argument("--target-version", default="target_v1")
    create_holdout_parser.add_argument("--git-commit")
    create_holdout_parser.add_argument("--candidate-json")
    create_holdout_parser.add_argument("--notes", default="")
    create_holdout_parser.add_argument("--source-root", default=".")

    l2_manifest_parser = subparsers.add_parser(
        "l2-manifest",
        help="Create an OKX/Bybit historical L2 acquisition manifest.",
    )
    l2_manifest_parser.add_argument("--source", required=True, choices=["okx", "bybit"])
    l2_manifest_parser.add_argument(
        "--symbols", required=True, help="Comma-separated symbols, e.g. BTC-USDT-SWAP,ETH-USDT-SWAP"
    )
    l2_manifest_parser.add_argument("--start", required=True)
    l2_manifest_parser.add_argument("--end", required=True)
    l2_manifest_parser.add_argument("--dataset", default="order_book_l2")
    l2_manifest_parser.add_argument("--market", default="swap")
    l2_manifest_parser.add_argument("--output", required=True)

    l2_download_manifest_parser = subparsers.add_parser(
        "l2-download-manifest",
        help="Download direct URLs listed in a historical L2 manifest.",
    )
    l2_download_manifest_parser.add_argument("manifest")
    l2_download_manifest_parser.add_argument("--raw-root", default="data/raw")
    l2_download_manifest_parser.add_argument("--overwrite", action="store_true")

    l2_resolve_okx_parser = subparsers.add_parser(
        "l2-resolve-okx",
        help="Resolve OKX historical L2 manifest rows through OKX's public download-link endpoint.",
    )
    l2_resolve_okx_parser.add_argument("manifest")
    l2_resolve_okx_parser.add_argument("--depth", choices=["400", "5000"], default="400")
    l2_resolve_okx_parser.add_argument("--overwrite", action="store_true")
    l2_resolve_okx_parser.add_argument("--throttle-seconds", type=float, default=1.0)
    l2_resolve_okx_parser.add_argument("--timeout-seconds", type=float, default=30.0)

    l2_resolve_bybit_parser = subparsers.add_parser(
        "l2-resolve-bybit",
        help="Resolve Bybit historical orderBook manifest rows through Bybit's public history-data API.",
    )
    l2_resolve_bybit_parser.add_argument("manifest")
    l2_resolve_bybit_parser.add_argument("--overwrite", action="store_true")
    l2_resolve_bybit_parser.add_argument("--throttle-seconds", type=float, default=1.0)
    l2_resolve_bybit_parser.add_argument("--timeout-seconds", type=float, default=30.0)

    l2_validate_parser = subparsers.add_parser(
        "l2-validate",
        help="Validate one historical L2 CSV/gzip/ZIP file schema.",
    )
    l2_validate_parser.add_argument("path")
    l2_validate_parser.add_argument(
        "--source",
        required=True,
        choices=["okx", "bybit", "binance", "coinbase", "tardis", "crypto_lake"],
    )
    l2_validate_parser.add_argument("--max-rows", type=int, default=1000)

    l2_import_parser = subparsers.add_parser(
        "l2-import",
        help="Validate and normalize one OKX/Bybit historical L2 file.",
    )
    l2_import_parser.add_argument("path")
    l2_import_parser.add_argument("--source", required=True, choices=["okx", "bybit"])
    l2_import_parser.add_argument("--symbol", required=True)
    l2_import_parser.add_argument("--session-date", required=True)
    l2_import_parser.add_argument("--output-root", default="data")
    l2_import_parser.add_argument("--format", choices=["csv", "parquet"], default="csv")
    l2_import_parser.add_argument("--max-validation-rows", type=int, default=1000)
    l2_import_parser.add_argument(
        "--max-import-rows",
        type=int,
        help="Stop after writing this many normalized rows; useful for laptop-safe smoke tests.",
    )
    l2_import_parser.add_argument("--require-sequence", action="store_true")

    l2_import_manifest_parser = subparsers.add_parser(
        "l2-import-manifest",
        help="Validate and normalize downloaded entries from a historical L2 manifest.",
    )
    l2_import_manifest_parser.add_argument("manifest")
    l2_import_manifest_parser.add_argument("--output-root", default="data")
    l2_import_manifest_parser.add_argument("--format", choices=["csv", "parquet"], default="csv")
    l2_import_manifest_parser.add_argument("--max-validation-rows", type=int, default=1000)
    l2_import_manifest_parser.add_argument(
        "--max-import-rows",
        type=int,
        help="Stop each manifest entry after this many normalized rows; useful for laptop-safe smoke tests.",
    )
    l2_import_manifest_parser.add_argument("--require-sequence", action="store_true")

    evidence_gates_parser = subparsers.add_parser(
        "evidence-gates",
        help="Check machine-readable evidence gates for remaining implementation TODOs.",
    )
    evidence_gates_parser.add_argument("--format", choices=["text", "csv", "json"], default="text")
    evidence_gates_parser.add_argument("--output")
    evidence_gates_parser.add_argument("--min-shadow-observations", type=int, default=20)
    evidence_gates_parser.add_argument("--simulated-fills", default="results/shadow_validation/simulated_fills.csv")
    evidence_gates_parser.add_argument("--shadow-decisions", default="results/shadow_validation/shadow_decisions.csv")
    evidence_gates_parser.add_argument("--max-shadow-price-error", type=float, default=5.0)
    evidence_gates_parser.add_argument("--max-shadow-size-error", type=float, default=0.01)
    evidence_gates_parser.add_argument("--max-shadow-fill-rate-error", type=float, default=0.05)
    evidence_gates_parser.add_argument(
        "--l2",
        dest="l2_paths",
        action="append",
        help="Normalized L2 candidate path. Repeatable; defaults to the OKX and Bybit BTC smoke paths.",
    )
    evidence_gates_parser.add_argument(
        "--baseline-audit", default="results/current/btc_full_day_edge_zero_fee_audit.csv"
    )
    evidence_gates_parser.add_argument("--kelly-artifact", default="results/current/btc_full_day_edge_zero_fee.csv")

    free_api_sources_parser = subparsers.add_parser(
        "free-api-sources",
        help="List free/freemium APIs that can unblock local data gaps.",
    )
    free_api_sources_parser.add_argument(
        "--gap",
        choices=[PAPER_FILL_GAP, TRUE_L2_SMOKE_GAP, QUOTE_TRADE_GAP],
        help="Filter to one local data gap.",
    )
    free_api_sources_parser.add_argument(
        "--evidence-gate",
        choices=LOCAL_EVIDENCE_GATES,
        help="Filter to the unfinished evidence gate the source can help satisfy.",
    )
    free_api_sources_parser.add_argument("--format", choices=["markdown", "csv", "json"], default="markdown")

    live_l2_capture_parser = subparsers.add_parser(
        "live-l2-capture",
        help="Capture live WebSocket L2 updates into normalized L2 CSV rows.",
    )
    live_l2_capture_parser.add_argument("--venue", required=True, choices=["binance", "okx", "bybit", "coinbase"])
    live_l2_capture_parser.add_argument("--symbol", required=True)
    live_l2_capture_parser.add_argument("--output", required=True)
    live_l2_capture_parser.add_argument("--seconds", type=float, default=60.0)
    live_l2_capture_parser.add_argument("--max-messages", type=int)
    live_l2_capture_parser.add_argument("--depth", type=int, default=50)
    live_l2_capture_parser.add_argument("--category", default="spot")
    live_l2_capture_parser.add_argument("--allow-gaps", action="store_true")
    live_l2_capture_parser.add_argument("--reset-on-gap", action="store_true")

    validate_shadow_fills_parser = subparsers.add_parser(
        "validate-shadow-fills",
        help="Compare simulated fill predictions against shadow/paper fill observations by decision_id.",
    )
    validate_shadow_fills_parser.add_argument("--simulated", required=True)
    validate_shadow_fills_parser.add_argument("--shadow", required=True)
    validate_shadow_fills_parser.add_argument("--max-price-error", type=float)
    validate_shadow_fills_parser.add_argument("--max-size-error", type=float)
    validate_shadow_fills_parser.add_argument("--max-fill-rate-error", type=float)
    validate_shadow_fills_parser.add_argument("--format", choices=["text", "csv"], default="text")

    import_observed_fills_parser = subparsers.add_parser(
        "import-observed-fills",
        help="Merge paper/live fill CSV rows into a shadow-decision CSV by decision/client order id.",
    )
    import_observed_fills_parser.add_argument("--shadow", required=True)
    import_observed_fills_parser.add_argument("--observed", required=True)
    import_observed_fills_parser.add_argument("--output", required=True)
    import_observed_fills_parser.add_argument("--format", choices=["text", "csv"], default="text")

    normalize_observed_fills_parser = subparsers.add_parser(
        "normalize-observed-fills",
        help="Normalize raw paper/demo fill exports into the observed-fill import CSV schema.",
    )
    normalize_observed_fills_parser.add_argument(
        "--provider",
        required=True,
        choices=SUPPORTED_OBSERVED_FILL_PROVIDERS,
        help="Raw export provider.",
    )
    normalize_observed_fills_parser.add_argument(
        "--input", required=True, help="Raw provider .csv, .json, or .jsonl file."
    )
    normalize_observed_fills_parser.add_argument("--output", required=True)
    normalize_observed_fills_parser.add_argument("--format", choices=["text", "csv"], default="text")

    fetch_observed_fills_parser = subparsers.add_parser(
        "fetch-observed-fills",
        help="Fetch raw Bybit/OKX demo or Binance USD-M testnet fill/order history from free APIs.",
    )
    fetch_observed_fills_parser.add_argument(
        "--provider",
        required=True,
        choices=SUPPORTED_OBSERVED_FILL_FETCH_PROVIDERS,
        help="Demo API provider.",
    )
    fetch_observed_fills_parser.add_argument("--output", required=True)
    fetch_observed_fills_parser.add_argument(
        "--symbol",
        help="Bybit/Binance symbol or OKX instId, for example BTCUSDT or BTC-USDT-SWAP.",
    )
    fetch_observed_fills_parser.add_argument("--start-time-ms", type=int)
    fetch_observed_fills_parser.add_argument("--end-time-ms", type=int)
    fetch_observed_fills_parser.add_argument("--limit", type=int, default=50)
    fetch_observed_fills_parser.add_argument("--category", default="linear", help="Bybit category.")
    fetch_observed_fills_parser.add_argument("--inst-type", default="SWAP", help="OKX instrument type.")
    fetch_observed_fills_parser.add_argument("--cursor", help="Bybit nextPageCursor.")
    fetch_observed_fills_parser.add_argument("--base-url", help="Override provider REST base URL.")
    fetch_observed_fills_parser.add_argument(
        "--recv-window",
        type=int,
        default=5000,
        help="Bybit recvWindow or Binance recvWindow.",
    )
    fetch_observed_fills_parser.add_argument("--timeout-seconds", type=float, default=30.0)
    fetch_observed_fills_parser.add_argument("--format", choices=["text", "csv"], default="text")

    observed_fill_template_parser = subparsers.add_parser(
        "observed-fill-template",
        help="Write a paper/live fill import template from shadow decisions.",
    )
    observed_fill_template_parser.add_argument("--shadow", required=True)
    observed_fill_template_parser.add_argument("--output", required=True)
    observed_fill_template_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum shadow decisions to include. 0 includes all.",
    )

    simulate_shadow_fills_parser = subparsers.add_parser(
        "simulate-shadow-fills",
        help="Replay shadow decisions against market events and write simulated fill predictions.",
    )
    simulate_shadow_fills_parser.add_argument("--shadow", required=True)
    simulate_shadow_fills_parser.add_argument("--market-events", required=True)
    simulate_shadow_fills_parser.add_argument("--output", required=True)
    simulate_shadow_fills_parser.add_argument("--mode", choices=["taker", "passive"], default="taker")
    simulate_shadow_fills_parser.add_argument("--tick-size", type=float, required=True)
    simulate_shadow_fills_parser.add_argument("--lot-size", type=float, required=True)
    simulate_shadow_fills_parser.add_argument("--min-quantity", type=float, required=True)
    simulate_shadow_fills_parser.add_argument("--min-notional", type=float, required=True)
    simulate_shadow_fills_parser.add_argument("--max-notional", type=float)
    simulate_shadow_fills_parser.add_argument("--order-latency-ms", type=int, default=0)
    simulate_shadow_fills_parser.add_argument("--websocket-delay-ms", type=int, default=0)
    simulate_shadow_fills_parser.add_argument("--rate-limit-interval-ms", type=int, default=0)
    simulate_shadow_fills_parser.add_argument("--queue-ahead-size", type=float, default=0.0)
    simulate_shadow_fills_parser.add_argument("--cancellation-rate-per-second", type=float, default=0.0)
    simulate_shadow_fills_parser.add_argument("--max-wait-ms", type=int, default=1000)
    simulate_shadow_fills_parser.add_argument("--cancel-replace-edge-bps", type=float)

    features_to_events_parser = subparsers.add_parser(
        "features-to-market-events",
        help="Convert feature CSV rows into market-event CSV rows for fill simulation.",
    )
    features_to_events_parser.add_argument("path")
    features_to_events_parser.add_argument("--output", required=True)

    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Run simple threshold baselines on a generated feature CSV.",
    )
    baseline_parser.add_argument("path")
    _add_holdout_manifest_arg(baseline_parser)
    baseline_parser.add_argument("--top", type=int, default=10)
    baseline_parser.add_argument("--features", help="Comma-separated feature columns to evaluate.")
    baseline_parser.add_argument("--thresholds", help="Comma-separated absolute thresholds to evaluate.")
    baseline_parser.add_argument("--execution-model", choices=["taker", "maker_entry"], default="taker")
    baseline_parser.add_argument("--maker-fee-bps", type=float, default=0.0)
    baseline_parser.add_argument("--taker-fee-bps", type=float, default=5.0)
    baseline_parser.add_argument("--slippage-bps", type=float, default=0.0)
    baseline_parser.add_argument(
        "--sort-by",
        choices=[
            "validation_macro_f1",
            "validation_balanced_accuracy",
            "validation_net_pnl",
            "validation_gross_pnl",
        ],
        default="validation_net_pnl",
    )

    fee_sweep_parser = subparsers.add_parser(
        "fee-sweep",
        help="Select the best rule across fee tiers and report break-even fee estimates.",
    )
    fee_sweep_parser.add_argument("path")
    _add_holdout_manifest_arg(fee_sweep_parser)
    fee_sweep_parser.add_argument("--fees", default="0,0.05,0.1,0.25,0.5,1,2,3,5")
    fee_sweep_parser.add_argument("--features", help="Comma-separated feature columns to evaluate.")
    fee_sweep_parser.add_argument("--thresholds", help="Comma-separated absolute thresholds to evaluate.")
    fee_sweep_parser.add_argument("--execution-model", choices=["taker", "maker_entry"], default="taker")
    fee_sweep_parser.add_argument("--maker-fee-bps", type=float, default=0.0)
    fee_sweep_parser.add_argument("--slippage-bps", type=float, default=0.0)
    fee_sweep_parser.add_argument(
        "--sort-by",
        choices=[
            "validation_macro_f1",
            "validation_balanced_accuracy",
            "validation_net_pnl",
            "validation_gross_pnl",
        ],
        default="validation_net_pnl",
    )

    walk_forward_parser = subparsers.add_parser(
        "walk-forward",
        help="Run purged walk-forward threshold selection on a generated feature CSV.",
    )
    walk_forward_parser.add_argument("path")
    _add_holdout_manifest_arg(walk_forward_parser)
    walk_forward_parser.add_argument("--train-size", type=int, default=2400)
    walk_forward_parser.add_argument("--validation-size", type=int, default=1200)
    walk_forward_parser.add_argument("--test-size", type=int, default=1200)
    walk_forward_parser.add_argument("--step-size", type=int)
    walk_forward_parser.add_argument("--no-purge", action="store_true")
    walk_forward_parser.add_argument("--features", help="Comma-separated feature columns to evaluate.")
    walk_forward_parser.add_argument("--thresholds", help="Comma-separated absolute thresholds to evaluate.")
    walk_forward_parser.add_argument("--execution-model", choices=["taker", "maker_entry"], default="taker")
    walk_forward_parser.add_argument("--maker-fee-bps", type=float, default=0.0)
    walk_forward_parser.add_argument("--taker-fee-bps", type=float, default=5.0)
    walk_forward_parser.add_argument("--slippage-bps", type=float, default=0.0)
    walk_forward_parser.add_argument(
        "--sort-by",
        choices=[
            "validation_macro_f1",
            "validation_balanced_accuracy",
            "validation_net_pnl",
            "validation_gross_pnl",
        ],
        default="validation_net_pnl",
    )

    calendar_walk_forward_parser = subparsers.add_parser(
        "calendar-walk-forward",
        help="Run source_date-grouped walk-forward threshold selection.",
    )
    calendar_walk_forward_parser.add_argument("path")
    _add_holdout_manifest_arg(calendar_walk_forward_parser)
    calendar_walk_forward_parser.add_argument("--train-days", type=int, default=20)
    calendar_walk_forward_parser.add_argument("--validation-days", type=int, default=5)
    calendar_walk_forward_parser.add_argument("--test-days", type=int, default=5)
    calendar_walk_forward_parser.add_argument("--step-days", type=int)
    calendar_walk_forward_parser.add_argument("--no-purge", action="store_true")
    calendar_walk_forward_parser.add_argument("--features", help="Comma-separated feature columns to evaluate.")
    calendar_walk_forward_parser.add_argument("--thresholds", help="Comma-separated absolute thresholds to evaluate.")
    calendar_walk_forward_parser.add_argument("--execution-model", choices=["taker", "maker_entry"], default="taker")
    calendar_walk_forward_parser.add_argument("--maker-fee-bps", type=float, default=0.0)
    calendar_walk_forward_parser.add_argument("--taker-fee-bps", type=float, default=5.0)
    calendar_walk_forward_parser.add_argument("--slippage-bps", type=float, default=0.0)
    calendar_walk_forward_parser.add_argument(
        "--sort-by",
        choices=[
            "validation_macro_f1",
            "validation_balanced_accuracy",
            "validation_net_pnl",
            "validation_gross_pnl",
        ],
        default="validation_net_pnl",
    )

    conditional_walk_forward_parser = subparsers.add_parser(
        "conditional-walk-forward",
        help="Run purged walk-forward threshold selection with optional regime filters.",
    )
    conditional_walk_forward_parser.add_argument("path")
    _add_holdout_manifest_arg(conditional_walk_forward_parser)
    conditional_walk_forward_parser.add_argument("--train-size", type=int, default=2400)
    conditional_walk_forward_parser.add_argument("--validation-size", type=int, default=1200)
    conditional_walk_forward_parser.add_argument("--test-size", type=int, default=1200)
    conditional_walk_forward_parser.add_argument("--step-size", type=int)
    conditional_walk_forward_parser.add_argument("--no-purge", action="store_true")
    conditional_walk_forward_parser.add_argument("--features", help="Comma-separated feature columns to evaluate.")
    conditional_walk_forward_parser.add_argument(
        "--thresholds", help="Comma-separated absolute thresholds to evaluate."
    )
    conditional_walk_forward_parser.add_argument("--regime-features", help="Comma-separated columns to quantile-bin.")
    conditional_walk_forward_parser.add_argument("--regime-bins", type=int, default=3)
    conditional_walk_forward_parser.add_argument("--min-validation-trades", type=int, default=20)
    conditional_walk_forward_parser.add_argument("--execution-model", choices=["taker", "maker_entry"], default="taker")
    conditional_walk_forward_parser.add_argument("--maker-fee-bps", type=float, default=0.0)
    conditional_walk_forward_parser.add_argument("--taker-fee-bps", type=float, default=5.0)
    conditional_walk_forward_parser.add_argument("--slippage-bps", type=float, default=0.0)
    conditional_walk_forward_parser.add_argument(
        "--sort-by",
        choices=[
            "validation_macro_f1",
            "validation_balanced_accuracy",
            "validation_net_pnl",
            "validation_gross_pnl",
        ],
        default="validation_net_pnl",
    )

    logistic_walk_forward_parser = subparsers.add_parser(
        "logistic-walk-forward",
        help="Run purged walk-forward softmax logistic regression with alpha-threshold selection.",
    )
    logistic_walk_forward_parser.add_argument("path")
    _add_holdout_manifest_arg(logistic_walk_forward_parser)
    logistic_walk_forward_parser.add_argument("--train-size", type=int, default=2400)
    logistic_walk_forward_parser.add_argument("--validation-size", type=int, default=1200)
    logistic_walk_forward_parser.add_argument("--test-size", type=int, default=1200)
    logistic_walk_forward_parser.add_argument("--step-size", type=int)
    logistic_walk_forward_parser.add_argument("--no-purge", action="store_true")
    logistic_walk_forward_parser.add_argument("--features", help="Comma-separated feature columns to train on.")
    logistic_walk_forward_parser.add_argument(
        "--alpha-thresholds",
        default="0,0.025,0.05,0.075,0.1,0.15,0.2,0.3,0.5,0.75",
        help="Comma-separated thresholds for p(up)-p(down).",
    )
    logistic_walk_forward_parser.add_argument("--epochs", type=int, default=120)
    logistic_walk_forward_parser.add_argument("--learning-rate", type=float, default=0.05)
    logistic_walk_forward_parser.add_argument("--l2", type=float, default=0.001)
    logistic_walk_forward_parser.add_argument("--class-weighting", choices=["balanced", "none"], default="balanced")
    logistic_walk_forward_parser.add_argument("--execution-model", choices=["taker", "maker_entry"], default="taker")
    logistic_walk_forward_parser.add_argument("--maker-fee-bps", type=float, default=0.0)
    logistic_walk_forward_parser.add_argument("--taker-fee-bps", type=float, default=5.0)
    logistic_walk_forward_parser.add_argument("--slippage-bps", type=float, default=0.0)
    logistic_walk_forward_parser.add_argument(
        "--sort-by",
        choices=[
            "validation_macro_f1",
            "validation_balanced_accuracy",
            "validation_net_pnl",
            "validation_gross_pnl",
        ],
        default="validation_net_pnl",
    )

    edge_walk_forward_parser = subparsers.add_parser(
        "edge-walk-forward",
        help="Run purged walk-forward ridge regression on expected taker edge.",
    )
    edge_walk_forward_parser.add_argument("path")
    _add_holdout_manifest_arg(edge_walk_forward_parser)
    edge_walk_forward_parser.add_argument("--train-size", type=int, default=2400)
    edge_walk_forward_parser.add_argument("--validation-size", type=int, default=1200)
    edge_walk_forward_parser.add_argument("--test-size", type=int, default=1200)
    edge_walk_forward_parser.add_argument("--step-size", type=int)
    edge_walk_forward_parser.add_argument("--max-folds", type=int)
    edge_walk_forward_parser.add_argument("--no-purge", action="store_true")
    edge_walk_forward_parser.add_argument(
        "--stream",
        action="store_true",
        default=True,
        help="Evaluate walk-forward folds with a bounded-memory rolling CSV window. Enabled by default.",
    )
    edge_walk_forward_parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Load the whole CSV before evaluating. Use only for tiny debug files or high-RAM machines.",
    )
    edge_walk_forward_parser.add_argument("--features", help="Comma-separated feature columns to train on.")
    edge_walk_forward_parser.add_argument(
        "--edge-thresholds-bps",
        default="0,0.025,0.05,0.075,0.1,0.15,0.2,0.3,0.5,0.75,1,2,5",
        help="Comma-separated predicted net-edge thresholds in bps.",
    )
    edge_walk_forward_parser.add_argument("--l2", type=float, default=1.0)
    edge_walk_forward_parser.add_argument("--taker-fee-bps", type=float, default=5.0)
    edge_walk_forward_parser.add_argument("--slippage-bps", type=float, default=0.0)
    edge_walk_forward_parser.add_argument(
        "--max-load-memory-gb",
        type=float,
        default=0.0,
        help="Refuse to load the CSV when estimated Python memory use exceeds this budget. 0 disables.",
    )
    edge_walk_forward_parser.add_argument(
        "--memory-estimate-multiplier",
        type=float,
        default=8.0,
        help="Multiplier applied to CSV file size for the memory guard.",
    )
    edge_walk_forward_parser.add_argument(
        "--sort-by",
        choices=[
            "validation_macro_f1",
            "validation_balanced_accuracy",
            "validation_net_pnl",
            "validation_gross_pnl",
        ],
        default="validation_net_pnl",
    )

    edge_shadow_parser = subparsers.add_parser(
        "edge-shadow-decisions",
        help="Export OOS shadow decisions from the expected-edge walk-forward model.",
    )
    edge_shadow_parser.add_argument("path")
    _add_holdout_manifest_arg(edge_shadow_parser)
    edge_shadow_parser.add_argument("--output", required=True)
    edge_shadow_parser.add_argument("--venue", required=True)
    edge_shadow_parser.add_argument("--symbol", required=True)
    edge_shadow_parser.add_argument("--order-type", choices=["paper_taker", "paper_limit"], default="paper_taker")
    edge_shadow_parser.add_argument("--intended-size", type=float)
    edge_shadow_parser.add_argument("--intended-notional", type=float)
    edge_shadow_parser.add_argument("--include-flat", action="store_true")
    edge_shadow_parser.add_argument("--train-size", type=int, default=2400)
    edge_shadow_parser.add_argument("--validation-size", type=int, default=1200)
    edge_shadow_parser.add_argument("--test-size", type=int, default=1200)
    edge_shadow_parser.add_argument("--step-size", type=int)
    edge_shadow_parser.add_argument("--max-folds", type=int)
    edge_shadow_parser.add_argument("--no-purge", action="store_true")
    edge_shadow_parser.add_argument("--features", help="Comma-separated feature columns to train on.")
    edge_shadow_parser.add_argument(
        "--edge-thresholds-bps",
        default="0,0.025,0.05,0.075,0.1,0.15,0.2,0.3,0.5,0.75,1,2,5",
        help="Comma-separated predicted net-edge thresholds in bps.",
    )
    edge_shadow_parser.add_argument("--l2", type=float, default=1.0)
    edge_shadow_parser.add_argument("--taker-fee-bps", type=float, default=5.0)
    edge_shadow_parser.add_argument("--slippage-bps", type=float, default=0.0)
    edge_shadow_parser.add_argument(
        "--sort-by",
        choices=[
            "validation_macro_f1",
            "validation_balanced_accuracy",
            "validation_net_pnl",
            "validation_gross_pnl",
        ],
        default="validation_net_pnl",
    )

    eval_rule_parser = subparsers.add_parser(
        "eval-rule",
        help="Evaluate one fixed threshold rule on a full file or a source_date slice.",
    )
    eval_rule_parser.add_argument("path")
    _add_holdout_manifest_arg(eval_rule_parser)
    eval_rule_parser.add_argument("--feature", required=True)
    eval_rule_parser.add_argument("--threshold", type=float, required=True)
    eval_rule_parser.add_argument("--source-date")
    eval_rule_parser.add_argument("--execution-model", choices=["taker", "maker_entry"], default="taker")
    eval_rule_parser.add_argument("--maker-fee-bps", type=float, default=0.0)
    eval_rule_parser.add_argument("--taker-fee-bps", type=float, default=5.0)
    eval_rule_parser.add_argument("--slippage-bps", type=float, default=0.0)

    regime_parser = subparsers.add_parser(
        "regime",
        help="Evaluate one fixed rule by spread, volatility, and liquidity regimes.",
    )
    regime_parser.add_argument("path")
    _add_holdout_manifest_arg(regime_parser)
    regime_parser.add_argument("--feature", required=True)
    regime_parser.add_argument("--threshold", type=float, required=True)
    regime_parser.add_argument("--regime-features", help="Comma-separated columns to quantile-bin.")
    regime_parser.add_argument("--bins", type=int, default=3)
    regime_parser.add_argument("--source-date")
    regime_parser.add_argument("--execution-model", choices=["taker", "maker_entry"], default="taker")
    regime_parser.add_argument("--maker-fee-bps", type=float, default=0.0)
    regime_parser.add_argument("--taker-fee-bps", type=float, default=5.0)
    regime_parser.add_argument("--slippage-bps", type=float, default=0.0)

    final_holdout_rule_parser = subparsers.add_parser(
        "final-holdout-rule",
        help="Evaluate one frozen threshold-rule candidate on the declared holdout exactly once.",
    )
    final_holdout_rule_parser.add_argument("path")
    final_holdout_rule_parser.add_argument("--holdout-manifest", required=True)
    final_holdout_rule_parser.add_argument("--candidate-json", required=True)
    final_holdout_rule_parser.add_argument("--output", required=True)
    final_holdout_rule_parser.add_argument("--lock-dir", default="artifacts/final_holdout_locks")
    final_holdout_rule_parser.add_argument("--explicit-final-evaluation", action="store_true")

    freeze_threshold_candidate_parser = subparsers.add_parser(
        "freeze-threshold-candidate",
        help="Freeze one threshold-rule candidate from a validation-selected result CSV for final holdout use.",
    )
    freeze_threshold_candidate_parser.add_argument("artifact")
    freeze_threshold_candidate_parser.add_argument("--output", required=True)
    freeze_threshold_candidate_parser.add_argument(
        "--sort-by",
        choices=["validation_net_pnl", "test_net_pnl", "validation_break_even_fee_bps", "test_break_even_fee_bps"],
        default="validation_net_pnl",
    )
    freeze_threshold_candidate_parser.add_argument(
        "--execution-model", choices=["taker", "maker_entry"], default="taker"
    )
    freeze_threshold_candidate_parser.add_argument("--order-type", choices=["market", "passive"])
    freeze_threshold_candidate_parser.add_argument("--maker-fee-bps", type=float, default=0.0)
    freeze_threshold_candidate_parser.add_argument("--taker-fee-bps", type=float, default=5.0)
    freeze_threshold_candidate_parser.add_argument("--slippage-bps", type=float, default=0.0)
    freeze_threshold_candidate_parser.add_argument("--target-notional", type=float)
    freeze_threshold_candidate_parser.add_argument("--initial-cash", type=float)
    freeze_threshold_candidate_parser.add_argument("--max-position-notional", type=float)
    freeze_threshold_candidate_parser.add_argument("--max-leverage", type=float)
    freeze_threshold_candidate_parser.add_argument("--latency-ms", type=int)
    freeze_threshold_candidate_parser.add_argument("--max-order-age-ms", type=int)
    freeze_threshold_candidate_parser.add_argument("--kill-switch-loss", type=float)
    freeze_threshold_candidate_parser.add_argument("--rate-limit-interval-ms", type=int)
    freeze_threshold_candidate_parser.add_argument("--queue-ahead-size", type=float)
    freeze_threshold_candidate_parser.add_argument("--cancellation-rate-per-second", type=float)
    freeze_threshold_candidate_parser.add_argument("--cancel-replace-edge-bps", type=float)

    fill_parser = subparsers.add_parser(
        "fill-diagnostics",
        help="Analyze conservative passive-entry fills and adverse selection for one threshold rule.",
    )
    fill_parser.add_argument("path")
    fill_parser.add_argument("--feature", required=True)
    fill_parser.add_argument("--threshold", type=float, required=True)
    fill_parser.add_argument("--source-date")
    fill_parser.add_argument("--by-source-date", action="store_true")
    fill_parser.add_argument("--maker-fee-bps", type=float, default=0.0)
    fill_parser.add_argument("--taker-fee-bps", type=float, default=0.0)
    fill_parser.add_argument("--slippage-bps", type=float, default=0.0)

    fill_regime_parser = subparsers.add_parser(
        "fill-regime",
        help="Analyze conservative passive-entry fills by market-state quantile buckets.",
    )
    fill_regime_parser.add_argument("path")
    fill_regime_parser.add_argument("--feature", required=True)
    fill_regime_parser.add_argument("--threshold", type=float, required=True)
    fill_regime_parser.add_argument("--regime-features", required=True)
    fill_regime_parser.add_argument("--bins", type=int, default=3)
    fill_regime_parser.add_argument("--source-date")
    fill_regime_parser.add_argument("--maker-fee-bps", type=float, default=0.0)
    fill_regime_parser.add_argument("--taker-fee-bps", type=float, default=0.0)
    fill_regime_parser.add_argument("--slippage-bps", type=float, default=0.0)

    capacity_parser = subparsers.add_parser(
        "capacity",
        help="Estimate conservative taker capacity from trade notional and displayed top-of-book size.",
    )
    capacity_parser.add_argument("path")
    capacity_parser.add_argument("--feature", required=True)
    capacity_parser.add_argument("--threshold", type=float, required=True)
    capacity_parser.add_argument("--participation-rate", type=float, default=0.01)
    capacity_parser.add_argument("--top-book-fraction", type=float, default=0.05)
    capacity_parser.add_argument("--max-notional", type=float)
    capacity_parser.add_argument("--source-date")
    capacity_parser.add_argument("--by-source-date", action="store_true")

    kelly_variance_parser = subparsers.add_parser(
        "kelly-variance-gate",
        help="Check whether fold-level OOS variance is stable enough to enable fractional Kelly sizing.",
    )
    kelly_variance_parser.add_argument("path")
    kelly_variance_parser.add_argument("--column", default="validation_net_pnl")
    kelly_variance_parser.add_argument("--min-observations", type=int, default=20)
    kelly_variance_parser.add_argument("--window-size", type=int, default=5)
    kelly_variance_parser.add_argument("--max-variance-cv", type=float, default=0.5)
    kelly_variance_parser.add_argument("--format", choices=["text", "csv"], default="text")

    model_readiness_parser = subparsers.add_parser(
        "model-readiness-gate",
        help="Check whether baseline and true-L2 prerequisites are ready before sequence/deep model experiments.",
    )
    model_readiness_parser.add_argument(
        "--model", required=True, choices=["sequence_tcn", "sequence_transformer", "lob_cnn"]
    )
    model_readiness_parser.add_argument("--baseline-audit", required=True)
    model_readiness_parser.add_argument("--l2", required=True)
    model_readiness_parser.add_argument("--min-fold-count", type=int, default=20)
    model_readiness_parser.add_argument("--min-l2-rows", type=int, default=1000)
    model_readiness_parser.add_argument("--max-l2-rows", type=int, default=100000)
    model_readiness_parser.add_argument("--allow-fi2010", action="store_true")
    model_readiness_parser.add_argument("--allow-snapshot-only", action="store_true")
    model_readiness_parser.add_argument("--format", choices=["text", "csv"], default="text")

    sequence_experiment_parser = subparsers.add_parser(
        "l2-sequence-experiment",
        help="Run a gated Torch TCN/Transformer experiment on verified normalized L2 tensors.",
    )
    sequence_experiment_parser.add_argument("--model", required=True, choices=["sequence_tcn", "sequence_transformer"])
    sequence_experiment_parser.add_argument("--baseline-audit", required=True)
    sequence_experiment_parser.add_argument("--l2", required=True)
    sequence_experiment_parser.add_argument("--output", required=True)
    sequence_experiment_parser.add_argument("--depth", type=int, default=5)
    sequence_experiment_parser.add_argument("--window", type=int, default=16)
    sequence_experiment_parser.add_argument("--label-horizon", type=int, default=1)
    sequence_experiment_parser.add_argument("--flat-threshold-bps", type=float, default=0.0)
    sequence_experiment_parser.add_argument("--epochs", type=int, default=3)
    sequence_experiment_parser.add_argument("--learning-rate", type=float, default=0.001)
    sequence_experiment_parser.add_argument("--batch-size", type=int, default=32)
    sequence_experiment_parser.add_argument("--early-stopping-patience", type=int, default=3)
    sequence_experiment_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    sequence_experiment_parser.add_argument("--class-weighting", choices=["none", "balanced"], default="none")
    sequence_experiment_parser.add_argument("--lr-scheduler-gamma", type=float, default=1.0)
    sequence_experiment_parser.add_argument("--checkpoint-path")
    sequence_experiment_parser.add_argument("--resume-from-checkpoint", action="store_true")
    sequence_experiment_parser.add_argument("--predictions-output")
    sequence_experiment_parser.add_argument(
        "--holdout-manifest",
        help="Verified holdout manifest for the source L2 CSV. When supplied, training uses a filtered development L2 CSV.",
    )
    sequence_experiment_parser.add_argument(
        "--development-l2-output",
        help="Optional output path for the manifest-filtered development L2 CSV.",
    )
    sequence_experiment_parser.add_argument("--economic-target-notional", type=float, default=100.0)
    sequence_experiment_parser.add_argument("--economic-taker-fee-bps", type=float, default=1.0)
    sequence_experiment_parser.add_argument("--economic-slippage-bps", type=float, default=0.0)
    sequence_experiment_parser.add_argument("--max-rows", type=int, default=100000)
    sequence_experiment_parser.add_argument("--max-snapshots", type=int, default=2000)
    sequence_experiment_parser.add_argument("--min-fold-count", type=int, default=20)
    sequence_experiment_parser.add_argument("--min-l2-rows", type=int, default=1000)
    sequence_experiment_parser.add_argument("--seed", type=int, default=7)
    sequence_experiment_parser.add_argument("--format", choices=["text", "csv"], default="text")

    l2_pretraining_parser = subparsers.add_parser(
        "l2-pretraining-smoke",
        help="Write a dependency-free masked reconstruction artifact from verified normalized L2 rows.",
    )
    l2_pretraining_parser.add_argument("--l2", required=True)
    l2_pretraining_parser.add_argument("--output", default="results/model_experiments/self_supervised_pretraining.csv")
    l2_pretraining_parser.add_argument("--depth", type=int, default=5)
    l2_pretraining_parser.add_argument("--window", type=int, default=4)
    l2_pretraining_parser.add_argument("--mask-probability", type=float, default=0.15)
    l2_pretraining_parser.add_argument("--max-rows", type=int, default=100000)
    l2_pretraining_parser.add_argument("--max-snapshots", type=int, default=2000)
    l2_pretraining_parser.add_argument("--format", choices=["text", "csv"], default="text")

    audit_parser = subparsers.add_parser(
        "audit-results",
        help="Audit a saved walk-forward result CSV against alpha-factory acceptance gates.",
    )
    audit_parser.add_argument("path")
    audit_parser.add_argument("--assumed-cost-bps", type=float, default=0.0)
    audit_parser.add_argument("--cost-safety-multiple", type=float, default=2.0)
    audit_parser.add_argument("--min-fold-count", type=int, default=20)
    audit_parser.add_argument("--min-positive-fold-rate", type=float, default=0.70)
    audit_parser.add_argument("--max-fold-contribution", type=float, default=0.40)
    audit_parser.add_argument("--allow-nonpositive-median", action="store_true")
    audit_parser.add_argument("--allow-nonpositive-total", action="store_true")
    audit_parser.add_argument("--require-positive-bootstrap-lower-bound", action="store_true")
    audit_parser.add_argument("--bootstrap-samples", type=int, default=2000)
    audit_parser.add_argument("--seed", type=int, default=7)
    audit_parser.add_argument("--format", choices=["csv", "markdown"], default="csv")

    pvalue_parser = subparsers.add_parser(
        "pvalue-correction",
        help="Apply Bonferroni and Benjamini-Hochberg correction to a p-value CSV.",
    )
    pvalue_parser.add_argument("path")
    pvalue_parser.add_argument("--id-column", default="hypothesis_id")
    pvalue_parser.add_argument("--p-value-column", default="p_value")
    pvalue_parser.add_argument("--metric-column", default="metric")
    pvalue_parser.add_argument("--q", type=float, default=0.05)

    args = parser.parse_args(argv)

    if args.command == "datasets":
        _print_datasets()
        return 0
    if args.command == "list":
        return _cmd_list(args)
    if args.command == "download":
        return _cmd_download(args)
    if args.command == "download-range":
        return _cmd_download_range(args)
    if args.command == "inspect":
        return _cmd_inspect(args)
    if args.command == "build-sample":
        return _cmd_build_sample(args)
    if args.command == "build-range":
        return _cmd_build_range(args)
    if args.command == "describe-features":
        return _cmd_describe_features(args)
    if args.command == "create-holdout-manifest":
        return _cmd_create_holdout_manifest(args)
    if args.command == "l2-manifest":
        return _cmd_l2_manifest(args)
    if args.command == "l2-download-manifest":
        return _cmd_l2_download_manifest(args)
    if args.command == "l2-resolve-okx":
        return _cmd_l2_resolve_okx(args)
    if args.command == "l2-resolve-bybit":
        return _cmd_l2_resolve_bybit(args)
    if args.command == "l2-validate":
        return _cmd_l2_validate(args)
    if args.command == "l2-import":
        return _cmd_l2_import(args)
    if args.command == "l2-import-manifest":
        return _cmd_l2_import_manifest(args)
    if args.command == "evidence-gates":
        return _cmd_evidence_gates(args)
    if args.command == "free-api-sources":
        return _cmd_free_api_sources(args)
    if args.command == "live-l2-capture":
        return _cmd_live_l2_capture(args)
    if args.command == "validate-shadow-fills":
        return _cmd_validate_shadow_fills(args)
    if args.command == "import-observed-fills":
        return _cmd_import_observed_fills(args)
    if args.command == "normalize-observed-fills":
        return _cmd_normalize_observed_fills(args)
    if args.command == "fetch-observed-fills":
        return _cmd_fetch_observed_fills(args)
    if args.command == "observed-fill-template":
        return _cmd_observed_fill_template(args)
    if args.command == "simulate-shadow-fills":
        return _cmd_simulate_shadow_fills(args)
    if args.command == "features-to-market-events":
        return _cmd_features_to_market_events(args)
    if args.command == "baseline":
        return _cmd_baseline(args)
    if args.command == "fee-sweep":
        return _cmd_fee_sweep(args)
    if args.command == "walk-forward":
        return _cmd_walk_forward(args)
    if args.command == "calendar-walk-forward":
        return _cmd_calendar_walk_forward(args)
    if args.command == "conditional-walk-forward":
        return _cmd_conditional_walk_forward(args)
    if args.command == "logistic-walk-forward":
        return _cmd_logistic_walk_forward(args)
    if args.command == "edge-walk-forward":
        return _cmd_edge_walk_forward(args)
    if args.command == "edge-shadow-decisions":
        return _cmd_edge_shadow_decisions(args)
    if args.command == "eval-rule":
        return _cmd_eval_rule(args)
    if args.command == "regime":
        return _cmd_regime(args)
    if args.command == "final-holdout-rule":
        return _cmd_final_holdout_rule(args)
    if args.command == "freeze-threshold-candidate":
        return _cmd_freeze_threshold_candidate(args)
    if args.command == "fill-diagnostics":
        return _cmd_fill_diagnostics(args)
    if args.command == "fill-regime":
        return _cmd_fill_regime(args)
    if args.command == "capacity":
        return _cmd_capacity(args)
    if args.command == "kelly-variance-gate":
        return _cmd_kelly_variance_gate(args)
    if args.command == "model-readiness-gate":
        return _cmd_model_readiness_gate(args)
    if args.command == "l2-sequence-experiment":
        return _cmd_l2_sequence_experiment(args)
    if args.command == "l2-pretraining-smoke":
        return _cmd_l2_pretraining_smoke(args)
    if args.command == "audit-results":
        return _cmd_audit_results(args)
    if args.command == "pvalue-correction":
        return _cmd_pvalue_correction(args)

    parser.error(f"unknown command: {args.command}")
    return 2


def _add_holdout_manifest_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--holdout-manifest",
        required=True,
        help="Required development-data manifest. The command runs on a physical CSV with declared holdout rows removed.",
    )


@contextmanager
def _development_feature_path(args: argparse.Namespace) -> Iterator[Path]:
    manifest = read_holdout_manifest(args.holdout_manifest)
    if not verify_holdout_manifest_file(args.holdout_manifest):
        raise ValueError("holdout manifest verification failed; create an official manifest from a Git checkout")
    with tempfile.TemporaryDirectory(prefix="lob_forge_development_") as tmpdir:
        result = write_development_csv(
            Path(args.path),
            manifest,
            Path(tmpdir) / "development.csv",
        )
        yield result.path


def _cmd_create_holdout_manifest(args: argparse.Namespace) -> int:
    candidate_sha256 = (
        canonical_json_sha256(json.loads(Path(args.candidate_json).read_text())) if args.candidate_json else ""
    )
    manifest = build_holdout_manifest(
        Path(args.path),
        split_column=args.split_column,
        holdout_values=_parse_string_list(args.holdout_values),
        created_at_utc=_utc_now_z(),
        feature_version=args.feature_version,
        target_version=args.target_version,
        git_commit=args.git_commit,
        candidate_sha256=candidate_sha256,
        notes=args.notes,
        source_root=Path(args.source_root),
    )
    output = write_holdout_manifest(manifest, Path(args.output))
    print(f"holdout_manifest={output}")
    print(f"source_sha256={manifest.source_sha256}")
    if candidate_sha256:
        print(f"candidate_sha256={candidate_sha256}")
    return 0


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _add_archive_args(parser: argparse.ArgumentParser, *, require_date: bool) -> None:
    parser.add_argument("--market", default="futures/um", help="Example: futures/um, futures/cm, spot")
    parser.add_argument("--frequency", default="daily", choices=["daily", "monthly"])
    parser.add_argument("--dataset", required=True, help="Example: bookDepth, bookTicker, trades, klines")
    parser.add_argument("--symbol", required=True, help="Example: BTCUSDT")
    parser.add_argument("--date", required=require_date, help="YYYY-MM-DD for daily, YYYY-MM for monthly")
    parser.add_argument("--interval", help="Required for kline datasets, e.g. 1m")


def _cmd_list(args: argparse.Namespace) -> int:
    if args.date:
        key = archive_key(
            market=args.market,
            frequency=args.frequency,
            dataset=args.dataset,
            symbol=args.symbol,
            date_value=args.date,
            interval=args.interval,
        )
        prefix = key
    else:
        prefix = dataset_prefix(
            market=args.market,
            frequency=args.frequency,
            dataset=args.dataset,
            symbol=args.symbol,
            interval=args.interval,
        )

    objects = list_objects(prefix, limit=args.limit)
    for obj in objects:
        date_value = infer_date_from_key(obj.key) or "unknown"
        print(f"{date_value}\t{obj.size}\t{obj.key}")
    print(f"listed={len(objects)}", file=sys.stderr)
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    path = download_archive(
        market=args.market,
        frequency=args.frequency,
        dataset=args.dataset,
        symbol=args.symbol,
        date_value=args.date,
        interval=args.interval,
        root=Path(args.root),
        overwrite=args.overwrite,
        verify_checksum=not args.no_verify,
    )
    print(path)
    return 0


def _cmd_download_range(args: argparse.Namespace) -> int:
    if args.frequency != "daily":
        raise SystemExit("download-range currently supports daily archives only")
    for date_value in iter_dates(args.start, args.end):
        path = download_archive(
            market=args.market,
            frequency=args.frequency,
            dataset=args.dataset,
            symbol=args.symbol,
            date_value=date_value,
            interval=args.interval,
            root=Path(args.root),
            overwrite=args.overwrite,
            verify_checksum=not args.no_verify,
        )
        print(path)
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    sample = sample_zip_csv(Path(args.path), rows=args.rows)
    print(f"archive: {sample.archive}")
    print(f"inner_csv: {sample.inner_name}")
    if sample.header:
        print("columns: " + ", ".join(sample.header))
    else:
        print("columns: unknown")
    print("sample:")
    for row in sample.rows:
        print(",".join(row))
    return 0


def _cmd_build_sample(args: argparse.Namespace) -> int:
    output = build_quote_trade_dataset(
        book_ticker_zip=Path(args.book_ticker_zip),
        agg_trades_zip=Path(args.agg_trades_zip) if args.agg_trades_zip else None,
        book_depth_zip=Path(args.book_depth_zip) if args.book_depth_zip else None,
        output_csv=Path(args.output),
        bucket_ms=args.bucket_ms,
        horizon_ms=args.horizon_ms,
        execution_latency_ms=args.execution_latency_ms,
        threshold=args.threshold,
        min_tick=args.min_tick,
        large_trade_notional=args.large_trade_notional,
        max_quote_buckets=args.max_quote_buckets,
        max_feature_build_memory_gb=args.max_feature_build_memory_gb,
        memory_estimate_multiplier=args.feature_memory_estimate_multiplier,
        execution_quote_resolution=args.execution_quote_resolution,
    )
    print(output)
    return 0


def _cmd_build_range(args: argparse.Namespace) -> int:
    results, combined = build_daily_feature_range(
        market=args.market,
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        raw_root=Path(args.raw_root),
        output_dir=Path(args.output_dir),
        combined_output=Path(args.combined_output) if args.combined_output else None,
        bucket_ms=args.bucket_ms,
        horizon_ms=args.horizon_ms,
        execution_latency_ms=args.execution_latency_ms,
        threshold=args.threshold,
        min_tick=args.min_tick,
        large_trade_notional=args.large_trade_notional,
        max_quote_buckets=args.max_quote_buckets,
        with_book_depth=args.with_book_depth,
        overwrite_download=args.overwrite_download,
        verify_checksum=not args.no_verify,
        max_feature_build_memory_gb=args.max_feature_build_memory_gb,
        memory_estimate_multiplier=args.feature_memory_estimate_multiplier,
        execution_quote_resolution=args.execution_quote_resolution,
    )
    for result in results:
        print(f"{result.date}\t{result.feature_csv}")
    if combined:
        print(f"combined\t{combined}")
    return 0


def _cmd_describe_features(args: argparse.Namespace) -> int:
    summary = summarize_feature_csv(Path(args.path))
    print(f"rows: {summary.rows}")
    print(f"label_counts: {summary.label_counts}")
    print(f"rows_with_trades: {summary.rows_with_trades}")
    print(f"rows_with_depth: {summary.rows_with_depth}")
    print(f"spread_min_max: {summary.spread_min}, {summary.spread_max}")
    print(f"delta_mid_min_max: {summary.delta_mid_min}, {summary.delta_mid_max}")
    return 0


def _cmd_l2_manifest(args: argparse.Namespace) -> int:
    entries = build_historical_l2_manifest(
        source_id=args.source,
        symbols=_parse_string_list(args.symbols),
        start=args.start,
        end=args.end,
        dataset=args.dataset,
        market=args.market,
    )
    path = write_historical_l2_manifest(entries, args.output)
    print(f"manifest={path} entries={len(entries)}")
    return 0


def _cmd_l2_download_manifest(args: argparse.Namespace) -> int:
    entries = download_historical_l2_manifest(
        args.manifest,
        raw_root=Path(args.raw_root),
        overwrite=args.overwrite,
    )
    downloaded = [entry for entry in entries if entry.status == "downloaded"]
    pending = [entry for entry in entries if entry.status != "downloaded"]
    print(f"manifest={args.manifest} downloaded={len(downloaded)} pending={len(pending)}")
    return 0


def _cmd_l2_resolve_okx(args: argparse.Namespace) -> int:
    entries = resolve_okx_historical_l2_manifest(
        args.manifest,
        depth=args.depth,
        overwrite=args.overwrite,
        throttle_seconds=args.throttle_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    okx_entries = [entry for entry in entries if entry.source_id == "okx"]
    resolved = [entry for entry in okx_entries if entry.direct_url]
    pending = [entry for entry in okx_entries if not entry.direct_url]
    failed = [entry for entry in okx_entries if entry.status in {"resolve_failed", "url_missing"}]
    print(
        f"manifest={args.manifest} okx_entries={len(okx_entries)} "
        f"resolved={len(resolved)} pending={len(pending)} failed={len(failed)}"
    )
    return 0 if not failed else 1


def _cmd_l2_resolve_bybit(args: argparse.Namespace) -> int:
    entries = resolve_bybit_historical_l2_manifest(
        args.manifest,
        overwrite=args.overwrite,
        throttle_seconds=args.throttle_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    bybit_entries = [entry for entry in entries if entry.source_id == "bybit"]
    resolved = [entry for entry in bybit_entries if entry.direct_url]
    pending = [entry for entry in bybit_entries if not entry.direct_url]
    failed = [entry for entry in bybit_entries if entry.status in {"resolve_failed", "url_missing"}]
    print(
        f"manifest={args.manifest} bybit_entries={len(bybit_entries)} "
        f"resolved={len(resolved)} pending={len(pending)} failed={len(failed)}"
    )
    return 0 if not failed else 1


def _cmd_l2_validate(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if args.source == "bybit" and _looks_like_bybit_data_zip(path):
        result = validate_bybit_orderbook_data_zip(path, max_rows=args.max_rows)
    else:
        result = validate_l2_csv_schema(
            path,
            source_format=args.source,
            max_rows=args.max_rows,
        )
    print(format_schema_validation(result))
    return 0 if result.ok else 1


def _cmd_l2_import(args: argparse.Namespace) -> int:
    result = import_historical_l2_file(
        args.path,
        source_id=args.source,
        symbol=args.symbol,
        session_date=args.session_date,
        output_root=Path(args.output_root),
        storage_format=args.format,
        max_validation_rows=args.max_validation_rows,
        max_import_rows=args.max_import_rows,
        require_sequence=True if args.require_sequence else None,
    )
    print(
        f"source={result.source_id} symbol={result.symbol} session_date={result.session_date} "
        f"rows_written={result.rows_written} validation_rows={result.validation.rows_checked} "
        f"output={result.output_path}"
    )
    return 0


def _cmd_l2_import_manifest(args: argparse.Namespace) -> int:
    results = import_historical_l2_manifest(
        args.manifest,
        output_root=Path(args.output_root),
        storage_format=args.format,
        max_validation_rows=args.max_validation_rows,
        max_import_rows=args.max_import_rows,
        require_sequence=True if args.require_sequence else None,
    )
    for result in results:
        print(
            f"source={result.source_id} symbol={result.symbol} session_date={result.session_date} "
            f"rows_written={result.rows_written} validation_rows={result.validation.rows_checked} "
            f"output={result.output_path}"
        )
    print(f"manifest={args.manifest} imported={len(results)}")
    return 0


def _cmd_evidence_gates(args: argparse.Namespace) -> int:
    report = evaluate_remaining_evidence_gates(
        simulated_fills=args.simulated_fills,
        shadow_decisions=args.shadow_decisions,
        min_shadow_observations=args.min_shadow_observations,
        max_price_error=args.max_shadow_price_error,
        max_size_error=args.max_shadow_size_error,
        max_fill_rate_error=args.max_shadow_fill_rate_error,
        l2_paths=args.l2_paths,
        baseline_audit=args.baseline_audit,
        kelly_artifact=args.kelly_artifact,
    )
    if args.output:
        output_path = write_evidence_gate_report(report, Path(args.output), output_format=args.format)
        print(f"evidence_gates={output_path}")
    print(format_evidence_gate_report(report, output_format=args.format))
    return 0 if report.passed else 1


def _cmd_free_api_sources(args: argparse.Namespace) -> int:
    sources = list_local_api_sources(data_gap=args.gap, evidence_gate=args.evidence_gate)
    if args.format == "csv":
        print(format_local_api_sources_csv(sources))
    elif args.format == "json":
        print(format_local_api_sources_json(sources))
    else:
        print(format_local_api_sources_markdown(sources))
    return 0


def _cmd_live_l2_capture(args: argparse.Namespace) -> int:
    summary = asyncio.run(
        capture_live_l2(
            venue=args.venue,
            symbol=args.symbol,
            output_path=Path(args.output),
            seconds=args.seconds,
            max_messages=args.max_messages,
            depth=args.depth,
            category=args.category,
            fail_on_gap=not (args.allow_gaps or args.reset_on_gap),
            reset_on_gap=args.reset_on_gap,
        )
    )
    print(
        f"venue={summary.venue} symbol={summary.symbol} messages_seen={summary.messages_seen} "
        f"rows_written={summary.rows_written} sequence_gaps={summary.sequence_gaps} output={summary.output_path}"
    )
    return 0


def _cmd_validate_shadow_fills(args: argparse.Namespace) -> int:
    report = validate_shadow_fill_predictions(
        simulated_path=Path(args.simulated),
        shadow_path=Path(args.shadow),
        max_price_error=args.max_price_error,
        max_size_error=args.max_size_error,
        max_fill_rate_error=args.max_fill_rate_error,
    )
    print(format_shadow_fill_validation_report(report, output_format=args.format))
    return 0 if report.passed else 1


def _cmd_import_observed_fills(args: argparse.Namespace) -> int:
    report = merge_observed_fills_into_shadow_decisions(
        shadow_path=Path(args.shadow),
        observed_path=Path(args.observed),
        output_path=Path(args.output),
    )
    print(format_observed_fill_merge_report(report, output_format=args.format))
    return 0


def _cmd_normalize_observed_fills(args: argparse.Namespace) -> int:
    report = normalize_observed_fills(
        provider=args.provider,
        input_path=Path(args.input),
        output_path=Path(args.output),
    )
    print(format_observed_fill_normalization_report(report, output_format=args.format))
    return 0


def _cmd_fetch_observed_fills(args: argparse.Namespace) -> int:
    report = fetch_observed_fill_export(
        provider=args.provider,
        output_path=Path(args.output),
        symbol=args.symbol,
        start_time_ms=args.start_time_ms,
        end_time_ms=args.end_time_ms,
        limit=args.limit,
        category=args.category,
        inst_type=args.inst_type,
        cursor=args.cursor,
        base_url=args.base_url,
        recv_window=args.recv_window,
        timeout_seconds=args.timeout_seconds,
    )
    print(format_observed_fill_fetch_report(report, output_format=args.format))
    return 0


def _cmd_observed_fill_template(args: argparse.Namespace) -> int:
    output = write_observed_fill_template(
        shadow_path=Path(args.shadow),
        output_path=Path(args.output),
        limit=args.limit,
    )
    print(f"observed_fill_template={output}")
    return 0


def _cmd_simulate_shadow_fills(args: argparse.Namespace) -> int:
    output = simulate_shadow_fills_to_file(
        shadow_path=Path(args.shadow),
        market_events_path=Path(args.market_events),
        output_path=Path(args.output),
        mode=args.mode,
        constraints=OrderConstraints(
            tick_size=args.tick_size,
            lot_size=args.lot_size,
            min_quantity=args.min_quantity,
            min_notional=args.min_notional,
            max_notional=args.max_notional,
        ),
        latency=LatencyAssumptions(
            order_latency_ms=args.order_latency_ms,
            websocket_delay_ms=args.websocket_delay_ms,
            rate_limit_interval_ms=args.rate_limit_interval_ms,
        ),
        queue=QueueAssumptions(
            queue_ahead_size=args.queue_ahead_size,
            cancellation_rate_per_second=args.cancellation_rate_per_second,
            max_wait_ms=args.max_wait_ms,
            cancel_replace_edge_bps=args.cancel_replace_edge_bps,
        ),
    )
    print(f"simulated_fills={output}")
    return 0


def _cmd_features_to_market_events(args: argparse.Namespace) -> int:
    output = write_market_events_from_feature_csv(Path(args.path), Path(args.output))
    print(f"market_events={output}")
    return 0


def _cmd_baseline(args: argparse.Namespace) -> int:
    with _development_feature_path(args) as feature_path:
        results = run_threshold_baselines(
            feature_path,
            features=_parse_string_list(args.features) if args.features else None,
            thresholds=_parse_float_list(args.thresholds) if args.thresholds else None,
            execution_model=args.execution_model,
            maker_fee_bps=args.maker_fee_bps,
            taker_fee_bps=args.taker_fee_bps,
            slippage_bps=args.slippage_bps,
            sort_by=args.sort_by,
        )
    print(format_baseline_results(results, top=args.top))
    return 0


def _cmd_fee_sweep(args: argparse.Namespace) -> int:
    fees_bps = _parse_float_list(args.fees)
    with _development_feature_path(args) as feature_path:
        sweep = run_fee_sweep(
            feature_path,
            fees_bps=fees_bps,
            features=_parse_string_list(args.features) if args.features else None,
            thresholds=_parse_float_list(args.thresholds) if args.thresholds else None,
            execution_model=args.execution_model,
            maker_fee_bps=args.maker_fee_bps,
            slippage_bps=args.slippage_bps,
            sort_by=args.sort_by,
        )
    print(format_fee_sweep(sweep))
    return 0


def _cmd_walk_forward(args: argparse.Namespace) -> int:
    with _development_feature_path(args) as feature_path:
        folds = run_walk_forward_thresholds(
            feature_path,
            train_size=args.train_size,
            validation_size=args.validation_size,
            test_size=args.test_size,
            step_size=args.step_size,
            purge_label_overlap=not args.no_purge,
            features=_parse_string_list(args.features) if args.features else None,
            thresholds=_parse_float_list(args.thresholds) if args.thresholds else None,
            execution_model=args.execution_model,
            maker_fee_bps=args.maker_fee_bps,
            taker_fee_bps=args.taker_fee_bps,
            slippage_bps=args.slippage_bps,
            sort_by=args.sort_by,
        )
    print(format_walk_forward_results(folds))
    return 0


def _cmd_calendar_walk_forward(args: argparse.Namespace) -> int:
    with _development_feature_path(args) as feature_path:
        folds = run_calendar_walk_forward_thresholds(
            feature_path,
            train_days=args.train_days,
            validation_days=args.validation_days,
            test_days=args.test_days,
            step_days=args.step_days,
            purge_label_overlap=not args.no_purge,
            features=_parse_string_list(args.features) if args.features else None,
            thresholds=_parse_float_list(args.thresholds) if args.thresholds else None,
            execution_model=args.execution_model,
            maker_fee_bps=args.maker_fee_bps,
            taker_fee_bps=args.taker_fee_bps,
            slippage_bps=args.slippage_bps,
            sort_by=args.sort_by,
        )
    print(format_calendar_walk_forward_results(folds))
    return 0


def _cmd_conditional_walk_forward(args: argparse.Namespace) -> int:
    with _development_feature_path(args) as feature_path:
        folds = run_conditional_walk_forward_thresholds(
            feature_path,
            train_size=args.train_size,
            validation_size=args.validation_size,
            test_size=args.test_size,
            step_size=args.step_size,
            purge_label_overlap=not args.no_purge,
            features=_parse_string_list(args.features) if args.features else None,
            thresholds=_parse_float_list(args.thresholds) if args.thresholds else None,
            regime_features=_parse_string_list(args.regime_features) if args.regime_features else None,
            regime_bins=args.regime_bins,
            min_validation_trades=args.min_validation_trades,
            execution_model=args.execution_model,
            maker_fee_bps=args.maker_fee_bps,
            taker_fee_bps=args.taker_fee_bps,
            slippage_bps=args.slippage_bps,
            sort_by=args.sort_by,
        )
    print(format_conditional_walk_forward_results(folds))
    return 0


def _cmd_logistic_walk_forward(args: argparse.Namespace) -> int:
    with _development_feature_path(args) as feature_path:
        folds = run_logistic_walk_forward(
            feature_path,
            train_size=args.train_size,
            validation_size=args.validation_size,
            test_size=args.test_size,
            step_size=args.step_size,
            purge_label_overlap=not args.no_purge,
            features=_parse_string_list(args.features) if args.features else None,
            alpha_thresholds=_parse_float_list(args.alpha_thresholds),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            class_weighting=args.class_weighting,
            execution_model=args.execution_model,
            maker_fee_bps=args.maker_fee_bps,
            taker_fee_bps=args.taker_fee_bps,
            slippage_bps=args.slippage_bps,
            sort_by=args.sort_by,
        )
    print(format_logistic_walk_forward_results(folds))
    return 0


def _cmd_edge_walk_forward(args: argparse.Namespace) -> int:
    runner = run_edge_walk_forward_streaming if args.stream else run_edge_walk_forward
    with _development_feature_path(args) as feature_path:
        if not args.stream:
            assert_csv_load_budget(
                feature_path,
                max_memory_gb=args.max_load_memory_gb,
                multiplier=args.memory_estimate_multiplier,
            )
        folds = runner(
            feature_path,
            train_size=args.train_size,
            validation_size=args.validation_size,
            test_size=args.test_size,
            step_size=args.step_size,
            purge_label_overlap=not args.no_purge,
            features=_parse_string_list(args.features) if args.features else None,
            edge_thresholds_bps=_parse_float_list(args.edge_thresholds_bps),
            l2=args.l2,
            taker_fee_bps=args.taker_fee_bps,
            slippage_bps=args.slippage_bps,
            sort_by=args.sort_by,
            max_folds=args.max_folds,
        )
    print(format_edge_walk_forward_results(folds))
    return 0


def _cmd_edge_shadow_decisions(args: argparse.Namespace) -> int:
    with _development_feature_path(args) as feature_path:
        output = write_edge_shadow_decisions_streaming(
            Path(args.output),
            feature_path,
            venue=args.venue,
            symbol=args.symbol,
            intended_size=args.intended_size,
            intended_notional=args.intended_notional,
            order_type=args.order_type,
            include_flat=args.include_flat,
            train_size=args.train_size,
            validation_size=args.validation_size,
            test_size=args.test_size,
            step_size=args.step_size,
            purge_label_overlap=not args.no_purge,
            features=_parse_string_list(args.features) if args.features else None,
            edge_thresholds_bps=_parse_float_list(args.edge_thresholds_bps),
            l2=args.l2,
            taker_fee_bps=args.taker_fee_bps,
            slippage_bps=args.slippage_bps,
            sort_by=args.sort_by,
            max_folds=args.max_folds,
        )
    print(f"shadow_decisions={output}")
    return 0


def _cmd_eval_rule(args: argparse.Namespace) -> int:
    with _development_feature_path(args) as feature_path:
        evaluation = evaluate_rule_file(
            feature_path,
            feature=args.feature,
            threshold=args.threshold,
            source_date=args.source_date,
            execution_model=args.execution_model,
            maker_fee_bps=args.maker_fee_bps,
            taker_fee_bps=args.taker_fee_bps,
            slippage_bps=args.slippage_bps,
        )
    print(format_rule_evaluation(evaluation))
    return 0


def _cmd_regime(args: argparse.Namespace) -> int:
    with _development_feature_path(args) as feature_path:
        evaluations = run_regime_analysis(
            feature_path,
            feature=args.feature,
            threshold=args.threshold,
            regime_features=_parse_string_list(args.regime_features) if args.regime_features else None,
            bins=args.bins,
            source_date=args.source_date,
            execution_model=args.execution_model,
            maker_fee_bps=args.maker_fee_bps,
            taker_fee_bps=args.taker_fee_bps,
            slippage_bps=args.slippage_bps,
        )
    print(format_regime_analysis(evaluations))
    return 0


def _cmd_final_holdout_rule(args: argparse.Namespace) -> int:
    source_root = holdout_manifest_source_root(args.holdout_manifest)
    if source_root is None:
        raise ValueError("holdout manifest verification failed; create an official manifest from a Git checkout")
    manifest = read_holdout_manifest(args.holdout_manifest)
    candidate_path = Path(args.candidate_json)
    candidate = json.loads(candidate_path.read_text())
    candidate_sha256 = canonical_json_sha256(candidate)
    if not manifest.candidate_sha256:
        raise ValueError(
            "holdout manifest missing pre-registered candidate_sha256; "
            "create it with create-holdout-manifest --candidate-json"
        )
    if candidate_sha256 != manifest.candidate_sha256:
        raise ValueError("frozen candidate hash does not match holdout manifest candidate_sha256")
    allowed_fields = {
        "feature",
        "threshold",
        "execution_model",
        "order_type",
        "maker_fee_bps",
        "taker_fee_bps",
        "slippage_bps",
        "target_notional",
        "initial_cash",
        "max_position_notional",
        "max_leverage",
        "latency_ms",
        "max_order_age_ms",
        "kill_switch_loss",
        "rate_limit_interval_ms",
        "queue_ahead_size",
        "cancellation_rate_per_second",
        "cancel_replace_edge_bps",
        "source_artifact",
        "selection_metric",
        "selection_score",
        "selected_rows",
        "selection_grain",
        "selection_method",
        "validation_net_pnl",
        "test_net_pnl",
        "validation_break_even_fee_bps",
        "test_break_even_fee_bps",
    }
    extra_fields = set(candidate) - allowed_fields
    if extra_fields:
        raise ValueError(f"frozen candidate contains unsupported fields: {', '.join(sorted(extra_fields))}")
    feature = str(candidate["feature"])
    threshold = float(candidate["threshold"])
    execution_model = str(candidate.get("execution_model", "taker"))
    maker_fee_bps = float(candidate.get("maker_fee_bps", 0.0))
    taker_fee_bps = float(candidate.get("taker_fee_bps", 5.0))
    slippage_bps = float(candidate.get("slippage_bps", 0.0))
    if execution_model not in {"taker", "maker_entry"}:
        raise ValueError("frozen candidate execution_model must be taker or maker_entry")
    order_type = str(candidate.get("order_type") or ("passive" if execution_model == "maker_entry" else "market"))
    if order_type not in {"market", "passive"}:
        raise ValueError("frozen candidate order_type must be market or passive")
    initial_cash = float(candidate.get("initial_cash", 1000.0))
    target_notional = float(candidate.get("target_notional", min(100.0, initial_cash * 0.1)))
    rows = read_holdout_rows(Path(args.path), manifest)

    def predictor(row: dict[str, str]) -> int:
        return predict_feature_threshold(row, feature, threshold)

    labels = [int(float(row["label"])) for row in rows]
    predictions = [predictor(row) for row in rows]
    metrics = compute_metrics(labels, predictions)
    simulation = simulate_stateful_execution(
        _market_events_from_feature_rows(rows, target_notional=target_notional),
        _signals_from_candidate_rows(
            rows,
            predictor,
            order_type=order_type,
            target_notional=target_notional,
            feature=feature,
        ),
        config=StatefulExecutionConfig(
            initial_cash=initial_cash,
            max_position_notional=float(
                candidate.get("max_position_notional", max(target_notional * 2.0, target_notional))
            ),
            max_leverage=float(candidate.get("max_leverage", 1.0)),
            taker_fee_bps=taker_fee_bps,
            maker_fee_bps=maker_fee_bps,
            slippage_bps=slippage_bps,
            latency_ms=int(candidate.get("latency_ms", 0)),
            max_order_age_ms=int(candidate.get("max_order_age_ms", 1000)),
            kill_switch_loss=(
                float(candidate["kill_switch_loss"]) if candidate.get("kill_switch_loss") is not None else None
            ),
            rate_limit_interval_ms=int(candidate.get("rate_limit_interval_ms", 0)),
            queue_ahead_size=float(candidate.get("queue_ahead_size", 0.0)),
            cancellation_rate_per_second=float(candidate.get("cancellation_rate_per_second", 0.0)),
            cancel_replace_edge_bps=(
                float(candidate["cancel_replace_edge_bps"])
                if candidate.get("cancel_replace_edge_bps") is not None
                else None
            ),
        ),
    )
    signal_count = sum(1 for prediction in predictions if prediction != 0)
    total_fees = sum(fill.fee for fill in simulation.fills)
    net_pnl = simulation.final_equity - initial_cash
    gross_pnl = net_pnl + total_fees
    break_even_fee_bps = gross_pnl / simulation.turnover * 10_000.0 if simulation.turnover else 0.0
    output = write_final_holdout_result(
        manifest=manifest,
        metrics={
            "candidate": candidate,
            "candidate_sha256": candidate_sha256,
            "rows": metrics.n,
            "accuracy": metrics.accuracy,
            "balanced_accuracy": metrics.balanced_accuracy,
            "macro_f1": metrics.macro_f1,
            "coverage": metrics.coverage,
            "signals": signal_count,
            "trades": len(simulation.fills),
            "fill_rate": len(simulation.fills) / signal_count if signal_count else 0.0,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "final_equity": simulation.final_equity,
            "turnover": simulation.turnover,
            "fees": total_fees,
            "break_even_taker_fee_bps": break_even_fee_bps,
            "stateful_simulator": True,
            "kill_switch_triggered": simulation.kill_switch_triggered,
        },
        output_path=Path(args.output),
        explicit_final_evaluation=args.explicit_final_evaluation,
        candidate_sha256=candidate_sha256,
        lock_dir=Path(args.lock_dir),
        source_root=source_root,
    )
    print(f"final_holdout_result={output}")
    return 0


def _cmd_freeze_threshold_candidate(args: argparse.Namespace) -> int:
    candidate = _freeze_threshold_candidate_payload(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    print(f"candidate_json={output_path}")
    print(f"candidate_sha256={canonical_json_sha256(candidate)}")
    print(f"feature={candidate['feature']} threshold={candidate['threshold']}")
    print(f"selection_metric={candidate['selection_metric']} selection_score={candidate['selection_score']}")
    return 0


def _freeze_threshold_candidate_payload(args: argparse.Namespace) -> dict[str, object]:
    metric_columns = {
        "validation_net_pnl": "val_net_pnl",
        "test_net_pnl": "test_net_pnl",
        "validation_break_even_fee_bps": "val_break_even_fee_bps",
        "test_break_even_fee_bps": "test_break_even_fee_bps",
    }
    metric_column = metric_columns[args.sort_by]
    groups: dict[tuple[str, float], dict[str, float | int]] = {}
    with Path(args.artifact).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = {"feature", "threshold", metric_column} - fieldnames
        if missing:
            raise ValueError(
                "freeze-threshold-candidate requires a threshold-rule artifact with columns: "
                + ", ".join(sorted(missing))
            )
        for row in reader:
            raw_fold = str(row.get("fold", "")).strip()
            if raw_fold == "summary" or not raw_fold:
                continue
            feature = str(row.get("feature", "")).strip()
            name = str(row.get("name", "")).strip()
            if not feature or feature == "constant" or name == "always_flat":
                continue
            threshold = float(row["threshold"])
            key = (feature, threshold)
            bucket = groups.setdefault(
                key,
                {
                    "rows": 0,
                    "selection_score": 0.0,
                    "validation_net_pnl": 0.0,
                    "test_net_pnl": 0.0,
                    "validation_break_even_fee_bps": 0.0,
                    "test_break_even_fee_bps": 0.0,
                },
            )
            bucket["rows"] = int(bucket["rows"]) + 1
            bucket["selection_score"] = float(bucket["selection_score"]) + _csv_float(row.get(metric_column))
            bucket["validation_net_pnl"] = float(bucket["validation_net_pnl"]) + _csv_float(row.get("val_net_pnl"))
            bucket["test_net_pnl"] = float(bucket["test_net_pnl"]) + _csv_float(row.get("test_net_pnl"))
            bucket["validation_break_even_fee_bps"] = float(bucket["validation_break_even_fee_bps"]) + _csv_float(
                row.get("val_break_even_fee_bps")
            )
            bucket["test_break_even_fee_bps"] = float(bucket["test_break_even_fee_bps"]) + _csv_float(
                row.get("test_break_even_fee_bps")
            )
    if not groups:
        raise ValueError("no non-flat threshold-rule candidates found to freeze")
    feature, threshold = max(
        groups,
        key=lambda key: (float(groups[key]["selection_score"]), int(groups[key]["rows"]), key[0], -key[1]),
    )
    selected = groups[(feature, threshold)]
    candidate: dict[str, object] = {
        "feature": feature,
        "threshold": threshold,
        "execution_model": args.execution_model,
        "order_type": args.order_type or ("passive" if args.execution_model == "maker_entry" else "market"),
        "maker_fee_bps": args.maker_fee_bps,
        "taker_fee_bps": args.taker_fee_bps,
        "slippage_bps": args.slippage_bps,
        "source_artifact": str(Path(args.artifact)),
        "selection_metric": args.sort_by,
        "selection_score": float(selected["selection_score"]),
        "selected_rows": int(selected["rows"]),
        "selection_grain": "fold_rows_aggregated_by_feature_threshold",
        "selection_method": "max_aggregate_metric",
        "validation_net_pnl": float(selected["validation_net_pnl"]),
        "test_net_pnl": float(selected["test_net_pnl"]),
        "validation_break_even_fee_bps": float(selected["validation_break_even_fee_bps"]),
        "test_break_even_fee_bps": float(selected["test_break_even_fee_bps"]),
    }
    optional_fields = {
        "target_notional": args.target_notional,
        "initial_cash": args.initial_cash,
        "max_position_notional": args.max_position_notional,
        "max_leverage": args.max_leverage,
        "latency_ms": args.latency_ms,
        "max_order_age_ms": args.max_order_age_ms,
        "kill_switch_loss": args.kill_switch_loss,
        "rate_limit_interval_ms": args.rate_limit_interval_ms,
        "queue_ahead_size": args.queue_ahead_size,
        "cancellation_rate_per_second": args.cancellation_rate_per_second,
        "cancel_replace_edge_bps": args.cancel_replace_edge_bps,
    }
    candidate.update({key: value for key, value in optional_fields.items() if value is not None})
    return candidate


def _csv_float(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _market_events_from_feature_rows(rows: list[dict[str, str]], *, target_notional: float) -> list[MarketEvent]:
    events: list[MarketEvent] = []
    for index, row in enumerate(rows, start=1):
        event_time = _row_time(row, "event_time", fallback=index * 1000)
        future_time = _row_time(row, "future_event_time", fallback=event_time + 1)
        bid = _row_float(row, "entry_bid", "bid")
        ask = _row_float(row, "entry_ask", "ask")
        bid_size = _top_size(row, "bid_qty", price=bid, target_notional=target_notional)
        ask_size = _top_size(row, "ask_qty", price=ask, target_notional=target_notional)
        events.append(
            MarketEvent(
                event_time,
                bid=bid,
                ask=ask,
                bid_size=bid_size,
                ask_size=ask_size,
                trade_side=(row.get("trade_side") or None),
                trade_size=_row_optional_float(row, "trade_size", "trade_qty") or 0.0,
            )
        )
        if future_time > event_time:
            maker_trade_side = _maker_trade_side_from_row(row)
            future_bid = _row_float(row, "future_bid", default=bid)
            future_ask = _row_float(row, "future_ask", default=ask)
            events.append(
                MarketEvent(
                    future_time,
                    bid=future_bid,
                    ask=future_ask,
                    bid_size=bid_size,
                    ask_size=ask_size,
                    trade_side=maker_trade_side,
                    trade_size=max(bid_size, ask_size) if maker_trade_side else 0.0,
                )
            )
    return events


def _signals_from_candidate_rows(
    rows: list[dict[str, str]],
    predictor: Callable[[dict[str, str]], int],
    *,
    order_type: str,
    target_notional: float,
    feature: str,
) -> list[SignalEvent]:
    signals: list[SignalEvent] = []
    for index, row in enumerate(rows, start=1):
        side = predictor(row)
        limit_price = None
        if order_type == "passive" and side != 0:
            limit_price = _row_float(row, "entry_bid", "bid") if side == 1 else _row_float(row, "entry_ask", "ask")
        signals.append(
            SignalEvent(
                _row_time(row, "event_time", fallback=index * 1000),
                target_side=side,
                target_notional=target_notional if side else 0.0,
                order_type=order_type,
                limit_price=limit_price,
                signal_id=f"final-holdout-{index}",
                predicted_edge_bps=abs(_row_optional_float(row, feature) or 0.0),
            )
        )
    return signals


def _row_time(row: dict[str, str], column: str, *, fallback: int) -> int:
    raw = row.get(column)
    if raw is None or raw == "":
        return fallback
    return int(float(raw))


def _row_float(row: dict[str, str], *columns: str, default: float | None = None) -> float:
    value = _row_optional_float(row, *columns)
    if value is not None:
        return value
    if default is not None:
        return default
    raise ValueError(f"feature CSV missing required numeric column: {' or '.join(columns)}")


def _row_optional_float(row: dict[str, str], *columns: str) -> float | None:
    for column in columns:
        raw = row.get(column)
        if raw is not None and raw != "":
            return float(raw)
    return None


def _top_size(row: dict[str, str], column: str, *, price: float, target_notional: float) -> float:
    value = _row_optional_float(row, column)
    if value is not None:
        return max(0.0, value)
    if price <= 0.0:
        return 0.0
    return max(1.0, target_notional / price * 2.0)


def _maker_trade_side_from_row(row: dict[str, str]) -> str | None:
    if _truthy(row.get("maker_long_fillable")):
        return "sell"
    if _truthy(row.get("maker_short_fillable")):
        return "buy"
    return None


def _truthy(value: str | None) -> bool:
    return value in {"1", "true", "True", "yes", "YES"}


def _cmd_fill_diagnostics(args: argparse.Namespace) -> int:
    diagnostics = run_fill_diagnostics(
        Path(args.path),
        feature=args.feature,
        threshold=args.threshold,
        source_date=args.source_date,
        by_source_date=args.by_source_date,
        maker_fee_bps=args.maker_fee_bps,
        taker_fee_bps=args.taker_fee_bps,
        slippage_bps=args.slippage_bps,
    )
    print(format_fill_diagnostics(diagnostics))
    return 0


def _cmd_fill_regime(args: argparse.Namespace) -> int:
    diagnostics = run_fill_regime_diagnostics(
        Path(args.path),
        feature=args.feature,
        threshold=args.threshold,
        regime_features=_parse_string_list(args.regime_features),
        bins=args.bins,
        source_date=args.source_date,
        maker_fee_bps=args.maker_fee_bps,
        taker_fee_bps=args.taker_fee_bps,
        slippage_bps=args.slippage_bps,
    )
    print(format_fill_regime_diagnostics(diagnostics))
    return 0


def _cmd_capacity(args: argparse.Namespace) -> int:
    diagnostics = run_capacity_diagnostics(
        Path(args.path),
        feature=args.feature,
        threshold=args.threshold,
        participation_rate=args.participation_rate,
        top_book_fraction=args.top_book_fraction,
        max_notional=args.max_notional,
        source_date=args.source_date,
        by_source_date=args.by_source_date,
    )
    print(format_capacity_diagnostics(diagnostics))
    return 0


def _cmd_kelly_variance_gate(args: argparse.Namespace) -> int:
    report = evaluate_oos_variance_stability(
        Path(args.path),
        column=args.column,
        min_observations=args.min_observations,
        window_size=args.window_size,
        max_variance_cv=args.max_variance_cv,
    )
    print(format_variance_stability_report(report, output_format=args.format))
    return 0 if report.passed else 1


def _cmd_model_readiness_gate(args: argparse.Namespace) -> int:
    report = evaluate_model_readiness(
        model_name=args.model,
        baseline_audit_path=Path(args.baseline_audit),
        l2_path=Path(args.l2),
        min_fold_count=args.min_fold_count,
        min_l2_rows=args.min_l2_rows,
        require_delta=not args.allow_snapshot_only,
        allow_fi2010=args.allow_fi2010,
        max_l2_rows=args.max_l2_rows,
    )
    print(format_model_readiness_report(report, output_format=args.format))
    return 0 if report.passed else 1


def _cmd_l2_sequence_experiment(args: argparse.Namespace) -> int:
    try:
        report = run_l2_torch_sequence_experiment(
            model_name=args.model,
            l2_path=Path(args.l2),
            baseline_audit_path=Path(args.baseline_audit),
            output_path=Path(args.output),
            depth=args.depth,
            window=args.window,
            label_horizon=args.label_horizon,
            flat_threshold_bps=args.flat_threshold_bps,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            early_stopping_patience=args.early_stopping_patience,
            device=args.device,
            class_weighting=args.class_weighting,
            lr_scheduler_gamma=args.lr_scheduler_gamma,
            checkpoint_path=args.checkpoint_path,
            resume_from_checkpoint=args.resume_from_checkpoint,
            prediction_output_path=args.predictions_output,
            holdout_manifest_path=args.holdout_manifest,
            development_l2_output_path=args.development_l2_output,
            economic_target_notional=args.economic_target_notional,
            economic_taker_fee_bps=args.economic_taker_fee_bps,
            economic_slippage_bps=args.economic_slippage_bps,
            max_rows=args.max_rows,
            max_snapshots=args.max_snapshots,
            min_fold_count=args.min_fold_count,
            min_l2_rows=args.min_l2_rows,
            seed=args.seed,
        )
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(format_l2_sequence_experiment_report(report, output_format=args.format))
    return 0 if report.passed else 1


def _cmd_l2_pretraining_smoke(args: argparse.Namespace) -> int:
    report = run_l2_masked_pretraining_smoke(
        l2_path=Path(args.l2),
        output_path=Path(args.output),
        depth=args.depth,
        window=args.window,
        mask_probability=args.mask_probability,
        max_rows=args.max_rows,
        max_snapshots=args.max_snapshots,
    )
    print(format_l2_masked_pretraining_report(report, output_format=args.format))
    return 0 if report.passed else 1


def _cmd_audit_results(args: argparse.Namespace) -> int:
    audit = audit_result_artifact(
        Path(args.path),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    criteria = AcceptanceCriteria(
        min_fold_count=args.min_fold_count,
        min_positive_fold_rate=args.min_positive_fold_rate,
        max_positive_fold_share_of_total_net=args.max_fold_contribution,
        require_positive_total_net_pnl=not args.allow_nonpositive_total,
        require_positive_median_fold=not args.allow_nonpositive_median,
        min_break_even_fee_bps=args.assumed_cost_bps * args.cost_safety_multiple,
        require_positive_bootstrap_lower_bound=args.require_positive_bootstrap_lower_bound,
    )
    verdict = evaluate_acceptance(audit, criteria)
    if args.format == "markdown":
        print(format_result_audit_markdown(audit, verdict))
    else:
        print(format_result_audit_csv(audit, verdict))
    return 0


def _cmd_pvalue_correction(args: argparse.Namespace) -> int:
    records = read_p_value_records(
        Path(args.path),
        id_column=args.id_column,
        p_value_column=args.p_value_column,
        metric_column=args.metric_column,
    )
    corrections = correct_p_values(records, q=args.q)
    print(format_p_value_corrections(corrections))
    return 0


def _parse_float_list(raw: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise SystemExit("expected at least one comma-separated float")
    return values


def _parse_string_list(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise SystemExit("expected at least one comma-separated value")
    return values


def _looks_like_bybit_data_zip(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".data.zip") or ("_ob" in name and name.endswith(".zip"))


def _print_datasets() -> None:
    print("V1 primary market: futures/um")
    print("V1 symbols: BTCUSDT, ETHUSDT")
    print("Primary datasets:")
    print("  bookTicker: best bid/ask updates for mid, spread, top-of-book labels")
    print("  trades: raw trade flow for volume, side proxy, aggressor pressure")
    print("  aggTrades: compact trade flow alternative")
    print("  bookDepth: aggregate depth bands, not full L2 price levels")
    print("")
    print("Critical limitation:")
    print("  Binance Vision bookDepth is not full level-by-level order book data.")
    print("  True deep LOB claims require live depth capture or another historical L2 source.")


if __name__ == "__main__":
    raise SystemExit(main())
