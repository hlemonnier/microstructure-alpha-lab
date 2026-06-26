from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from lob_forge.binance_vision import iter_dates
from lob_forge.edge_model import DEFAULT_EDGE_THRESHOLDS_BPS
from lob_forge.memory_guard import physical_memory_gb


@dataclass(frozen=True)
class ExpectedEdgeRunPlan:
    profile: str
    start_date: str
    end_date: str
    symbols: list[str]
    horizons_ms: list[int]
    fees_bps: list[float]
    latency_ms: int
    total_days: int
    binance_daily_archives: int
    feature_build_jobs: int
    edge_eval_jobs: int
    max_quote_buckets: int | None
    capped_rows_per_daily_feature: int | None
    max_combined_rows_per_symbol_horizon: int | None
    with_book_depth: bool
    train_size: int
    validation_size: int
    test_size: int
    step_size: int
    edge_streaming: bool
    edge_thresholds_bps: list[float]
    threshold_candidate_attempts: int
    model_classes: list[str]
    feature_sets: list[str]
    selection_metric: str
    min_ram_gb: float
    max_csv_load_memory_gb: float
    max_feature_build_memory_gb: float
    physical_ram_gb: float | None
    ram_preflight_passed: bool | None
    can_start_on_current_machine: bool
    risk_level: str
    out_dir: str
    processed_root: str
    raw_root: str
    recommendations: list[str]


def build_expected_edge_run_plan(
    *,
    profile: str,
    start_date: str,
    end_date: str,
    symbols: Sequence[str],
    horizons_ms: Sequence[int],
    fees_bps: Sequence[float],
    latency_ms: int,
    max_quote_buckets: int | None,
    train_size: int,
    validation_size: int,
    test_size: int,
    step_size: int,
    edge_streaming: bool,
    min_ram_gb: float,
    max_csv_load_memory_gb: float,
    out_dir: str,
    processed_root: str,
    raw_root: str,
    physical_ram_gb_value: float | None = None,
    with_book_depth: bool = True,
    max_feature_build_memory_gb: float = 0.0,
    edge_thresholds_bps: Sequence[float] | None = None,
    model_classes: Sequence[str] = ("ridge_expected_edge",),
    feature_sets: Sequence[str] = ("default_microstructure",),
    selection_metric: str = "validation_net_pnl",
) -> ExpectedEdgeRunPlan:
    normalized_profile = _normalize_profile(profile)
    clean_symbols = [symbol.strip().upper() for symbol in symbols if symbol.strip()]
    if not clean_symbols:
        raise ValueError("study plan needs at least one symbol")
    if not horizons_ms:
        raise ValueError("study plan needs at least one horizon")
    if not fees_bps:
        raise ValueError("study plan needs at least one fee level")
    clean_edge_thresholds = (
        list(edge_thresholds_bps) if edge_thresholds_bps is not None else list(DEFAULT_EDGE_THRESHOLDS_BPS)
    )
    if not clean_edge_thresholds:
        raise ValueError("study plan needs at least one edge threshold")
    clean_model_classes = [value.strip() for value in model_classes if value.strip()]
    clean_feature_sets = [value.strip() for value in feature_sets if value.strip()]
    if not clean_model_classes:
        raise ValueError("study plan needs at least one model class")
    if not clean_feature_sets:
        raise ValueError("study plan needs at least one feature set")

    dates = list(iter_dates(start_date, end_date))
    total_days = len(dates)
    feature_build_jobs = len(clean_symbols) * len(horizons_ms)
    edge_eval_jobs = feature_build_jobs * len(fees_bps)
    threshold_candidate_attempts = (
        edge_eval_jobs * len(clean_edge_thresholds) * len(clean_model_classes) * len(clean_feature_sets)
    )
    datasets_per_day = 3 if with_book_depth else 2
    binance_daily_archives = total_days * len(clean_symbols) * datasets_per_day
    physical_ram = physical_memory_gb() if physical_ram_gb_value is None else physical_ram_gb_value
    ram_preflight_passed = None if physical_ram is None else physical_ram >= min_ram_gb
    capped_rows = max_quote_buckets
    max_combined_rows = total_days * max_quote_buckets if max_quote_buckets is not None else None
    risk_level = _risk_level(normalized_profile)
    can_start = ram_preflight_passed is not False
    recommendations = _recommendations(
        profile=normalized_profile,
        edge_streaming=edge_streaming,
        max_quote_buckets=max_quote_buckets,
        min_ram_gb=min_ram_gb,
        max_csv_load_memory_gb=max_csv_load_memory_gb,
        max_feature_build_memory_gb=max_feature_build_memory_gb,
        physical_ram_gb_value=physical_ram,
        ram_preflight_passed=ram_preflight_passed,
    )

    return ExpectedEdgeRunPlan(
        profile=normalized_profile,
        start_date=start_date,
        end_date=end_date,
        symbols=clean_symbols,
        horizons_ms=list(horizons_ms),
        fees_bps=list(fees_bps),
        latency_ms=latency_ms,
        total_days=total_days,
        binance_daily_archives=binance_daily_archives,
        feature_build_jobs=feature_build_jobs,
        edge_eval_jobs=edge_eval_jobs,
        max_quote_buckets=max_quote_buckets,
        capped_rows_per_daily_feature=capped_rows,
        max_combined_rows_per_symbol_horizon=max_combined_rows,
        with_book_depth=with_book_depth,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        step_size=step_size,
        edge_streaming=edge_streaming,
        edge_thresholds_bps=clean_edge_thresholds,
        threshold_candidate_attempts=threshold_candidate_attempts,
        model_classes=clean_model_classes,
        feature_sets=clean_feature_sets,
        selection_metric=selection_metric,
        min_ram_gb=min_ram_gb,
        max_csv_load_memory_gb=max_csv_load_memory_gb,
        max_feature_build_memory_gb=max_feature_build_memory_gb,
        physical_ram_gb=physical_ram,
        ram_preflight_passed=ram_preflight_passed,
        can_start_on_current_machine=can_start,
        risk_level=risk_level,
        out_dir=out_dir,
        processed_root=processed_root,
        raw_root=raw_root,
        recommendations=recommendations,
    )


def write_expected_edge_run_plan(plan: ExpectedEdgeRunPlan, path: Path | str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(asdict(plan), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def format_expected_edge_run_plan(plan: ExpectedEdgeRunPlan) -> str:
    ram = "unknown" if plan.physical_ram_gb is None else f"{plan.physical_ram_gb:.1f}GB"
    combined_rows = (
        "uncapped"
        if plan.max_combined_rows_per_symbol_horizon is None
        else str(plan.max_combined_rows_per_symbol_horizon)
    )
    lines = [
        f"profile={plan.profile} risk={plan.risk_level} dates={plan.start_date}..{plan.end_date} days={plan.total_days}",
        f"symbols={','.join(plan.symbols)} horizons_ms={','.join(str(value) for value in plan.horizons_ms)} latency_ms={plan.latency_ms} fees={len(plan.fees_bps)}",
        f"archives={plan.binance_daily_archives} feature_jobs={plan.feature_build_jobs} edge_jobs={plan.edge_eval_jobs} threshold_candidate_attempts={plan.threshold_candidate_attempts}",
        f"edge_thresholds_bps={','.join(format(value, 'g') for value in plan.edge_thresholds_bps)} selection_metric={plan.selection_metric}",
        f"with_book_depth={int(plan.with_book_depth)} edge_streaming={int(plan.edge_streaming)} max_combined_rows_per_symbol_horizon={combined_rows}",
        f"physical_ram={ram} min_ram_gb={plan.min_ram_gb:.1f} can_start={int(plan.can_start_on_current_machine)}",
    ]
    for recommendation in plan.recommendations:
        lines.append(f"recommendation={recommendation}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a run plan for the expected-edge study profiles.")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--horizons-ms", required=True)
    parser.add_argument("--fees-bps", required=True)
    parser.add_argument("--latency-ms", type=int, required=True)
    parser.add_argument("--max-quote-buckets", default="")
    parser.add_argument("--with-book-depth", choices=["0", "1"], default="1")
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--validation-size", type=int, required=True)
    parser.add_argument("--test-size", type=int, required=True)
    parser.add_argument("--step-size", type=int, required=True)
    parser.add_argument("--edge-streaming", choices=["0", "1"], required=True)
    parser.add_argument("--edge-thresholds-bps", default="")
    parser.add_argument("--model-classes", default="ridge_expected_edge")
    parser.add_argument("--feature-sets", default="default_microstructure")
    parser.add_argument("--selection-metric", default="validation_net_pnl")
    parser.add_argument("--min-ram-gb", type=float, required=True)
    parser.add_argument("--max-load-memory-gb", type=float, required=True)
    parser.add_argument("--max-feature-build-memory-gb", type=float, default=0.0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    plan = build_expected_edge_run_plan(
        profile=args.profile,
        start_date=args.start,
        end_date=args.end,
        symbols=args.symbols.split(),
        horizons_ms=[int(value) for value in args.horizons_ms.split()],
        fees_bps=[float(value) for value in args.fees_bps.split()],
        latency_ms=args.latency_ms,
        max_quote_buckets=int(args.max_quote_buckets) if args.max_quote_buckets else None,
        with_book_depth=args.with_book_depth == "1",
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        step_size=args.step_size,
        edge_streaming=args.edge_streaming == "1",
        min_ram_gb=args.min_ram_gb,
        max_csv_load_memory_gb=args.max_load_memory_gb,
        max_feature_build_memory_gb=args.max_feature_build_memory_gb,
        out_dir=args.out_dir,
        processed_root=args.processed_root,
        raw_root=args.raw_root,
        edge_thresholds_bps=_parse_number_list(args.edge_thresholds_bps, default=DEFAULT_EDGE_THRESHOLDS_BPS),
        model_classes=_parse_string_list(args.model_classes),
        feature_sets=_parse_string_list(args.feature_sets),
        selection_metric=args.selection_metric,
    )
    if args.output:
        output_path = write_expected_edge_run_plan(plan, args.output)
        print(f"run_plan={output_path}")
    print(format_expected_edge_run_plan(plan))
    return 0


def _normalize_profile(profile: str) -> str:
    return {
        "local16": "local16_60day",
        "full": "cloud_full",
    }.get(profile, profile)


def _risk_level(profile: str) -> str:
    if profile in {"laptop_tiny", "laptop_quick"}:
        return "laptop_safe"
    if profile == "local16_60day":
        return "capped_local_data_heavy"
    if profile == "cloud_full":
        return "cloud_full_heavy"
    return "custom"


def _parse_number_list(value: str, *, default: Sequence[float]) -> list[float]:
    if not value.strip():
        return list(default)
    return [float(item) for item in value.replace(",", " ").split() if item]


def _parse_string_list(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def _recommendations(
    *,
    profile: str,
    edge_streaming: bool,
    max_quote_buckets: int | None,
    min_ram_gb: float,
    max_csv_load_memory_gb: float,
    max_feature_build_memory_gb: float,
    physical_ram_gb_value: float | None,
    ram_preflight_passed: bool | None,
) -> list[str]:
    recommendations: list[str] = []
    if profile == "laptop_tiny":
        recommendations.append(
            "tiny laptop profile is for proof-of-pipeline only; use cloud for serious 60-day evidence"
        )
    if profile == "cloud_full":
        recommendations.append("run on 64-128GB RAM cloud machine; do not run full profile on a 16GB laptop")
    if profile == "local16_60day":
        recommendations.append("local profile is RAM-capped but still network/disk heavy; prefer PLAN_ONLY first")
    if not edge_streaming:
        recommendations.append("EDGE_STREAMING=0 can load large CSVs; keep streaming enabled unless on high-RAM cloud")
    if max_quote_buckets is None:
        recommendations.append("daily feature builds are uncapped; expect larger disk, runtime, and network load")
    if max_csv_load_memory_gb > min_ram_gb and not edge_streaming:
        recommendations.append("max CSV load budget exceeds RAM gate; lower MAX_LOAD_MEMORY_GB or enable streaming")
    if max_feature_build_memory_gb > min_ram_gb:
        recommendations.append("feature-build memory budget exceeds RAM gate; lower MAX_FEATURE_BUILD_MEMORY_GB")
    if ram_preflight_passed is False and physical_ram_gb_value is not None:
        recommendations.append(
            f"current machine has {physical_ram_gb_value:.1f}GB RAM, below required {min_ram_gb:.1f}GB"
        )
    return recommendations


if __name__ == "__main__":
    raise SystemExit(main())
