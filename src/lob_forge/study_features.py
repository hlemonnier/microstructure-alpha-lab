from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from lob_forge.binance_vision import iter_dates


@dataclass(frozen=True)
class FeatureJobCoverage:
    symbol: str
    horizon_ms: int
    latency_ms: int
    symbol_dir: str
    combined_path: str
    expected_daily_markers: int
    present_daily_markers: int
    missing_daily_markers: list[str]
    combined_present: bool
    complete: bool


@dataclass(frozen=True)
class ExpectedEdgeFeatureStatus:
    plan_path: Path
    processed_root: Path
    profile: str
    expected_feature_jobs: int
    complete_feature_jobs: int
    expected_daily_markers: int
    present_daily_markers: int
    missing_daily_markers: list[str]
    missing_combined_files: list[str]
    feature_jobs: list[FeatureJobCoverage]
    complete: bool


def evaluate_expected_edge_feature_status(
    *,
    plan_path: Path | str,
    processed_root: Path | str | None = None,
) -> ExpectedEdgeFeatureStatus:
    plan_path = Path(plan_path)
    if not plan_path.exists():
        raise FileNotFoundError(f"missing run plan: {plan_path}")
    plan = _read_plan(plan_path)
    processed_root_path = (
        Path(processed_root) if processed_root is not None else Path(plan.get("processed_root", "data/processed"))
    )
    symbols = [str(symbol).upper() for symbol in plan.get("symbols", [])]
    horizons_ms = [int(horizon) for horizon in plan.get("horizons_ms", [])]
    start_date = str(plan["start_date"])
    end_date = str(plan["end_date"])
    latency_ms = int(plan.get("latency_ms", 1000))
    dates = list(iter_dates(start_date, end_date))

    missing_daily: list[str] = []
    missing_combined: list[str] = []
    present_daily = 0
    complete_jobs = 0
    feature_jobs: list[FeatureJobCoverage] = []
    for symbol in symbols:
        symbol_lower = symbol.lower()
        for horizon_ms in horizons_ms:
            symbol_dir = processed_root_path / f"{symbol_lower}_{horizon_ms}ms_latency_{latency_ms}"
            job_daily_missing = 0
            job_present_daily = 0
            job_missing_daily: list[str] = []
            for date_value in dates:
                marker = symbol_dir / f"{symbol}-{date_value}-quote-trade-features.csv.done"
                if marker.exists():
                    present_daily += 1
                    job_present_daily += 1
                else:
                    job_daily_missing += 1
                    missing_path = str(marker)
                    missing_daily.append(missing_path)
                    job_missing_daily.append(missing_path)
            combined = symbol_dir / f"{symbol}-{start_date}_{end_date}-combined-features.csv"
            combined_present = combined.exists() and combined.stat().st_size > 0
            if not combined_present:
                missing_combined.append(str(combined))
            job_complete = job_daily_missing == 0 and combined_present
            feature_jobs.append(
                FeatureJobCoverage(
                    symbol=symbol,
                    horizon_ms=horizon_ms,
                    latency_ms=latency_ms,
                    symbol_dir=str(symbol_dir),
                    combined_path=str(combined),
                    expected_daily_markers=len(dates),
                    present_daily_markers=job_present_daily,
                    missing_daily_markers=job_missing_daily,
                    combined_present=combined_present,
                    complete=job_complete,
                )
            )
            if job_complete:
                complete_jobs += 1

    expected_jobs = len(symbols) * len(horizons_ms)
    expected_daily = expected_jobs * len(dates)
    complete = expected_jobs > 0 and complete_jobs == expected_jobs and not missing_daily and not missing_combined

    return ExpectedEdgeFeatureStatus(
        plan_path=plan_path,
        processed_root=processed_root_path,
        profile=str(plan.get("profile", "unknown")),
        expected_feature_jobs=expected_jobs,
        complete_feature_jobs=complete_jobs,
        expected_daily_markers=expected_daily,
        present_daily_markers=present_daily,
        missing_daily_markers=missing_daily,
        missing_combined_files=missing_combined,
        feature_jobs=feature_jobs,
        complete=complete,
    )


def write_expected_edge_feature_status(status: ExpectedEdgeFeatureStatus, path: Path | str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(status)
    payload["plan_path"] = str(status.plan_path)
    payload["processed_root"] = str(status.processed_root)
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def format_expected_edge_feature_status(status: ExpectedEdgeFeatureStatus) -> str:
    lines = [
        f"plan_path={status.plan_path}",
        f"processed_root={status.processed_root}",
        f"profile={status.profile}",
        f"expected_feature_jobs={status.expected_feature_jobs}",
        f"complete_feature_jobs={status.complete_feature_jobs}",
        f"expected_daily_markers={status.expected_daily_markers}",
        f"present_daily_markers={status.present_daily_markers}",
        f"missing_daily_markers={len(status.missing_daily_markers)}",
        f"missing_combined_files={len(status.missing_combined_files)}",
        f"complete={int(status.complete)}",
    ]
    for path in status.missing_combined_files[:20]:
        lines.append(f"missing_combined={path}")
    if len(status.missing_combined_files) > 20:
        lines.append(f"missing_combined_more={len(status.missing_combined_files) - 20}")
    for path in status.missing_daily_markers[:20]:
        lines.append(f"missing_daily={path}")
    if len(status.missing_daily_markers) > 20:
        lines.append(f"missing_daily_more={len(status.missing_daily_markers) - 20}")
    for job in status.feature_jobs:
        lines.append(
            "feature_job="
            f"symbol={job.symbol},horizon_ms={job.horizon_ms},complete={int(job.complete)},"
            f"present_daily={job.present_daily_markers}/{job.expected_daily_markers},"
            f"combined_present={int(job.combined_present)}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify daily and combined feature coverage from run_plan.json.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--processed-root")
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    status = evaluate_expected_edge_feature_status(
        plan_path=args.plan,
        processed_root=args.processed_root,
    )
    if args.output:
        output_path = write_expected_edge_feature_status(status, args.output)
        print(f"feature_status={output_path}")
    print(format_expected_edge_feature_status(status))
    return 0 if status.complete else 1


def _read_plan(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"run plan must be a JSON object: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
