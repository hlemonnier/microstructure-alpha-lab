from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from lob_forge.edge_model import DEFAULT_EDGE_THRESHOLDS_BPS


@dataclass(frozen=True)
class ExpectedEdgeCandidateAttempt:
    run_id: str
    family: str
    symbol: str
    horizon_ms: int
    taker_fee_bps: float
    latency_ms: int
    edge_threshold_bps: float
    model_class: str
    feature_set: str
    selection_metric: str
    train_size: int
    validation_size: int
    test_size: int
    step_size: int
    status: str
    selected: bool
    selected_fold_count: int
    fold_count: int
    validation_net_pnl: float | None
    test_net_pnl: float | None
    audit_acceptance_passed: bool | None
    audit_rejection_reasons: str
    artifact_path: str
    audit_path: str
    failure_reason: str
    config_sha256: str
    config_json: str


@dataclass(frozen=True)
class _ArtifactSummary:
    result_exists: bool
    audit_exists: bool
    fold_count: int
    selected_fold_counts: dict[float, int]
    validation_net_pnl: dict[float, float]
    test_net_pnl: dict[float, float]
    audit_acceptance_passed: bool | None
    audit_rejection_reasons: str
    failure_reason: str


def build_expected_edge_candidate_registry(
    *,
    plan_path: Path | str,
    result_dir: Path | str | None = None,
) -> list[ExpectedEdgeCandidateAttempt]:
    plan_path = Path(plan_path)
    plan = _read_plan(plan_path)
    result_dir_path = Path(result_dir) if result_dir is not None else Path(plan.get("out_dir", plan_path.parent))
    run_id = _run_id(plan, result_dir_path)
    thresholds = _float_list(plan.get("edge_thresholds_bps"), default=DEFAULT_EDGE_THRESHOLDS_BPS)
    model_classes = _string_list(plan.get("model_classes"), default=("ridge_expected_edge",))
    feature_sets = _string_list(plan.get("feature_sets"), default=("default_microstructure",))
    selection_metric = str(plan.get("selection_metric") or "validation_net_pnl")
    attempts: list[ExpectedEdgeCandidateAttempt] = []
    summary_cache: dict[tuple[str, int, float], _ArtifactSummary] = {}

    for symbol in _string_list(plan.get("symbols"), default=()):
        normalized_symbol = symbol.upper()
        for horizon_ms in _int_list(plan.get("horizons_ms")):
            for fee_bps in _float_list(plan.get("fees_bps"), default=()):
                result_path, audit_path = _expected_result_paths(
                    result_dir=result_dir_path,
                    symbol=normalized_symbol,
                    horizon_ms=horizon_ms,
                    fee_bps=fee_bps,
                )
                cache_key = (normalized_symbol, horizon_ms, fee_bps)
                summary = summary_cache.get(cache_key)
                if summary is None:
                    summary = _summarize_artifacts(result_path=result_path, audit_path=audit_path)
                    summary_cache[cache_key] = summary
                for model_class in model_classes:
                    for feature_set in feature_sets:
                        for threshold in thresholds:
                            selected_fold_count = summary.selected_fold_counts.get(threshold, 0)
                            selected = selected_fold_count > 0
                            config = {
                                "family": "expected_edge_threshold_grid",
                                "symbol": normalized_symbol,
                                "horizon_ms": horizon_ms,
                                "taker_fee_bps": fee_bps,
                                "latency_ms": int(plan.get("latency_ms", 0)),
                                "edge_threshold_bps": threshold,
                                "model_class": model_class,
                                "feature_set": feature_set,
                                "selection_metric": selection_metric,
                                "train_size": int(plan.get("train_size", 0)),
                                "validation_size": int(plan.get("validation_size", 0)),
                                "test_size": int(plan.get("test_size", 0)),
                                "step_size": int(plan.get("step_size", 0)),
                            }
                            config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
                            attempts.append(
                                ExpectedEdgeCandidateAttempt(
                                    run_id=run_id,
                                    family="expected_edge_threshold_grid",
                                    symbol=normalized_symbol,
                                    horizon_ms=horizon_ms,
                                    taker_fee_bps=fee_bps,
                                    latency_ms=int(plan.get("latency_ms", 0)),
                                    edge_threshold_bps=threshold,
                                    model_class=model_class,
                                    feature_set=feature_set,
                                    selection_metric=selection_metric,
                                    train_size=int(plan.get("train_size", 0)),
                                    validation_size=int(plan.get("validation_size", 0)),
                                    test_size=int(plan.get("test_size", 0)),
                                    step_size=int(plan.get("step_size", 0)),
                                    status=_candidate_status(summary, selected=selected),
                                    selected=selected,
                                    selected_fold_count=selected_fold_count,
                                    fold_count=summary.fold_count,
                                    validation_net_pnl=summary.validation_net_pnl.get(threshold),
                                    test_net_pnl=summary.test_net_pnl.get(threshold),
                                    audit_acceptance_passed=summary.audit_acceptance_passed,
                                    audit_rejection_reasons=summary.audit_rejection_reasons,
                                    artifact_path=str(result_path) if summary.result_exists else "",
                                    audit_path=str(audit_path) if summary.audit_exists else "",
                                    failure_reason=_candidate_failure_reason(summary),
                                    config_sha256=hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
                                    config_json=config_json,
                                )
                            )
    return attempts


def write_expected_edge_candidate_registry(
    attempts: Sequence[ExpectedEdgeCandidateAttempt],
    path: Path | str,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for attempt in attempts:
            handle.write(json.dumps(asdict(attempt), sort_keys=True) + "\n")
    return output_path


def read_expected_edge_candidate_registry(path: Path | str) -> list[ExpectedEdgeCandidateAttempt]:
    attempts: list[ExpectedEdgeCandidateAttempt] = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                attempts.append(ExpectedEdgeCandidateAttempt(**json.loads(line)))
    return attempts


def format_expected_edge_candidate_registry(attempts: Sequence[ExpectedEdgeCandidateAttempt]) -> str:
    statuses: dict[str, int] = {}
    for attempt in attempts:
        statuses[attempt.status] = statuses.get(attempt.status, 0) + 1
    lines = [f"candidate_attempts={len(attempts)}"]
    for status, count in sorted(statuses.items()):
        lines.append(f"status_{status}={count}")
    lines.append(f"selected_candidates={sum(1 for attempt in attempts if attempt.selected)}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write an expected-edge candidate registry from a run plan.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--result-dir")
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    attempts = build_expected_edge_candidate_registry(plan_path=args.plan, result_dir=args.result_dir)
    output = (
        Path(args.output)
        if args.output
        else Path(args.result_dir or Path(args.plan).parent) / "candidate_registry.jsonl"
    )
    output_path = write_expected_edge_candidate_registry(attempts, output)
    print(f"candidate_registry={output_path}")
    print(format_expected_edge_candidate_registry(attempts))
    return 0


def _read_plan(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"run plan must be a JSON object: {path}")
    return payload


def _run_id(plan: dict[str, Any], result_dir: Path) -> str:
    if plan.get("profile") and plan.get("start_date") and plan.get("end_date"):
        return f"{plan['profile']}_{str(plan['start_date']).replace('-', '')}_{str(plan['end_date']).replace('-', '')}"
    return result_dir.name or "expected_edge_study"


def _expected_result_paths(*, result_dir: Path, symbol: str, horizon_ms: int, fee_bps: float) -> tuple[Path, Path]:
    stem = f"{symbol}_{horizon_ms}ms_fee_{_fee_token(fee_bps)}_edge"
    return result_dir / f"{stem}.csv", result_dir / f"{stem}_audit.csv"


def _fee_token(fee_bps: float) -> str:
    if fee_bps.is_integer():
        value = str(int(fee_bps))
    else:
        value = format(fee_bps, "g")
    return value.replace(".", "p")


def _summarize_artifacts(*, result_path: Path, audit_path: Path) -> _ArtifactSummary:
    result_exists = result_path.exists() and result_path.stat().st_size > 0
    audit_exists = audit_path.exists() and audit_path.stat().st_size > 0
    fold_count = 0
    selected_fold_counts: dict[float, int] = {}
    validation_net_pnl: dict[float, float] = {}
    test_net_pnl: dict[float, float] = {}
    failure_reasons: list[str] = []

    if result_exists:
        try:
            with result_path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if not str(row.get("fold", "")).isdigit():
                        continue
                    fold_count += 1
                    threshold = float(row["edge_threshold_bps"])
                    selected_fold_counts[threshold] = selected_fold_counts.get(threshold, 0) + 1
                    validation_net_pnl[threshold] = validation_net_pnl.get(threshold, 0.0) + _float_cell(
                        row.get("val_net_pnl")
                    )
                    test_net_pnl[threshold] = test_net_pnl.get(threshold, 0.0) + _float_cell(row.get("test_net_pnl"))
        except Exception as exc:
            failure_reasons.append(f"result_parse_error={type(exc).__name__}: {exc}")

    audit_acceptance_passed: bool | None = None
    audit_rejection_reasons = ""
    if audit_exists:
        try:
            with audit_path.open(newline="") as handle:
                audit_row = next(csv.DictReader(handle), None)
            if audit_row is not None:
                audit_acceptance_passed = str(audit_row.get("acceptance_passed", "")).strip() == "1"
                audit_rejection_reasons = str(audit_row.get("rejection_reasons", ""))
            else:
                failure_reasons.append("audit_parse_error=empty_csv")
        except Exception as exc:
            failure_reasons.append(f"audit_parse_error={type(exc).__name__}: {exc}")

    return _ArtifactSummary(
        result_exists=result_exists,
        audit_exists=audit_exists,
        fold_count=fold_count,
        selected_fold_counts=selected_fold_counts,
        validation_net_pnl=validation_net_pnl,
        test_net_pnl=test_net_pnl,
        audit_acceptance_passed=audit_acceptance_passed,
        audit_rejection_reasons=audit_rejection_reasons,
        failure_reason="; ".join(failure_reasons),
    )


def _candidate_status(summary: _ArtifactSummary, *, selected: bool) -> str:
    if summary.failure_reason:
        return "artifact_error"
    if not summary.result_exists and not summary.audit_exists:
        return "planned"
    if not summary.result_exists or not summary.audit_exists:
        return "incomplete_artifact"
    return "selected_in_artifact" if selected else "evaluated_unselected"


def _candidate_failure_reason(summary: _ArtifactSummary) -> str:
    if summary.failure_reason:
        return summary.failure_reason
    if not summary.result_exists and not summary.audit_exists:
        return ""
    missing = []
    if not summary.result_exists:
        missing.append("result")
    if not summary.audit_exists:
        missing.append("audit")
    return f"missing {','.join(missing)}" if missing else ""


def _float_cell(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _string_list(value: Any, *, default: Sequence[str]) -> list[str]:
    if value is None or value == "":
        return [item for item in default if item]
    if isinstance(value, str):
        return [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _float_list(value: Any, *, default: Sequence[float]) -> list[float]:
    if value is None or value == "":
        return list(default)
    if isinstance(value, str):
        return [float(item) for item in value.replace(",", " ").split() if item]
    return [float(item) for item in value]


def _int_list(value: Any) -> list[int]:
    if isinstance(value, str):
        return [int(item) for item in value.replace(",", " ").split() if item]
    return [int(item) for item in value]


if __name__ == "__main__":
    raise SystemExit(main())
