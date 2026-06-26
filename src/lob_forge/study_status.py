from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from lob_forge.result_verifier import verify_result_artifacts
from lob_forge.study_registry import read_expected_edge_candidate_registry


@dataclass(frozen=True)
class ExpectedEdgeStudyStatus:
    result_dir: Path
    plan_path: Path
    profile: str
    expected_edge_jobs: int
    present_result_files: int
    present_audit_files: int
    missing_result_files: list[str]
    missing_audit_files: list[str]
    missing_required_files: list[str]
    artifact_verifier_passed: bool
    artifact_verifier_errors: list[str]
    complete: bool


def evaluate_expected_edge_study_status(
    *,
    plan_path: Path | str,
    result_dir: Path | str | None = None,
    require_pvalues: bool = True,
    min_audit_fold_count: int = 1,
) -> ExpectedEdgeStudyStatus:
    plan_path = Path(plan_path)
    if not plan_path.exists():
        raise FileNotFoundError(f"missing run plan: {plan_path}")
    plan = _read_plan(plan_path)
    result_dir_path = Path(result_dir) if result_dir is not None else Path(plan.get("out_dir", plan_path.parent))

    expected_pairs = _expected_result_pairs(plan)
    missing_results: list[str] = []
    missing_audits: list[str] = []
    present_results = 0
    present_audits = 0
    for result_name, audit_name in expected_pairs:
        result_path = result_dir_path / result_name
        audit_path = result_dir_path / audit_name
        if result_path.exists() and result_path.stat().st_size > 0:
            present_results += 1
        else:
            missing_results.append(result_name)
        if audit_path.exists() and audit_path.stat().st_size > 0:
            present_audits += 1
        else:
            missing_audits.append(audit_name)

    missing_required: list[str] = []
    required_files = ["candidate_registry.jsonl"]
    if require_pvalues:
        required_files.extend(["pvalues.csv", "pvalue_corrections.csv"])
    for filename in required_files:
        path = result_dir_path / filename
        if not path.exists() or path.stat().st_size == 0:
            missing_required.append(filename)

    if min_audit_fold_count <= 0:
        raise ValueError("min_audit_fold_count must be positive")

    verifier = verify_result_artifacts(result_dir_path, strict_metadata=False)
    verifier_errors = _filter_verifier_errors(verifier.errors, require_pvalues=require_pvalues)
    verifier_errors = tuple(verifier_errors) + tuple(
        _audit_fold_count_errors(
            result_dir=result_dir_path,
            expected_pairs=expected_pairs,
            min_audit_fold_count=min_audit_fold_count,
        )
    )
    registry_path = result_dir_path / "candidate_registry.jsonl"
    if registry_path.exists() and registry_path.stat().st_size > 0:
        verifier_errors = tuple(verifier_errors) + tuple(_candidate_registry_errors(registry_path))
    verifier_passed = not verifier_errors
    complete = (
        not missing_results
        and not missing_audits
        and not missing_required
        and verifier_passed
        and len(expected_pairs) > 0
    )

    return ExpectedEdgeStudyStatus(
        result_dir=result_dir_path,
        plan_path=plan_path,
        profile=str(plan.get("profile", "unknown")),
        expected_edge_jobs=len(expected_pairs),
        present_result_files=present_results,
        present_audit_files=present_audits,
        missing_result_files=missing_results,
        missing_audit_files=missing_audits,
        missing_required_files=missing_required,
        artifact_verifier_passed=verifier_passed,
        artifact_verifier_errors=list(verifier_errors),
        complete=complete,
    )


def write_expected_edge_study_status(status: ExpectedEdgeStudyStatus, path: Path | str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(status)
    payload["result_dir"] = str(status.result_dir)
    payload["plan_path"] = str(status.plan_path)
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def format_expected_edge_study_status(status: ExpectedEdgeStudyStatus) -> str:
    lines = [
        f"result_dir={status.result_dir}",
        f"plan_path={status.plan_path}",
        f"profile={status.profile}",
        f"expected_edge_jobs={status.expected_edge_jobs}",
        f"present_result_files={status.present_result_files}",
        f"present_audit_files={status.present_audit_files}",
        f"missing_result_files={len(status.missing_result_files)}",
        f"missing_audit_files={len(status.missing_audit_files)}",
        f"missing_required_files={len(status.missing_required_files)}",
        f"artifact_verifier_passed={int(status.artifact_verifier_passed)}",
        f"complete={int(status.complete)}",
    ]
    for filename in status.missing_required_files:
        lines.append(f"missing_required={filename}")
    for filename in status.missing_result_files[:20]:
        lines.append(f"missing_result={filename}")
    if len(status.missing_result_files) > 20:
        lines.append(f"missing_result_more={len(status.missing_result_files) - 20}")
    for filename in status.missing_audit_files[:20]:
        lines.append(f"missing_audit={filename}")
    if len(status.missing_audit_files) > 20:
        lines.append(f"missing_audit_more={len(status.missing_audit_files) - 20}")
    for error in status.artifact_verifier_errors:
        lines.append(f"artifact_error={error}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify completion of an expected-edge study from run_plan.json.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--result-dir")
    parser.add_argument("--output")
    parser.add_argument("--no-pvalues", action="store_true")
    parser.add_argument("--min-audit-fold-count", type=int, default=1)
    args = parser.parse_args(argv)

    status = evaluate_expected_edge_study_status(
        plan_path=args.plan,
        result_dir=args.result_dir,
        require_pvalues=not args.no_pvalues,
        min_audit_fold_count=args.min_audit_fold_count,
    )
    if args.output:
        output_path = write_expected_edge_study_status(status, args.output)
        print(f"study_status={output_path}")
    print(format_expected_edge_study_status(status))
    return 0 if status.complete else 1


def _read_plan(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"run plan must be a JSON object: {path}")
    return payload


def _expected_result_pairs(plan: dict[str, Any]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    symbols = [str(symbol).upper() for symbol in plan.get("symbols", [])]
    horizons = [int(horizon) for horizon in plan.get("horizons_ms", [])]
    fees = [float(fee) for fee in plan.get("fees_bps", [])]
    for symbol in symbols:
        for horizon_ms in horizons:
            for fee_bps in fees:
                fee_token = _fee_token(fee_bps)
                stem = f"{symbol}_{horizon_ms}ms_fee_{fee_token}_edge"
                pairs.append((f"{stem}.csv", f"{stem}_audit.csv"))
    return pairs


def _fee_token(fee_bps: float) -> str:
    if fee_bps.is_integer():
        value = str(int(fee_bps))
    else:
        value = format(fee_bps, "g")
    return value.replace(".", "p")


def _filter_verifier_errors(errors: Sequence[str], *, require_pvalues: bool) -> tuple[str, ...]:
    if require_pvalues:
        return tuple(errors)
    ignored = {
        "missing or empty required artifact: pvalues.csv",
        "missing or empty required artifact: pvalue_corrections.csv",
    }
    return tuple(error for error in errors if error not in ignored)


def _audit_fold_count_errors(
    *,
    result_dir: Path,
    expected_pairs: Sequence[tuple[str, str]],
    min_audit_fold_count: int,
) -> list[str]:
    errors: list[str] = []
    for _, audit_name in expected_pairs:
        audit_path = result_dir / audit_name
        if not audit_path.exists() or audit_path.stat().st_size == 0:
            continue
        try:
            with audit_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        except Exception as exc:
            errors.append(f"{audit_name}: could not read audit fold count: {exc!r}")
            continue
        if len(rows) != 1:
            continue
        try:
            fold_count = int(float(rows[0].get("fold_count", "0") or 0))
        except ValueError:
            errors.append(f"{audit_name}: fold_count is not parseable")
            continue
        if fold_count < min_audit_fold_count:
            errors.append(f"{audit_name}: fold_count {fold_count} < required {min_audit_fold_count}")
    return errors


def _candidate_registry_errors(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        attempts = read_expected_edge_candidate_registry(path)
    except Exception as exc:
        return [f"{path.name}: candidate registry parse failed: {exc!r}"]
    if not attempts:
        return [f"{path.name}: no candidate attempts"]
    invalid_statuses = {"planned", "incomplete_artifact", "artifact_error"}
    invalid_counts = {status: 0 for status in invalid_statuses}
    selected_count = 0
    for attempt in attempts:
        if attempt.status in invalid_counts:
            invalid_counts[attempt.status] += 1
        if attempt.selected:
            selected_count += 1
        if not attempt.config_sha256:
            errors.append(f"{path.name}: candidate attempt missing config_sha256")
            break
    for status, count in sorted(invalid_counts.items()):
        if count:
            errors.append(f"{path.name}: status_{status}={count}")
    if selected_count == 0:
        errors.append(f"{path.name}: selected_candidates=0")
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
