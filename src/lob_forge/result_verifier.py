from __future__ import annotations

import csv
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path


GATED_CLI_COMMANDS = {
    "baseline",
    "fee-sweep",
    "walk-forward",
    "calendar-walk-forward",
    "conditional-walk-forward",
    "logistic-walk-forward",
    "edge-walk-forward",
    "edge-shadow-decisions",
    "eval-rule",
    "regime",
}

INVALID_GIT_REVS = {
    "",
    "no-git-commit",
    "fixture-run-without-git",
    "package-no-git",
    "unknown",
    "none",
    "null",
}

REQUIRED_RESULT_FILES = (
    "hypotheses.jsonl",
    "experiment_ledger.jsonl",
    "pvalues.csv",
    "pvalue_corrections.csv",
)

REQUIRED_STUDY_FILES = (
    "pvalues.csv",
    "pvalue_corrections.csv",
)


@dataclass(frozen=True)
class ResultVerification:
    result_dir: Path
    checked_files: int
    audit_files: int
    passed: bool
    errors: tuple[str, ...]


def verify_result_artifacts(result_dir: Path | str, *, strict_metadata: bool = True) -> ResultVerification:
    result_dir = Path(result_dir)
    errors: list[str] = []
    checked = 0
    if not result_dir.exists():
        return ResultVerification(result_dir, 0, 0, False, (f"missing result dir: {result_dir}",))

    required_files = REQUIRED_RESULT_FILES if strict_metadata else REQUIRED_STUDY_FILES
    for filename in required_files:
        checked += 1
        path = result_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"missing or empty required artifact: {filename}")
        elif strict_metadata and filename == "experiment_ledger.jsonl":
            errors.extend(_verify_experiment_ledger(path))

    audit_files = sorted(result_dir.glob("*_audit.csv"))
    if not audit_files:
        errors.append("no audit CSV files found")
    for path in audit_files:
        checked += 1
        errors.extend(_verify_audit_csv(path))

    corrections = result_dir / "pvalue_corrections.csv"
    if corrections.exists():
        errors.extend(_verify_pvalue_corrections(corrections))

    return ResultVerification(
        result_dir=result_dir,
        checked_files=checked,
        audit_files=len(audit_files),
        passed=not errors,
        errors=tuple(errors),
    )


def format_result_verification(report: ResultVerification) -> str:
    lines = [
        f"result_dir={report.result_dir}",
        f"checked_files={report.checked_files}",
        f"audit_files={report.audit_files}",
        f"passed={int(report.passed)}",
    ]
    for error in report.errors:
        lines.append(f"error={error}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    strict_metadata = True
    if "--study" in args:
        strict_metadata = False
        args = [arg for arg in args if arg != "--study"]
    result_dir = Path(args[0]) if args else Path("results/current")
    report = verify_result_artifacts(result_dir, strict_metadata=strict_metadata)
    print(format_result_verification(report))
    return 0 if report.passed else 1


def _verify_audit_csv(path: Path) -> list[str]:
    errors: list[str] = []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        return [f"{path.name}: expected exactly one audit row"]
    row = rows[0]
    required = {
        "inference_grain",
        "fold_count",
        "total_test_rows",
        "total_test_trades",
        "total_test_net_pnl",
        "acceptance_passed",
        "rejection_reasons",
    }
    missing = required - set(row)
    if missing:
        return [f"{path.name}: missing audit columns {sorted(missing)}"]
    try:
        fold_count = int(float(row["fold_count"]))
        test_rows = int(float(row["total_test_rows"]))
        test_trades = int(float(row["total_test_trades"]))
        int(float(row["acceptance_passed"]))
    except ValueError:
        return [f"{path.name}: numeric audit columns are not parseable"]
    if fold_count <= 0:
        errors.append(f"{path.name}: fold_count must be positive")
    if test_rows <= 0:
        errors.append(f"{path.name}: total_test_rows must be positive")
    if test_trades < 0:
        errors.append(f"{path.name}: total_test_trades must be non-negative")
    if not row["inference_grain"].strip():
        errors.append(f"{path.name}: inference_grain must be non-empty")
    if row["acceptance_passed"] in {"0", "0.0"} and not row["rejection_reasons"].strip():
        errors.append(f"{path.name}: rejected audit needs rejection_reasons")
    return errors


def _verify_pvalue_corrections(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return [f"{path.name}: no p-value correction rows"]
    required = {"hypothesis_id", "p_value", "bonferroni_p_value", "bh_adjusted_p_value", "bh_accept"}
    missing = required - set(rows[0])
    if missing:
        return [f"{path.name}: missing columns {sorted(missing)}"]
    return []


def _verify_experiment_ledger(path: Path) -> list[str]:
    errors: list[str] = []
    rows = list(_read_jsonl(path, errors))
    if not rows:
        errors.append(f"{path.name}: no experiment ledger rows")
        return errors
    seen_experiment_ids: set[str] = set()
    for line_number, row in rows:
        experiment_id = str(row.get("experiment_id") or f"line {line_number}")
        if experiment_id in seen_experiment_ids:
            errors.append(f"{path.name}:{line_number} {experiment_id}: duplicate experiment_id")
        seen_experiment_ids.add(experiment_id)
        git_rev = str(row.get("git_rev") or "").strip()
        if git_rev.lower() in INVALID_GIT_REVS:
            errors.append(f"{path.name}:{line_number} {experiment_id}: invalid git_rev {git_rev!r}")
        command = str(row.get("command") or "")
        if not _uses_gated_cli_command(command):
            continue
        if "--holdout-manifest" not in _command_tokens(command):
            errors.append(f"{path.name}:{line_number} {experiment_id}: gated command lacks --holdout-manifest")
        manifest_path = str(row.get("holdout_manifest_path") or "").strip()
        manifest_sha256 = str(row.get("holdout_manifest_sha256") or "").strip()
        if not manifest_path:
            errors.append(f"{path.name}:{line_number} {experiment_id}: missing holdout_manifest_path")
        if not _is_sha256(manifest_sha256):
            errors.append(f"{path.name}:{line_number} {experiment_id}: missing or invalid holdout_manifest_sha256")
    return errors


def _read_jsonl(path: Path, errors: list[str]) -> list[tuple[int, dict[str, object]]]:
    rows: list[tuple[int, dict[str, object]]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{line_number}: invalid JSON: {exc.msg}")
                continue
            if not isinstance(value, dict):
                errors.append(f"{path.name}:{line_number}: expected JSON object")
                continue
            rows.append((line_number, value))
    return rows


def _uses_gated_cli_command(command: str) -> bool:
    return any(token in GATED_CLI_COMMANDS for token in _command_tokens(command))


def _command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)


if __name__ == "__main__":
    raise SystemExit(main())
