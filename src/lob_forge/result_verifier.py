from __future__ import annotations

import csv
import json
import re
import shlex
import subprocess
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
GIT_REV_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")

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

ALLOWED_INFERENCE_GRAINS = {
    "fold_summary",
    "trade_ledger",
    "day_ledger",
    "position_ledger",
}

LEDGER_BACKED_INFERENCE_GRAINS = {
    "trade_ledger",
    "day_ledger",
    "position_ledger",
}

LEDGER_PATH_FIELDS_BY_GRAIN = {
    "trade_ledger": ("ledger_path", "trade_ledger_path", "fills_path"),
    "day_ledger": ("ledger_path", "day_ledger_path", "positions_path"),
    "position_ledger": ("ledger_path", "position_ledger_path", "positions_path"),
}


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

    pvalues = result_dir / "pvalues.csv"
    if pvalues.exists():
        errors.extend(_verify_pvalues(pvalues))
        errors.extend(_verify_pvalues_cover_audits(pvalues, audit_files))

    corrections = result_dir / "pvalue_corrections.csv"
    if corrections.exists():
        errors.extend(_verify_pvalue_corrections(corrections, pvalues_path=pvalues if pvalues.exists() else None))

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
    inference_grain = row["inference_grain"].strip()
    if not inference_grain:
        errors.append(f"{path.name}: inference_grain must be non-empty")
    elif inference_grain not in ALLOWED_INFERENCE_GRAINS:
        allowed = ", ".join(sorted(ALLOWED_INFERENCE_GRAINS))
        errors.append(f"{path.name}: unsupported inference_grain {inference_grain!r}; allowed values: {allowed}")
    elif inference_grain in LEDGER_BACKED_INFERENCE_GRAINS:
        errors.extend(_verify_ledger_backed_inference(path, row, inference_grain))
    if row["acceptance_passed"] in {"0", "0.0"} and not row["rejection_reasons"].strip():
        errors.append(f"{path.name}: rejected audit needs rejection_reasons")
    return errors


def _verify_ledger_backed_inference(path: Path, row: dict[str, str], inference_grain: str) -> list[str]:
    ledger_fields = LEDGER_PATH_FIELDS_BY_GRAIN[inference_grain]
    ledger_path_text = next((row.get(field, "").strip() for field in ledger_fields if row.get(field, "").strip()), "")
    if not ledger_path_text:
        fields = ", ".join(ledger_fields)
        return [f"{path.name}: {inference_grain} inference requires one of these ledger path columns: {fields}"]
    ledger_path = Path(ledger_path_text)
    if not ledger_path.is_absolute():
        ledger_path = path.parent / ledger_path
    if not ledger_path.exists():
        return [f"{path.name}: {inference_grain} ledger artifact is missing: {ledger_path_text}"]
    if ledger_path.stat().st_size == 0:
        return [f"{path.name}: {inference_grain} ledger artifact is empty: {ledger_path_text}"]
    return []


def _verify_pvalues(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return [f"{path.name}: no p-value rows"]
    required = {"hypothesis_id", "metric", "p_value"}
    missing = required - set(rows[0])
    if missing:
        return [f"{path.name}: missing columns {sorted(missing)}"]
    errors: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=2):
        hypothesis_id = str(row.get("hypothesis_id") or "").strip()
        if not hypothesis_id:
            errors.append(f"{path.name}:{index}: missing hypothesis_id")
            continue
        if hypothesis_id in seen:
            errors.append(f"{path.name}:{index}: duplicate hypothesis_id {hypothesis_id}")
        seen.add(hypothesis_id)
        try:
            p_value = float(str(row.get("p_value") or ""))
        except ValueError:
            errors.append(f"{path.name}:{index}: p_value is not parseable")
            continue
        if not 0.0 <= p_value <= 1.0:
            errors.append(f"{path.name}:{index}: p_value outside [0, 1]")
    return errors


def _verify_pvalues_cover_audits(path: Path, audit_files: list[Path]) -> list[str]:
    if not audit_files:
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        return []
    if "config_sha256" in fieldnames:
        return []
    expected_ids = {_audit_result_id(audit_path) for audit_path in audit_files}
    pvalue_ids = {str(row.get("hypothesis_id") or "").strip() for row in rows if row.get("hypothesis_id")}
    errors: list[str] = []
    missing_ids = sorted(expected_ids - pvalue_ids)
    extra_ids = sorted(pvalue_ids - expected_ids)
    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        suffix = "" if len(missing_ids) <= 10 else f", +{len(missing_ids) - 10} more"
        errors.append(f"{path.name}: missing p-value rows for audit artifacts: {preview}{suffix}")
    if extra_ids:
        preview = ", ".join(extra_ids[:10])
        suffix = "" if len(extra_ids) <= 10 else f", +{len(extra_ids) - 10} more"
        errors.append(f"{path.name}: p-value IDs without matching audit artifact: {preview}{suffix}")
    return errors


def _audit_result_id(path: Path) -> str:
    stem = path.stem
    return stem.removesuffix("_audit") if stem.endswith("_audit") else stem


def _verify_pvalue_corrections(path: Path, *, pvalues_path: Path | None = None) -> list[str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return [f"{path.name}: no p-value correction rows"]
    required = {"hypothesis_id", "p_value", "bonferroni_p_value", "bh_adjusted_p_value", "bh_accept"}
    missing = required - set(rows[0])
    if missing:
        return [f"{path.name}: missing columns {sorted(missing)}"]
    errors: list[str] = []
    corrected_ids = {str(row.get("hypothesis_id") or "").strip() for row in rows if row.get("hypothesis_id")}
    for index, row in enumerate(rows, start=2):
        try:
            p_value = float(str(row.get("p_value") or ""))
            bonferroni = float(str(row.get("bonferroni_p_value") or ""))
            bh_adjusted = float(str(row.get("bh_adjusted_p_value") or ""))
        except ValueError:
            errors.append(f"{path.name}:{index}: numeric p-value columns are not parseable")
            continue
        if not all(0.0 <= value <= 1.0 for value in (p_value, bonferroni, bh_adjusted)):
            errors.append(f"{path.name}:{index}: adjusted p-value outside [0, 1]")
    if pvalues_path is not None and pvalues_path.exists():
        with pvalues_path.open(newline="") as handle:
            pvalue_rows = list(csv.DictReader(handle))
        pvalue_ids = {str(row.get("hypothesis_id") or "").strip() for row in pvalue_rows if row.get("hypothesis_id")}
        if corrected_ids != pvalue_ids:
            missing_ids = sorted(pvalue_ids - corrected_ids)
            extra_ids = sorted(corrected_ids - pvalue_ids)
            if missing_ids:
                preview = ", ".join(missing_ids[:10])
                suffix = "" if len(missing_ids) <= 10 else f", +{len(missing_ids) - 10} more"
                errors.append(f"{path.name}: missing corrections for pvalues.csv IDs: {preview}{suffix}")
            if extra_ids:
                preview = ", ".join(extra_ids[:10])
                suffix = "" if len(extra_ids) <= 10 else f", +{len(extra_ids) - 10} more"
                errors.append(f"{path.name}: correction IDs not present in pvalues.csv: {preview}{suffix}")
    return errors


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
        if not _is_valid_git_rev(git_rev):
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


def _is_valid_git_rev(value: str) -> bool:
    if value.lower() in INVALID_GIT_REVS:
        return False
    if not GIT_REV_RE.fullmatch(value):
        return False
    if len(value) >= 40:
        return True
    return _git_rev_resolves(value)


def _git_rev_resolves(value: str) -> bool:
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{value}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
