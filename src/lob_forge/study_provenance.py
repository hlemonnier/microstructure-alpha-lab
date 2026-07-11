from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from lob_forge.holdout import sha256_file
from lob_forge.study_features import verify_expected_edge_feature_artifact


PROVENANCE_VERSION = 2
GIT_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}([0-9a-fA-F]{24})?$")
PLAN_CONFIG_FIELDS = (
    "profile",
    "start_date",
    "end_date",
    "symbols",
    "horizons_ms",
    "fees_bps",
    "latency_ms",
    "bucket_ms",
    "execution_quote_resolution",
    "max_quote_buckets",
    "with_book_depth",
    "feature_threshold",
    "large_trade_notional",
    "symbol_min_ticks",
    "holdout_split_column",
    "holdout_values",
    "train_size",
    "validation_size",
    "test_size",
    "step_size",
    "edge_streaming",
    "edge_thresholds_bps",
    "model_classes",
    "feature_sets",
    "selection_metric",
)


@dataclass(frozen=True)
class ProvenanceVerification:
    provenance_path: Path
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.errors


def provenance_path_for(result_path: Path | str) -> Path:
    return Path(f"{Path(result_path)}.provenance.json")


def write_expected_edge_provenance(
    *,
    plan_path: Path | str,
    feature_path: Path | str,
    holdout_manifest_path: Path | str,
    result_path: Path | str,
    audit_path: Path | str,
    symbol: str,
    horizon_ms: int,
    taker_fee_bps: float,
    output_path: Path | str | None = None,
    source_git_commit: str | None = None,
    working_tree_dirty: bool | None = None,
) -> Path:
    plan = Path(plan_path)
    feature = Path(feature_path)
    holdout = Path(holdout_manifest_path)
    result = Path(result_path)
    audit = Path(audit_path)
    for path in (plan, feature, holdout, result, audit):
        if not path.exists() or path.stat().st_size <= 0:
            raise ValueError(f"cannot write provenance for missing or empty artifact: {path}")
    plan_payload = _read_json_object(plan)
    feature_errors = verify_expected_edge_feature_artifact(
        plan=plan_payload,
        combined_path=feature,
        symbol=symbol,
        horizon_ms=horizon_ms,
    )
    if feature_errors:
        raise ValueError("feature artifact does not match run plan: " + "; ".join(feature_errors))
    holdout_errors = _validate_holdout_against_plan(holdout, feature, plan_payload)
    if holdout_errors:
        raise ValueError("holdout manifest does not match run plan: " + "; ".join(holdout_errors))
    split_errors = validate_result_against_plan(
        result,
        audit,
        plan_payload,
        expected_symbol=symbol,
        expected_horizon_ms=horizon_ms,
        expected_taker_fee_bps=taker_fee_bps,
    )
    if split_errors:
        raise ValueError("result does not match run plan: " + "; ".join(split_errors))
    detected_commit, detected_dirty, source_tree_sha256 = _source_code_provenance()
    resolved_commit = source_git_commit or detected_commit
    resolved_dirty = detected_dirty if working_tree_dirty is None else working_tree_dirty
    if not GIT_COMMIT_RE.fullmatch(resolved_commit):
        raise ValueError("source provenance requires a 40- or 64-character Git commit hash")
    payload = {
        "provenance_version": PROVENANCE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "study_cell": {
            "symbol": symbol.upper(),
            "horizon_ms": int(horizon_ms),
            "taker_fee_bps": float(taker_fee_bps),
        },
        "code_provenance": {
            "git_commit": resolved_commit,
            "source_tree_sha256": source_tree_sha256,
            "working_tree_dirty": resolved_dirty,
        },
        "plan_config": _plan_config(plan_payload),
        "plan_path": str(plan),
        "plan_sha256": sha256_file(plan),
        "feature_path": str(feature),
        "feature_sha256": sha256_file(feature),
        "holdout_manifest_path": str(holdout),
        "holdout_manifest_sha256": sha256_file(holdout),
        "result_path": str(result),
        "result_sha256": sha256_file(result),
        "audit_path": str(audit),
        "audit_sha256": sha256_file(audit),
    }
    output = Path(output_path) if output_path is not None else provenance_path_for(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def verify_expected_edge_provenance(
    provenance_path: Path | str,
    *,
    plan_path: Path | str,
    result_path: Path | str,
    audit_path: Path | str,
    expected_symbol: str,
    expected_horizon_ms: int,
    expected_taker_fee_bps: float,
    require_source_files: bool = False,
) -> ProvenanceVerification:
    provenance = Path(provenance_path)
    plan = Path(plan_path)
    result = Path(result_path)
    audit = Path(audit_path)
    errors: list[str] = []
    if not provenance.exists() or provenance.stat().st_size <= 0:
        return ProvenanceVerification(provenance, ("missing result provenance sidecar",))
    try:
        payload = _read_json_object(provenance)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return ProvenanceVerification(provenance, (f"invalid provenance JSON: {exc}",))
    if payload.get("provenance_version") != PROVENANCE_VERSION:
        errors.append(f"unsupported provenance_version={payload.get('provenance_version')!r}")
    errors.extend(_code_provenance_errors(payload))

    expected_cell = {
        "symbol": expected_symbol.upper(),
        "horizon_ms": int(expected_horizon_ms),
        "taker_fee_bps": float(expected_taker_fee_bps),
    }
    if payload.get("study_cell") != expected_cell:
        errors.append(f"study_cell mismatch: {payload.get('study_cell')!r} != {expected_cell!r}")

    if not plan.exists() or plan.stat().st_size <= 0:
        errors.append(f"missing run plan: {plan}")
        plan_payload: dict[str, Any] = {}
    else:
        plan_payload = _read_json_object(plan)
        if payload.get("plan_sha256") != sha256_file(plan):
            errors.append("run plan SHA-256 mismatch")
        if payload.get("plan_config") != _plan_config(plan_payload):
            errors.append("run plan configuration mismatch")

    for label, path, hash_field in (
        ("result", result, "result_sha256"),
        ("audit", audit, "audit_sha256"),
    ):
        if not path.exists() or path.stat().st_size <= 0:
            errors.append(f"missing or empty {label} artifact: {path}")
        elif payload.get(hash_field) != sha256_file(path):
            errors.append(f"{label} SHA-256 mismatch")

    resolved_sources: dict[str, Path] = {}
    for label, path_field, hash_field in (
        ("feature", "feature_path", "feature_sha256"),
        ("holdout manifest", "holdout_manifest_path", "holdout_manifest_sha256"),
    ):
        source = _resolve_recorded_path(str(payload.get(path_field, "")), provenance=provenance, plan=plan)
        if source is None:
            if require_source_files:
                errors.append(f"recorded {label} source is unavailable")
            continue
        resolved_sources[label] = source
        if payload.get(hash_field) != sha256_file(source):
            errors.append(f"{label} SHA-256 mismatch")

    feature_source = resolved_sources.get("feature")
    if plan_payload and feature_source is not None:
        errors.extend(
            verify_expected_edge_feature_artifact(
                plan=plan_payload,
                combined_path=feature_source,
                symbol=expected_symbol,
                horizon_ms=expected_horizon_ms,
            )
        )
        holdout_source = resolved_sources.get("holdout manifest")
        if holdout_source is not None:
            errors.extend(_validate_holdout_against_plan(holdout_source, feature_source, plan_payload))

    if plan_payload and result.exists() and audit.exists():
        errors.extend(
            validate_result_against_plan(
                result,
                audit,
                plan_payload,
                expected_symbol=expected_symbol,
                expected_horizon_ms=expected_horizon_ms,
                expected_taker_fee_bps=expected_taker_fee_bps,
            )
        )
    return ProvenanceVerification(provenance, tuple(errors))


def verify_recorded_expected_edge_provenance(
    provenance_path: Path | str,
    *,
    result_path: Path | str,
    audit_path: Path | str,
    require_source_files: bool = False,
) -> ProvenanceVerification:
    """Verify a sidecar using the plan and study cell recorded inside it."""
    provenance = Path(provenance_path)
    if not provenance.exists() or provenance.stat().st_size <= 0:
        return ProvenanceVerification(provenance, ("missing result provenance sidecar",))
    try:
        payload = _read_json_object(provenance)
        cell = payload["study_cell"]
        if not isinstance(cell, dict):
            raise ValueError("study_cell must be a JSON object")
        symbol = str(cell["symbol"])
        horizon_ms = int(cell["horizon_ms"])
        taker_fee_bps = float(cell["taker_fee_bps"])
        recorded_plan = Path(str(payload["plan_path"]))
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        return ProvenanceVerification(provenance, (f"invalid recorded provenance metadata: {exc}",))

    plan_candidates = [recorded_plan]
    if not recorded_plan.is_absolute():
        plan_candidates.extend(parent / recorded_plan for parent in (provenance.parent, *provenance.parents))
    plan = next(
        (candidate for candidate in plan_candidates if candidate.exists() and candidate.stat().st_size > 0),
        recorded_plan,
    )
    return verify_expected_edge_provenance(
        provenance,
        plan_path=plan,
        result_path=result_path,
        audit_path=audit_path,
        expected_symbol=symbol,
        expected_horizon_ms=horizon_ms,
        expected_taker_fee_bps=taker_fee_bps,
        require_source_files=require_source_files,
    )


def validate_result_against_plan(
    result_path: Path | str,
    audit_path: Path | str,
    plan: dict[str, Any],
    *,
    expected_symbol: str,
    expected_horizon_ms: int,
    expected_taker_fee_bps: float,
) -> list[str]:
    result = Path(result_path)
    audit = Path(audit_path)
    errors: list[str] = []
    if expected_symbol.upper() not in {str(value).upper() for value in plan.get("symbols", [])}:
        errors.append(f"symbol {expected_symbol.upper()} is not declared by the run plan")
    if int(expected_horizon_ms) not in {int(value) for value in plan.get("horizons_ms", [])}:
        errors.append(f"horizon {expected_horizon_ms} is not declared by the run plan")
    if not any(
        math.isclose(float(expected_taker_fee_bps), float(value), abs_tol=1e-12) for value in plan.get("fees_bps", [])
    ):
        errors.append(f"fee {expected_taker_fee_bps} is not declared by the run plan")

    with result.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    fold_rows = [row for row in rows if str(row.get("fold", "")).isdigit()]
    if not fold_rows:
        return errors + ["result contains no numeric fold rows"]
    required = {
        "train_rows",
        "validation_rows",
        "test_rows",
        "purged_train_rows",
        "purged_validation_rows",
        "edge_threshold_bps",
        "test_trades",
        "test_net_pnl",
    }
    missing = required - set(fold_rows[0])
    if missing:
        return errors + [f"result is missing provenance-check columns: {sorted(missing)}"]

    expected_train = int(plan.get("train_size", 0))
    expected_validation = int(plan.get("validation_size", 0))
    expected_test = int(plan.get("test_size", 0))
    expected_step = int(plan.get("step_size", 0))
    if expected_test and expected_step and expected_step < expected_test:
        errors.append(
            f"planned step size {expected_step} is smaller than test size {expected_test}; OOS windows overlap"
        )
    declared_thresholds = [float(value) for value in plan.get("edge_thresholds_bps", [])]
    total_test_rows = 0
    total_test_trades = 0
    total_test_net_pnl = 0.0
    for row in fold_rows:
        fold = row["fold"]
        try:
            raw_train = int(float(row["train_rows"])) + int(float(row["purged_train_rows"]))
            raw_validation = int(float(row["validation_rows"])) + int(float(row["purged_validation_rows"]))
            test_rows = int(float(row["test_rows"]))
            threshold = float(row["edge_threshold_bps"])
            test_trades = int(float(row["test_trades"]))
            test_net_pnl = float(row["test_net_pnl"])
        except ValueError:
            errors.append(f"fold {fold}: non-parseable configuration or result value")
            continue
        if expected_train and raw_train != expected_train:
            errors.append(f"fold {fold}: raw train rows {raw_train} != planned {expected_train}")
        if expected_validation and raw_validation != expected_validation:
            errors.append(f"fold {fold}: raw validation rows {raw_validation} != planned {expected_validation}")
        if expected_test and test_rows != expected_test:
            errors.append(f"fold {fold}: test rows {test_rows} != planned {expected_test}")
        if declared_thresholds and not any(
            math.isclose(threshold, value, abs_tol=1e-12) for value in declared_thresholds
        ):
            errors.append(f"fold {fold}: selected threshold {threshold} is not in the planned grid")
        total_test_rows += test_rows
        total_test_trades += test_trades
        total_test_net_pnl += test_net_pnl

    with audit.open(newline="") as handle:
        audit_rows = list(csv.DictReader(handle))
    if len(audit_rows) != 1:
        return errors + ["audit must contain exactly one row"]
    audit_row = audit_rows[0]
    for field, expected in (
        ("fold_count", len(fold_rows)),
        ("total_test_rows", total_test_rows),
        ("total_test_trades", total_test_trades),
    ):
        try:
            observed = int(float(audit_row.get(field, "")))
        except ValueError:
            errors.append(f"audit {field} is not parseable")
            continue
        if observed != expected:
            errors.append(f"audit {field} {observed} != result-derived {expected}")
    try:
        audit_net = float(audit_row.get("total_test_net_pnl", ""))
        if not math.isclose(audit_net, total_test_net_pnl, rel_tol=1e-9, abs_tol=1e-6):
            errors.append(f"audit total_test_net_pnl {audit_net} != result-derived {total_test_net_pnl}")
    except ValueError:
        errors.append("audit total_test_net_pnl is not parseable")
    return errors


def _plan_config(plan: dict[str, Any]) -> dict[str, Any]:
    return {field: plan.get(field) for field in PLAN_CONFIG_FIELDS}


def _validate_holdout_against_plan(holdout: Path, feature: Path, plan: dict[str, Any]) -> list[str]:
    try:
        payload = _read_json_object(holdout)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"invalid holdout manifest JSON: {exc}"]
    errors: list[str] = []
    expected_split = str(plan.get("holdout_split_column", "source_date"))
    expected_values = sorted(
        str(value) for value in plan.get("holdout_values", [str(plan.get("end_date", ""))]) if str(value)
    )
    observed_values = sorted(str(value) for value in payload.get("holdout_values", []))
    if payload.get("split_column") != expected_split:
        errors.append(f"split_column {payload.get('split_column')!r} != planned {expected_split!r}")
    if observed_values != expected_values:
        errors.append(f"holdout_values {observed_values!r} != planned {expected_values!r}")
    feature_sha256 = sha256_file(feature)
    if payload.get("source_sha256") != feature_sha256 or payload.get("dataset_fingerprint") != feature_sha256:
        errors.append("holdout source fingerprint does not match combined feature content")
    return errors


@lru_cache(maxsize=1)
def _source_code_provenance() -> tuple[str, bool | None, str]:
    root = Path(__file__).resolve().parents[2]
    commit = ""
    dirty: bool | None = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        dirty = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        archive_commit = root / ".source-git-commit"
        if archive_commit.exists():
            commit = archive_commit.read_text().strip()
    return commit, dirty, _source_tree_sha256(root)


def _source_tree_sha256(root: Path) -> str:
    hasher = hashlib.sha256()
    paths: list[Path] = []
    for directory in (root / "src", root / "scripts"):
        if directory.exists():
            paths.extend(path for path in directory.rglob("*") if path.is_file())
    paths.extend(
        path
        for path in (
            root / "pyproject.toml",
            root / "requirements-ci.txt",
            root / "requirements-research.txt",
        )
        if path.exists()
    )
    for path in sorted(paths, key=lambda value: str(value.relative_to(root))):
        relative = str(path.relative_to(root))
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def _code_provenance_errors(payload: dict[str, Any]) -> list[str]:
    code = payload.get("code_provenance")
    if not isinstance(code, dict):
        return ["missing code provenance"]
    errors: list[str] = []
    recorded_commit = str(code.get("git_commit", ""))
    recorded_tree = str(code.get("source_tree_sha256", ""))
    recorded_dirty = code.get("working_tree_dirty")
    if not GIT_COMMIT_RE.fullmatch(recorded_commit):
        errors.append("invalid source Git commit")
    if not re.fullmatch(r"[0-9a-f]{64}", recorded_tree):
        errors.append("invalid source tree SHA-256")
    if recorded_dirty is True:
        errors.append("result was generated from a dirty working tree")
    elif recorded_dirty not in {False, None}:
        errors.append("working_tree_dirty must be false or null")

    current_commit, _current_dirty, current_tree = _source_code_provenance()
    if GIT_COMMIT_RE.fullmatch(recorded_commit) and current_commit and recorded_commit != current_commit:
        errors.append("source Git commit does not match current checkout")
    if recorded_tree and recorded_tree != current_tree:
        errors.append("source tree SHA-256 does not match current checkout")
    return errors


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _resolve_recorded_path(value: str, *, provenance: Path, plan: Path) -> Path | None:
    if not value:
        return None
    recorded = Path(value)
    candidates = [recorded] if recorded.is_absolute() else [Path.cwd() / recorded]
    if not recorded.is_absolute():
        candidates.extend(parent / recorded for parent in (provenance.parent, plan.parent, *plan.parents))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write or verify expected-edge result provenance.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("write", "verify"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--plan", required=True)
        sub.add_argument("--result", required=True)
        sub.add_argument("--audit", required=True)
        sub.add_argument("--symbol", required=True)
        sub.add_argument("--horizon-ms", type=int, required=True)
        sub.add_argument("--taker-fee-bps", type=float, required=True)
        sub.add_argument("--output")
    write_parser = subparsers.choices["write"]
    write_parser.add_argument("--feature", required=True)
    write_parser.add_argument("--holdout-manifest", required=True)
    verify_parser = subparsers.choices["verify"]
    verify_parser.add_argument("--require-source-files", action="store_true")
    args = parser.parse_args(argv)
    provenance = Path(args.output) if args.output else provenance_path_for(args.result)
    if args.command == "write":
        output = write_expected_edge_provenance(
            plan_path=args.plan,
            feature_path=args.feature,
            holdout_manifest_path=args.holdout_manifest,
            result_path=args.result,
            audit_path=args.audit,
            symbol=args.symbol,
            horizon_ms=args.horizon_ms,
            taker_fee_bps=args.taker_fee_bps,
            output_path=provenance,
        )
        print(f"provenance={output}")
        return 0
    report = verify_expected_edge_provenance(
        provenance,
        plan_path=args.plan,
        result_path=args.result,
        audit_path=args.audit,
        expected_symbol=args.symbol,
        expected_horizon_ms=args.horizon_ms,
        expected_taker_fee_bps=args.taker_fee_bps,
        require_source_files=args.require_source_files,
    )
    print(f"provenance={report.provenance_path}")
    print(f"passed={int(report.passed)}")
    for error in report.errors:
        print(f"error={error}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
