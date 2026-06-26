from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from lob_forge.evidence_gates import format_evidence_gate_report
from lob_forge.live_validation import has_observed_fill, read_shadow_decisions, validate_shadow_fill_predictions
from lob_forge.provider_order_ids import read_order_plan_client_id_map
from lob_forge.study_status import evaluate_expected_edge_study_status


EXCLUDED_PACKAGE_PREFIXES = (
    "data/",
    "results/",
    ".git/",
    ".venv/",
    "venv/",
    ".next/",
    "node_modules/",
)

DEFAULT_SHADOW_ORDER_PLAN_FILENAMES = (
    "bybit_order_plan.jsonl",
    "okx_order_plan.jsonl",
    "binance_usdm_order_plan.jsonl",
)

REQUIRED_CLOUD_PACKAGE_SUFFIXES = (
    ".source-git-commit",
    "requirements-ci.txt",
    "requirements-research.txt",
    "scripts/bootstrap_cloud_expected_edge.sh",
    "scripts/modal_expected_edge_job.py",
)


@dataclass(frozen=True)
class ReadinessCheck:
    check_id: str
    status: str
    passed: bool
    evidence: str
    next_action: str


@dataclass(frozen=True)
class ExternalGateReadinessReport:
    checks: tuple[ReadinessCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


def evaluate_external_gate_readiness(
    *,
    project_root: Path | str = ".",
    modal_binary: str | None = None,
    cloud_package: Path | str | None = None,
    full_plan: Path | str = "results/expected_edge_60day_20230516_20230714/run_plan.json",
    full_result_dir: Path | str = "results/expected_edge_60day_20230516_20230714",
    min_audit_fold_count: int = 20,
    shadow_decisions: Path | str = "results/shadow_validation/shadow_decisions.csv",
    simulated_fills: Path | str = "results/shadow_validation/simulated_fills.csv",
    observed_template: Path | str = "results/shadow_validation/observed_fills_template.csv",
    order_plans: Sequence[Path | str] | None = None,
    min_shadow_observations: int = 20,
    max_price_error: float | None = 5.0,
    max_size_error: float | None = 0.01,
    max_fill_rate_error: float | None = 0.05,
) -> ExternalGateReadinessReport:
    root = Path(project_root)
    package_path = Path(cloud_package) if cloud_package is not None else _latest_cloud_package(root)
    resolved_modal = _resolve_modal_binary(root=root, modal_binary=modal_binary)
    observed_template_path = _resolve(root, observed_template)
    order_plan_paths = (
        tuple(_resolve(root, path) for path in order_plans)
        if order_plans is not None
        else _default_shadow_order_plan_paths(observed_template_path)
    )
    checks = (
        _modal_cli_check(resolved_modal=resolved_modal),
        _modal_auth_check(resolved_modal=resolved_modal),
        _cloud_package_check(package_path, current_git_head=_current_git_head(root)),
        _full_cloud_study_check(
            plan_path=_resolve(root, full_plan),
            result_dir=_resolve(root, full_result_dir),
            min_audit_fold_count=min_audit_fold_count,
        ),
        _shadow_fill_readiness_check(
            shadow_path=_resolve(root, shadow_decisions),
            simulated_path=_resolve(root, simulated_fills),
            observed_template_path=observed_template_path,
            order_plan_paths=order_plan_paths,
            min_shadow_observations=min_shadow_observations,
            max_price_error=max_price_error,
            max_size_error=max_size_error,
            max_fill_rate_error=max_fill_rate_error,
        ),
    )
    return ExternalGateReadinessReport(checks=checks)


def format_external_gate_readiness(report: ExternalGateReadinessReport, *, output_format: str = "text") -> str:
    if output_format == "json":
        return json.dumps(
            {"passed": report.passed, "checks": [asdict(check) for check in report.checks]}, indent=2, sort_keys=True
        )
    if output_format == "csv":
        from lob_forge.evidence_gates import EvidenceGate, EvidenceGateReport

        gate_report = EvidenceGateReport(
            gates=tuple(
                EvidenceGate(
                    gate_id=check.check_id,
                    todo_text="external gate readiness",
                    status=check.status,
                    passed=check.passed,
                    evidence=check.evidence,
                    next_action=check.next_action,
                )
                for check in report.checks
            )
        )
        return format_evidence_gate_report(gate_report, output_format="csv")
    if output_format != "text":
        raise ValueError("output_format must be text, csv, or json")
    lines = [f"passed={int(report.passed)}"]
    for check in report.checks:
        lines.extend(
            [
                f"check={check.check_id}",
                f"status={check.status}",
                f"passed={int(check.passed)}",
                f"evidence={check.evidence}",
                f"next_action={check.next_action}",
            ]
        )
    return "\n".join(lines)


def write_external_gate_readiness(
    report: ExternalGateReadinessReport,
    path: Path | str,
    *,
    output_format: str = "json",
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(format_external_gate_readiness(report, output_format=output_format) + "\n")
    return output


def _resolve_modal_binary(*, root: Path, modal_binary: str | None) -> str | None:
    if modal_binary:
        return (
            shutil.which(modal_binary)
            if "/" not in modal_binary
            else modal_binary
            if os.access(modal_binary, os.X_OK)
            else None
        )
    candidates = []
    venv_modal = root / ".venv" / "bin" / "modal"
    if venv_modal.exists():
        candidates.append(str(venv_modal))
    candidates.append("modal")

    for candidate in candidates:
        resolved = (
            shutil.which(candidate) if "/" not in candidate else candidate if os.access(candidate, os.X_OK) else None
        )
        if resolved:
            return resolved
    return None


def _modal_cli_check(*, resolved_modal: str | None) -> ReadinessCheck:
    if resolved_modal:
        return ReadinessCheck(
            "modal_cli",
            "passed",
            True,
            f"modal_binary={resolved_modal}",
            "authenticate with modal setup if modal_auth is not passed",
        )
    return ReadinessCheck(
        "modal_cli",
        "missing",
        False,
        "modal_binary=missing",
        'install with python3 -m pip install ".[cloud]" and authenticate with modal setup',
    )


def _modal_auth_check(*, resolved_modal: str | None) -> ReadinessCheck:
    if resolved_modal is None:
        return ReadinessCheck(
            "modal_auth",
            "missing",
            False,
            "modal_binary=missing",
            'install with python3 -m pip install ".[cloud]" and authenticate with modal setup',
        )
    try:
        completed = subprocess.run(
            [resolved_modal, "config", "show"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return ReadinessCheck(
            "modal_auth",
            "failed",
            False,
            f"modal_binary={resolved_modal} config_error={type(exc).__name__}",
            "run modal setup and rerun make external-readiness",
        )
    if completed.returncode != 0:
        stderr = completed.stderr.strip().splitlines()
        detail = stderr[0] if stderr else f"exit_code={completed.returncode}"
        return ReadinessCheck(
            "modal_auth",
            "failed",
            False,
            f"modal_binary={resolved_modal} config_show=failed detail={detail}",
            "run modal setup and rerun make external-readiness",
        )
    try:
        config = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return ReadinessCheck(
            "modal_auth",
            "failed",
            False,
            f"modal_binary={resolved_modal} config_json=0",
            "run modal setup and rerun make external-readiness",
        )
    token_id = config.get("token_id")
    token_secret = config.get("token_secret")
    if token_id and token_secret:
        redacted_id = str(token_id)[:6] + "..." if len(str(token_id)) > 6 else "present"
        return ReadinessCheck(
            "modal_auth",
            "passed",
            True,
            f"modal_binary={resolved_modal} token_id={redacted_id} token_secret_present=1",
            "run MODE=plan make modal-study, then MODE=run make modal-study when cloud spend is approved",
        )
    return ReadinessCheck(
        "modal_auth",
        "not_ready",
        False,
        f"modal_binary={resolved_modal} token_id_present={int(bool(token_id))} token_secret_present={int(bool(token_secret))}",
        "run modal setup or modal token set, then rerun make external-readiness",
    )


def _cloud_package_check(package_path: Path | None, *, current_git_head: str | None = None) -> ReadinessCheck:
    if package_path is None:
        return ReadinessCheck(
            "cloud_handoff_package",
            "missing",
            False,
            "package=missing",
            "run bash scripts/package_cloud_handoff.sh",
        )
    if not package_path.exists():
        return ReadinessCheck(
            "cloud_handoff_package",
            "missing",
            False,
            f"package={package_path} present=0",
            "run bash scripts/package_cloud_handoff.sh",
        )
    try:
        with zipfile.ZipFile(package_path) as archive:
            names = archive.namelist()
    except zipfile.BadZipFile:
        return ReadinessCheck(
            "cloud_handoff_package",
            "failed",
            False,
            f"package={package_path} zip_valid=0",
            "regenerate the package",
        )
    excluded = [name for name in names if _is_excluded_package_name(name)]
    if excluded:
        return ReadinessCheck(
            "cloud_handoff_package",
            "failed",
            False,
            f"package={package_path} entries={len(names)} excluded_entries={len(excluded)} first_excluded={excluded[0]}",
            "fix package exclusions and regenerate the handoff zip",
        )
    by_suffix = {_package_entry_suffix(name): name for name in names}
    missing = [suffix for suffix in REQUIRED_CLOUD_PACKAGE_SUFFIXES if suffix not in by_suffix]
    if missing:
        return ReadinessCheck(
            "cloud_handoff_package",
            "failed",
            False,
            f"package={package_path} entries={len(names)} excluded_entries=0 missing_required={','.join(missing)}",
            "regenerate the cloud handoff package from the current source tree",
        )
    source_git_commit = _read_zip_text(archive_path=package_path, member=by_suffix[".source-git-commit"]).strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", source_git_commit):
        return ReadinessCheck(
            "cloud_handoff_package",
            "failed",
            False,
            f"package={package_path} entries={len(names)} excluded_entries=0 source_provenance=0",
            "regenerate the cloud handoff package with a resolved source commit",
        )
    if current_git_head is not None and source_git_commit != current_git_head:
        return ReadinessCheck(
            "cloud_handoff_package",
            "failed",
            False,
            f"package={package_path} entries={len(names)} excluded_entries=0 source_provenance=1 source_commit_matches_head=0 package_commit={source_git_commit[:12]} head_commit={current_git_head[:12]}",
            "regenerate the cloud handoff package from the current committed source tree",
        )
    bootstrap = _read_zip_text(archive_path=package_path, member=by_suffix["scripts/bootstrap_cloud_expected_edge.sh"])
    modal = _read_zip_text(archive_path=package_path, member=by_suffix["scripts/modal_expected_edge_job.py"])
    pinned_install = (
        "pip install -r requirements-research.txt" in bootstrap
        and "pip install -e . --no-deps" in bootstrap
        and "pip install -r requirements-research.txt" in modal
        and "pip install -e . --no-deps" in modal
    )
    range_research_install = 'pip install -e ".[research]"' in bootstrap or "pip install -e '.[research]'" in modal
    if not pinned_install or range_research_install:
        return ReadinessCheck(
            "cloud_handoff_package",
            "failed",
            False,
            f"package={package_path} entries={len(names)} excluded_entries=0 pinned_research_install={int(pinned_install)} range_research_install={int(range_research_install)}",
            "regenerate the package after fixing cloud bootstrap dependency installation",
        )
    return ReadinessCheck(
        "cloud_handoff_package",
        "passed",
        True,
        f"package={package_path} entries={len(names)} excluded_entries=0 pinned_research_install=1 source_provenance=1 source_commit_matches_head={int(current_git_head is None or source_git_commit == current_git_head)}",
        "upload to a high-RAM VM or use the Modal runner",
    )


def _full_cloud_study_check(*, plan_path: Path, result_dir: Path, min_audit_fold_count: int) -> ReadinessCheck:
    if not plan_path.exists():
        return ReadinessCheck(
            "full_cloud_study",
            "missing",
            False,
            f"plan={plan_path} present=0",
            "run CONFIRM_HEAVY=1 STUDY_PROFILE=cloud_full PLAN_ONLY=1 bash scripts/run_60day_expected_edge_study.sh",
        )
    try:
        status = evaluate_expected_edge_study_status(
            plan_path=plan_path,
            result_dir=result_dir,
            min_audit_fold_count=min_audit_fold_count,
        )
    except Exception as exc:
        return ReadinessCheck(
            "full_cloud_study",
            "failed",
            False,
            f"plan={plan_path} result_dir={result_dir} error={type(exc).__name__}",
            "fix full study verifier errors before running or resuming cloud execution",
        )
    evidence = (
        f"profile={status.profile} result_dir={status.result_dir} "
        f"present_results={status.present_result_files}/{status.expected_edge_jobs} "
        f"present_audits={status.present_audit_files}/{status.expected_edge_jobs} "
        f"missing_required={len(status.missing_required_files)} min_audit_fold_count={min_audit_fold_count}"
    )
    if status.complete:
        return ReadinessCheck("full_cloud_study", "passed", True, evidence, "run make verify-evidence-gates")
    return ReadinessCheck(
        "full_cloud_study",
        "not_ready",
        False,
        evidence,
        "run or resume the full cloud study on Modal/high-RAM VM",
    )


def _shadow_fill_readiness_check(
    *,
    shadow_path: Path,
    simulated_path: Path,
    observed_template_path: Path,
    order_plan_paths: Sequence[Path],
    min_shadow_observations: int,
    max_price_error: float | None,
    max_size_error: float | None,
    max_fill_rate_error: float | None,
) -> ReadinessCheck:
    if not shadow_path.exists() or not simulated_path.exists():
        return ReadinessCheck(
            "paper_live_fill_validation",
            "missing",
            False,
            f"shadow_exists={int(shadow_path.exists())} simulated_exists={int(simulated_path.exists())}",
            "run edge-shadow-decisions, features-to-market-events, and simulate-shadow-fills first",
        )
    try:
        decisions = read_shadow_decisions(shadow_path)
        observed = [decision for decision in decisions if has_observed_fill(decision)]
    except Exception as exc:
        return ReadinessCheck(
            "paper_live_fill_validation",
            "failed",
            False,
            f"shadow={shadow_path} simulated={simulated_path} error={type(exc).__name__}",
            "fix shadow decision files and import observed fills",
        )
    order_plan_rows_and_errors = tuple(_count_order_plan_rows(path) for path in order_plan_paths)
    order_plan_rows = tuple(rows for rows, _error in order_plan_rows_and_errors)
    existing_order_plans = sum(1 for path in order_plan_paths if path.exists())
    nonempty_order_plans = sum(1 for rows in order_plan_rows if rows > 0)
    missing_order_plan_names = ",".join(path.name for path in order_plan_paths if not path.exists()) or "none"
    invalid_order_plan_names = (
        ",".join(
            path.name for path, (_rows, error) in zip(order_plan_paths, order_plan_rows_and_errors) if error is not None
        )
        or "none"
    )
    if len(observed) < min_shadow_observations:
        evidence = (
            f"observed_shadow_rows={len(observed)} required={min_shadow_observations} "
            "matched=0 validation_passed=0 "
            f"template_exists={int(observed_template_path.exists())} "
            f"order_plan_files={existing_order_plans}/{len(order_plan_paths)} "
            f"nonempty_order_plans={nonempty_order_plans}/{len(order_plan_paths)} "
            f"order_plan_rows={sum(order_plan_rows)} missing_order_plans={missing_order_plan_names} "
            f"invalid_order_plans={invalid_order_plan_names}"
        )
        next_action = (
            "generate observed-fill-template and paper-order-plan, run submit-paper-orders with --execute on demo/testnet, fetch order/fill history with fetch-observed-fills, then import observations"
            if not observed_template_path.exists()
            else "generate paper-order-plan, run submit-paper-orders with --execute on demo/testnet, fetch order/fill history with fetch-observed-fills, normalize/import observations; blank templates do not count"
            if existing_order_plans < len(order_plan_paths) or nonempty_order_plans == 0
            else "run submit-paper-orders with --execute on demo/testnet, fetch order/fill history with fetch-observed-fills, normalize/import observations; blank templates do not count"
        )
        return ReadinessCheck("paper_live_fill_validation", "not_ready", False, evidence, next_action)
    try:
        report = validate_shadow_fill_predictions(
            simulated_path=simulated_path,
            shadow_path=shadow_path,
            max_price_error=max_price_error,
            max_size_error=max_size_error,
            max_fill_rate_error=max_fill_rate_error,
        )
    except Exception as exc:
        return ReadinessCheck(
            "paper_live_fill_validation",
            "failed",
            False,
            f"shadow={shadow_path} simulated={simulated_path} error={type(exc).__name__}",
            "fix shadow/simulated fill files and import observed fills",
        )
    evidence = (
        f"observed_shadow_rows={len(observed)} required={min_shadow_observations} "
        f"matched={report.matched_observations} validation_passed={int(report.passed)} "
        f"template_exists={int(observed_template_path.exists())} "
        f"order_plan_files={existing_order_plans}/{len(order_plan_paths)} "
        f"nonempty_order_plans={nonempty_order_plans}/{len(order_plan_paths)} "
        f"order_plan_rows={sum(order_plan_rows)} missing_order_plans={missing_order_plan_names} "
        f"invalid_order_plans={invalid_order_plan_names}"
    )
    if report.passed:
        return ReadinessCheck("paper_live_fill_validation", "passed", True, evidence, "run make verify-evidence-gates")
    return ReadinessCheck(
        "paper_live_fill_validation",
        "failed",
        False,
        evidence,
        "tighten simulator assumptions or investigate paper/live fill mismatch",
    )


def _default_shadow_order_plan_paths(observed_template_path: Path) -> tuple[Path, ...]:
    root = observed_template_path.parent
    return tuple(root / filename for filename in DEFAULT_SHADOW_ORDER_PLAN_FILENAMES)


def _count_order_plan_rows(path: Path) -> tuple[int, str | None]:
    if not path.exists() or not path.is_file():
        return 0, None
    try:
        return len(read_order_plan_client_id_map(path)), None
    except Exception as exc:
        return 0, f"{type(exc).__name__}: {exc}"


def _latest_cloud_package(root: Path) -> Path | None:
    packages = sorted((root / "dist").glob("microstructure-alpha-lab-cloud-handoff-*.zip"))
    return packages[-1] if packages else None


def _current_git_head(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    commit = completed.stdout.strip()
    return commit if re.fullmatch(r"[0-9a-fA-F]{40,64}", commit) else None


def _is_excluded_package_name(name: str) -> bool:
    clean = name.lstrip("./")
    excluded_names = {prefix.rstrip("/") for prefix in EXCLUDED_PACKAGE_PREFIXES}
    parts = [part for part in clean.split("/") if part]
    return any(part in excluded_names for part in parts)


def _package_entry_suffix(name: str) -> str:
    parts = [part for part in name.lstrip("./").split("/") if part]
    if not parts:
        return ""
    if parts[-1] == ".source-git-commit":
        return ".source-git-commit"
    if len(parts) >= 2 and parts[-2] == "scripts":
        return f"scripts/{parts[-1]}"
    return parts[-1]


def _read_zip_text(*, archive_path: Path, member: str) -> str:
    with zipfile.ZipFile(archive_path) as archive:
        return archive.read(member).decode("utf-8")


def _resolve(root: Path, path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check operational readiness for the remaining external evidence gates."
    )
    parser.add_argument("--format", choices=["text", "csv", "json"], default="text")
    parser.add_argument("--output")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--modal-binary")
    parser.add_argument("--cloud-package")
    parser.add_argument("--full-plan", default="results/expected_edge_60day_20230516_20230714/run_plan.json")
    parser.add_argument("--full-result-dir", default="results/expected_edge_60day_20230516_20230714")
    parser.add_argument("--min-audit-fold-count", type=int, default=20)
    parser.add_argument("--shadow-decisions", default="results/shadow_validation/shadow_decisions.csv")
    parser.add_argument("--simulated-fills", default="results/shadow_validation/simulated_fills.csv")
    parser.add_argument("--observed-template", default="results/shadow_validation/observed_fills_template.csv")
    parser.add_argument(
        "--order-plan",
        action="append",
        dest="order_plans",
        help="Provider paper-order-plan artifact path. Repeatable; defaults to Bybit/OKX/Binance plans beside the observed-fill template.",
    )
    parser.add_argument("--min-shadow-observations", type=int, default=20)
    parser.add_argument("--max-shadow-price-error", type=float, default=5.0)
    parser.add_argument("--max-shadow-size-error", type=float, default=0.01)
    parser.add_argument("--max-shadow-fill-rate-error", type=float, default=0.05)
    args = parser.parse_args(argv)

    report = evaluate_external_gate_readiness(
        project_root=args.project_root,
        modal_binary=args.modal_binary,
        cloud_package=args.cloud_package,
        full_plan=args.full_plan,
        full_result_dir=args.full_result_dir,
        min_audit_fold_count=args.min_audit_fold_count,
        shadow_decisions=args.shadow_decisions,
        simulated_fills=args.simulated_fills,
        observed_template=args.observed_template,
        order_plans=args.order_plans,
        min_shadow_observations=args.min_shadow_observations,
        max_price_error=args.max_shadow_price_error,
        max_size_error=args.max_shadow_size_error,
        max_fill_rate_error=args.max_shadow_fill_rate_error,
    )
    if args.output:
        output_path = write_external_gate_readiness(report, args.output, output_format=args.format)
        print(f"external_gate_readiness={output_path}")
    print(format_external_gate_readiness(report, output_format=args.format))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
