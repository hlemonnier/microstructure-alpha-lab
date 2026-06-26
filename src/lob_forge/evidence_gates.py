from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from lob_forge.live_validation import read_shadow_decisions, validate_shadow_fill_predictions
from lob_forge.ml_models import (
    L2TensorReadiness,
    ModelReadinessReport,
    evaluate_l2_tensor_readiness,
    evaluate_model_readiness,
)
from lob_forge.portfolio import evaluate_oos_variance_stability
from lob_forge.study_status import evaluate_expected_edge_study_status


DEFAULT_L2_PATHS = (
    "data/normalized_l2/okx/BTC-USDT-SWAP/2023-05-16.csv",
    "data/normalized_l2/bybit/BTCUSDT/2023-05-16.csv",
)
DEFAULT_KELLY_ARTIFACT = Path("results/current/btc_full_day_edge_zero_fee.csv")
DEFAULT_KELLY_CANDIDATE_GLOBS = (
    "results/kelly_candidate_search/*_edge.csv",
    "results/expected_edge_local16_20230516_20230714/*_edge.csv",
)


@dataclass(frozen=True)
class EvidenceGate:
    gate_id: str
    todo_text: str
    status: str
    passed: bool
    evidence: str
    next_action: str


@dataclass(frozen=True)
class EvidenceGateReport:
    gates: tuple[EvidenceGate, ...]

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)


def evaluate_remaining_evidence_gates(
    *,
    capped_plan: Path | str = "results/expected_edge_local16_20230516_20230714/run_plan.json",
    capped_result_dir: Path | str = "results/expected_edge_local16_20230516_20230714",
    full_plan: Path | str = "results/expected_edge_60day_20230516_20230714/run_plan.json",
    full_result_dir: Path | str = "results/expected_edge_60day_20230516_20230714",
    simulated_fills: Path | str = "results/shadow_validation/simulated_fills.csv",
    shadow_decisions: Path | str = "results/shadow_validation/shadow_decisions.csv",
    min_shadow_observations: int = 20,
    max_price_error: float | None = 5.0,
    max_size_error: float | None = 0.01,
    max_fill_rate_error: float | None = 0.05,
    kelly_artifact: Path | str = DEFAULT_KELLY_ARTIFACT,
    kelly_artifacts: Sequence[Path | str] | None = None,
    kelly_column: str = "test_net_pnl",
    kelly_min_observations: int = 20,
    kelly_window_size: int = 5,
    kelly_max_variance_cv: float = 0.5,
    kelly_min_cost_bps: float = 0.05,
    baseline_audit: Path | str = "results/current/btc_full_day_edge_zero_fee_audit.csv",
    l2_path: Path | str | None = None,
    l2_paths: Sequence[Path | str] | None = None,
    min_fold_count: int = 20,
    min_l2_rows: int = 1000,
    transformer_artifact: Path | str = "results/model_experiments/sequence_transformer_results.csv",
    tcn_artifact: Path | str = "results/model_experiments/sequence_tcn_results.csv",
    pretraining_artifact: Path | str = "results/model_experiments/self_supervised_pretraining.csv",
) -> EvidenceGateReport:
    resolved_l2_paths = _resolve_l2_paths(l2_path=l2_path, l2_paths=l2_paths)
    resolved_kelly_artifacts = _resolve_kelly_artifacts(
        single_artifact=Path(kelly_artifact),
        artifacts=kelly_artifacts,
    )
    gates = (
        _study_gate(
            gate_id="capped_60day_btc_eth",
            todo_text="Run the capped 60-day BTC/ETH local profile, or run it on a larger machine if Binance downloads are too slow locally.",
            plan_path=Path(capped_plan),
            result_dir=Path(capped_result_dir),
            min_audit_fold_count=min_fold_count,
        ),
        _study_gate(
            gate_id="full_60_90day_cloud",
            todo_text="Run the full uncapped 60-90 day multi-symbol expected-edge study on a high-RAM/cloud machine.",
            plan_path=Path(full_plan),
            result_dir=Path(full_result_dir),
            min_audit_fold_count=min_fold_count,
        ),
        _shadow_gate(
            simulated_path=Path(simulated_fills),
            shadow_path=Path(shadow_decisions),
            min_shadow_observations=min_shadow_observations,
            max_price_error=max_price_error,
            max_size_error=max_size_error,
            max_fill_rate_error=max_fill_rate_error,
        ),
        _kelly_gate(
            artifact_paths=resolved_kelly_artifacts,
            column=kelly_column,
            min_observations=kelly_min_observations,
            window_size=kelly_window_size,
            max_variance_cv=kelly_max_variance_cv,
            min_cost_bps=kelly_min_cost_bps,
        ),
        _model_experiment_gate(
            gate_id="sequence_transformer_tcn_experiments",
            todo_text="Run Transformer/TCN experiments only after baseline and L2 paths are verified.",
            model_names=("sequence_transformer", "sequence_tcn"),
            baseline_audit=Path(baseline_audit),
            l2_paths=resolved_l2_paths,
            min_fold_count=min_fold_count,
            min_l2_rows=min_l2_rows,
            required_artifacts=(Path(transformer_artifact), Path(tcn_artifact)),
        ),
        _pretraining_gate(
            l2_paths=resolved_l2_paths,
            min_l2_rows=min_l2_rows,
            artifact_path=Path(pretraining_artifact),
        ),
    )
    return EvidenceGateReport(gates=gates)


def format_evidence_gate_report(report: EvidenceGateReport, *, output_format: str = "text") -> str:
    if output_format == "json":
        return json.dumps(
            {"passed": report.passed, "gates": [asdict(gate) for gate in report.gates]}, indent=2, sort_keys=True
        )
    if output_format == "csv":
        fields = ["gate_id", "status", "passed", "evidence", "next_action", "todo_text"]
        lines = [",".join(fields)]
        for gate in report.gates:
            lines.append(
                ",".join(
                    [
                        _csv_cell(gate.gate_id),
                        _csv_cell(gate.status),
                        str(int(gate.passed)),
                        _csv_cell(gate.evidence),
                        _csv_cell(gate.next_action),
                        _csv_cell(gate.todo_text),
                    ]
                )
            )
        return "\n".join(lines)
    if output_format != "text":
        raise ValueError("output_format must be text, csv, or json")
    lines = [f"passed={int(report.passed)}"]
    for gate in report.gates:
        lines.extend(
            [
                f"gate={gate.gate_id}",
                f"status={gate.status}",
                f"passed={int(gate.passed)}",
                f"evidence={gate.evidence}",
                f"next_action={gate.next_action}",
            ]
        )
    return "\n".join(lines)


def write_evidence_gate_report(report: EvidenceGateReport, path: Path | str, *, output_format: str = "json") -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_evidence_gate_report(report, output_format=output_format) + "\n")
    return output_path


def _study_gate(
    *, gate_id: str, todo_text: str, plan_path: Path, result_dir: Path, min_audit_fold_count: int
) -> EvidenceGate:
    if not plan_path.exists():
        return EvidenceGate(
            gate_id,
            todo_text,
            "missing",
            False,
            f"missing run_plan={plan_path}",
            "generate the run plan before running the study",
        )
    try:
        status = evaluate_expected_edge_study_status(
            plan_path=plan_path,
            result_dir=result_dir,
            min_audit_fold_count=min_audit_fold_count,
        )
    except Exception as exc:
        return EvidenceGate(
            gate_id, todo_text, "failed", False, repr(exc), "fix the study status error and rerun the verifier"
        )
    if status.complete:
        evidence = (
            f"profile={status.profile} result_dir={status.result_dir} "
            f"present_results={status.present_result_files}/{status.expected_edge_jobs} "
            f"present_audits={status.present_audit_files}/{status.expected_edge_jobs}"
        )
        return EvidenceGate(gate_id, todo_text, "passed", True, evidence, "checkbox can be marked complete")
    evidence = (
        f"profile={status.profile} result_dir={status.result_dir} "
        f"present_results={status.present_result_files}/{status.expected_edge_jobs} "
        f"present_audits={status.present_audit_files}/{status.expected_edge_jobs} "
        f"missing_results={len(status.missing_result_files)} missing_audits={len(status.missing_audit_files)} "
        f"missing_required={len(status.missing_required_files)} verifier_passed={int(status.artifact_verifier_passed)} "
        f"min_audit_fold_count={min_audit_fold_count}"
    )
    return EvidenceGate(
        gate_id,
        todo_text,
        "not_ready",
        False,
        evidence,
        "finish/resume the expected-edge study and rerun verify_expected_edge_study.sh",
    )


def _shadow_gate(
    *,
    simulated_path: Path,
    shadow_path: Path,
    min_shadow_observations: int,
    max_price_error: float | None,
    max_size_error: float | None,
    max_fill_rate_error: float | None,
) -> EvidenceGate:
    todo = "Run simulated-vs-paper/live fill validation on real shadow or paper observations."
    if not simulated_path.exists() or not shadow_path.exists():
        return EvidenceGate(
            "real_shadow_fill_validation",
            todo,
            "missing",
            False,
            f"simulated_exists={int(simulated_path.exists())} shadow_exists={int(shadow_path.exists())}",
            "run a shadow/paper session, fetch Bybit/OKX demo fills with fetch-observed-fills, import them, then validate",
        )
    try:
        decisions = read_shadow_decisions(shadow_path)
        observed = [
            decision
            for decision in decisions
            if decision.observed_fill_price is not None or decision.observed_fill_size is not None
        ]
        report = validate_shadow_fill_predictions(
            simulated_path=simulated_path,
            shadow_path=shadow_path,
            max_price_error=max_price_error,
            max_size_error=max_size_error,
            max_fill_rate_error=max_fill_rate_error,
        )
    except Exception as exc:
        return EvidenceGate(
            "real_shadow_fill_validation",
            todo,
            "failed",
            False,
            repr(exc),
            "fix shadow/simulated files, fetch/normalize/import demo fills, and rerun validate-shadow-fills",
        )
    if len(observed) < min_shadow_observations:
        evidence = f"observed_shadow_rows={len(observed)} required={min_shadow_observations} matched={report.matched_observations} validation_passed={int(report.passed)}"
        return EvidenceGate(
            "real_shadow_fill_validation",
            todo,
            "not_ready",
            False,
            evidence,
            "fetch Bybit/OKX demo fills with fetch-observed-fills, normalize/import them, then rerun validate-shadow-fills",
        )
    evidence = (
        f"observed_shadow_rows={len(observed)} matched={report.matched_observations} "
        f"mean_abs_price_error={report.validation.mean_abs_price_error:.12g} "
        f"mean_abs_size_error={report.validation.mean_abs_size_error:.12g} "
        f"fill_rate_error={report.validation.fill_rate_error:.12g} "
        f"max_price_error={max_price_error if max_price_error is not None else 'none'} "
        f"max_size_error={max_size_error if max_size_error is not None else 'none'} "
        f"max_fill_rate_error={max_fill_rate_error if max_fill_rate_error is not None else 'none'}"
    )
    return EvidenceGate(
        "real_shadow_fill_validation",
        todo,
        "passed" if report.passed else "failed",
        report.passed,
        evidence,
        "checkbox can be marked complete"
        if report.passed
        else "tighten simulator assumptions or investigate paper/live fill mismatch",
    )


def _kelly_gate(
    *,
    artifact_paths: tuple[Path, ...],
    column: str,
    min_observations: int,
    window_size: int,
    max_variance_cv: float,
    min_cost_bps: float,
) -> EvidenceGate:
    todo = "Enable fractional Kelly on real strategy artifacts only after the OOS variance-stability gate passes."
    if not artifact_paths:
        return EvidenceGate(
            "kelly_variance_stability",
            todo,
            "missing",
            False,
            "missing artifact candidates",
            "run accepted strategy artifacts first",
        )
    candidate_details = []
    variance_passed_candidates = 0
    for artifact_path in artifact_paths:
        if not artifact_path.exists():
            candidate_details.append(f"artifact={artifact_path} present=0")
            continue
        try:
            report = evaluate_oos_variance_stability(
                artifact_path,
                column=column,
                min_observations=min_observations,
                window_size=window_size,
                max_variance_cv=max_variance_cv,
            )
        except Exception as exc:
            candidate_details.append(f"artifact={artifact_path} error={type(exc).__name__}")
            continue
        audit_path = _audit_path_for_edge_artifact(artifact_path)
        audit_passed, audit_detail = _audit_acceptance_detail(audit_path)
        fee_bps = _fee_bps_from_artifact_name(artifact_path)
        fee_eligible = fee_bps is not None and fee_bps >= min_cost_bps
        if report.passed:
            variance_passed_candidates += 1
        detail = (
            f"artifact={artifact_path} audit={audit_path} observations={report.observations} "
            f"window_count={len(report.window_variances)} variance_cv={report.variance_cv:.12g} "
            f"variance_passed={int(report.passed)} fee_bps={fee_bps if fee_bps is not None else 'unknown'} "
            f"min_cost_bps={min_cost_bps:.12g} fee_eligible={int(fee_eligible)} {audit_detail}"
        )
        if report.passed and audit_passed and fee_eligible:
            return EvidenceGate(
                "kelly_variance_stability",
                todo,
                "passed",
                True,
                detail,
                "checkbox can be marked complete",
            )
        candidate_details.append(detail)
    evidence = (
        f"candidates={len(artifact_paths)} variance_passed_candidates={variance_passed_candidates} "
        + " | ".join(candidate_details[:5])
    )
    if len(candidate_details) > 5:
        evidence += f" | more_candidates={len(candidate_details) - 5}"
    return EvidenceGate(
        "kelly_variance_stability",
        todo,
        "not_ready",
        False,
        evidence,
        "keep Kelly disabled until the same nonzero-cost strategy artifact passes audit acceptance and variance stability",
    )


def _model_experiment_gate(
    *,
    gate_id: str,
    todo_text: str,
    model_names: tuple[str, ...],
    baseline_audit: Path,
    l2_paths: tuple[Path, ...],
    min_fold_count: int,
    min_l2_rows: int,
    required_artifacts: tuple[Path, ...],
) -> EvidenceGate:
    if not baseline_audit.exists():
        return EvidenceGate(
            gate_id,
            todo_text,
            "missing",
            False,
            f"missing baseline_audit={baseline_audit}",
            "produce accepted baseline audit before model experiments",
        )
    existing_l2_paths = tuple(path for path in l2_paths if path.exists())
    if not existing_l2_paths:
        return EvidenceGate(
            gate_id,
            todo_text,
            "missing",
            False,
            f"missing l2_paths={','.join(str(path) for path in l2_paths)}",
            "import verified normalized L2 rows before model experiments",
        )
    missing_artifacts = [path for path in required_artifacts if not path.exists() or path.stat().st_size == 0]
    candidate_readiness: list[tuple[Path, bool, tuple[ModelReadinessReport, ...], str]] = []
    for candidate_l2_path in existing_l2_paths:
        readiness: list[ModelReadinessReport] = []
        for model_name in model_names:
            try:
                readiness.append(
                    evaluate_model_readiness(
                        model_name=model_name,
                        baseline_audit_path=baseline_audit,
                        l2_path=candidate_l2_path,
                        min_fold_count=min_fold_count,
                        min_l2_rows=min_l2_rows,
                    )
                )
            except Exception as exc:
                candidate_readiness.append((candidate_l2_path, False, (), repr(exc)))
                break
        else:
            candidate_readiness.append(
                (candidate_l2_path, all(report.passed for report in readiness), tuple(readiness), "")
            )
    ready_candidate = next((candidate for candidate in candidate_readiness if candidate[1]), None)
    selected_l2_path, ready, selected_readiness, failure = ready_candidate or candidate_readiness[0]
    if failure:
        return EvidenceGate(
            gate_id,
            todo_text,
            "not_ready",
            False,
            failure,
            "produce accepted baseline audit and verified normalized L2 rows before model experiments",
        )
    artifact_checks = tuple(
        _model_experiment_artifact_status(path, expected_model=model_name, selected_l2_path=selected_l2_path)
        for model_name, path in zip(model_names, required_artifacts)
    )
    artifacts_passed = all(passed for passed, _ in artifact_checks)
    if ready and artifacts_passed:
        evidence = (
            f"models={','.join(model_names)} l2_path={selected_l2_path} "
            f"{' '.join(detail for _, detail in artifact_checks)}"
        )
        return EvidenceGate(gate_id, todo_text, "passed", True, evidence, "checkbox can be marked complete")
    evidence = (
        f"l2_path={selected_l2_path} readiness_passed={int(ready)} "
        f"model_reasons={' | '.join(';'.join(report.reasons) or 'ready' for report in selected_readiness)} "
        f"missing_artifacts={','.join(str(path) for path in missing_artifacts) or 'none'} "
        f"artifact_checks={' | '.join(detail for _, detail in artifact_checks)}"
    )
    return EvidenceGate(
        gate_id,
        todo_text,
        "not_ready",
        False,
        evidence,
        "satisfy model-readiness-gate, run experiments, and save result artifacts",
    )


def _pretraining_gate(*, l2_paths: tuple[Path, ...], min_l2_rows: int, artifact_path: Path) -> EvidenceGate:
    todo = "Run self-supervised order-book pretraining only after true L2 tensor data exists."
    existing_l2_paths = tuple(path for path in l2_paths if path.exists())
    if not existing_l2_paths:
        return EvidenceGate(
            "self_supervised_l2_pretraining",
            todo,
            "missing",
            False,
            f"missing l2_paths={','.join(str(path) for path in l2_paths)}",
            "import verified true L2 tensor rows before pretraining",
        )
    candidate_results: list[tuple[Path, L2TensorReadiness | None, str]] = []
    for candidate_l2_path in existing_l2_paths:
        try:
            l2 = evaluate_l2_tensor_readiness(
                candidate_l2_path, min_rows=min_l2_rows, require_delta=True, allow_fi2010=False
            )
        except Exception as exc:
            candidate_results.append((candidate_l2_path, None, repr(exc)))
        else:
            candidate_results.append((candidate_l2_path, l2, ""))
    passing_candidate = next(
        (candidate for candidate in candidate_results if candidate[1] is not None and candidate[1].passed), None
    )
    selected_l2_path, selected_l2, failure = passing_candidate or candidate_results[0]
    if selected_l2 is None:
        return EvidenceGate(
            "self_supervised_l2_pretraining",
            todo,
            "not_ready",
            False,
            failure,
            "import verified true L2 tensor rows before pretraining",
        )
    artifact_pipeline_completed, artifact_evidence = _pretraining_artifact_status(
        artifact_path, selected_l2_path=selected_l2_path
    )
    if selected_l2.passed and artifact_pipeline_completed:
        evidence = f"l2_path={selected_l2_path} l2_rows_checked={selected_l2.rows_checked} {artifact_evidence}"
        return EvidenceGate(
            "self_supervised_l2_pretraining", todo, "passed", True, evidence, "checkbox can be marked complete"
        )
    evidence = (
        f"l2_path={selected_l2_path} l2_passed={int(selected_l2.passed)} rows_checked={selected_l2.rows_checked} "
        f"has_delta={int(selected_l2.has_delta)} has_sequence={int(selected_l2.has_sequence)} {artifact_evidence}"
    )
    return EvidenceGate(
        "self_supervised_l2_pretraining",
        todo,
        "not_ready",
        False,
        evidence,
        "import true L2 rows, run pretraining, and save the artifact",
    )


def _model_experiment_artifact_status(
    artifact_path: Path, *, expected_model: str, selected_l2_path: Path
) -> tuple[bool, str]:
    prefix = f"artifact={artifact_path}"
    if not artifact_path.exists() or artifact_path.stat().st_size == 0:
        return False, f"{prefix} present=0 expected_model={expected_model}"
    try:
        with artifact_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:
        return False, f"{prefix} readable=0 error={type(exc).__name__}"
    if len(rows) != 1:
        return False, f"{prefix} row_count={len(rows)} expected_model={expected_model}"
    row = rows[0]
    model_match = row.get("model_name") == expected_model
    l2_match = row.get("l2_path") == str(selected_l2_path)
    pipeline_completed = row.get("pipeline_completed") in {"1", "true", "True"}
    readiness_passed = row.get("readiness_passed") in {"1", "true", "True"}
    dependency_available = row.get("dependency_available") in {"1", "true", "True"}
    ok = model_match and l2_match and pipeline_completed and readiness_passed and dependency_available
    detail = (
        f"{prefix} present=1 expected_model={expected_model} model_match={int(model_match)} "
        f"l2_match={int(l2_match)} readiness_passed={int(readiness_passed)} "
        f"dependency_available={int(dependency_available)} pipeline_completed={int(pipeline_completed)}"
    )
    return ok, detail


def _pretraining_artifact_status(artifact_path: Path, *, selected_l2_path: Path) -> tuple[bool, str]:
    if not artifact_path.exists() or artifact_path.stat().st_size == 0:
        return False, f"artifact_present=0 artifact={artifact_path}"
    try:
        with artifact_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:
        return False, f"artifact_present=1 artifact_read_error={exc!r} artifact={artifact_path}"
    if not rows:
        return False, f"artifact_present=1 artifact_rows=0 artifact={artifact_path}"
    row = rows[0]
    pipeline_completed = _truthy_csv_value(row.get("pipeline_completed", row.get("passed", "0")))
    artifact_l2_path = row.get("l2_path", "")
    l2_match = artifact_l2_path == str(selected_l2_path)
    evidence = (
        f"artifact_present=1 pipeline_completed={int(pipeline_completed)} "
        f"artifact_l2_match={int(l2_match)} artifact={artifact_path}"
    )
    return pipeline_completed and l2_match, evidence


def _truthy_csv_value(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    if normalized in {"true", "yes"}:
        return True
    try:
        return bool(int(float(normalized)))
    except ValueError:
        return False


def _resolve_l2_paths(*, l2_path: Path | str | None, l2_paths: Sequence[Path | str] | None) -> tuple[Path, ...]:
    raw_paths: list[Path | str] = []
    if l2_path is not None:
        raw_paths.append(l2_path)
    if l2_paths is not None:
        raw_paths.extend(l2_paths)
    if not raw_paths:
        raw_paths.extend(DEFAULT_L2_PATHS)

    resolved: list[Path] = []
    seen: set[str] = set()
    for raw_path in raw_paths:
        path = Path(raw_path)
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(path)
    return tuple(resolved)


def _resolve_kelly_artifacts(*, single_artifact: Path, artifacts: Sequence[Path | str] | None) -> tuple[Path, ...]:
    if artifacts is not None:
        return tuple(Path(path) for path in artifacts)
    if single_artifact == DEFAULT_KELLY_ARTIFACT:
        candidates = tuple(
            candidate for pattern in DEFAULT_KELLY_CANDIDATE_GLOBS for candidate in sorted(Path(".").glob(pattern))
        )
        if candidates:
            return candidates
    return (single_artifact,)


def _audit_path_for_edge_artifact(artifact_path: Path) -> Path:
    if artifact_path.name.endswith("_edge.csv"):
        return artifact_path.with_name(artifact_path.name[: -len("_edge.csv")] + "_edge_audit.csv")
    return artifact_path.with_name(artifact_path.stem + "_audit.csv")


def _audit_acceptance_detail(audit_path: Path) -> tuple[bool, str]:
    if not audit_path.exists() or audit_path.stat().st_size == 0:
        return False, "audit_passed=0 audit_present=0"
    try:
        with audit_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:
        return False, f"audit_passed=0 audit_error={type(exc).__name__}"
    if not rows:
        return False, "audit_passed=0 audit_rows=0"
    row = rows[0]
    passed = _truthy_csv_value(row.get("acceptance_passed", "0"))
    reasons = (row.get("rejection_reasons", "") or "").replace("|", ";").strip()
    return passed, f"audit_passed={int(passed)} rejection_reasons={reasons or 'none'}"


def _fee_bps_from_artifact_name(artifact_path: Path) -> float | None:
    name = artifact_path.name
    if "zero_fee" in name:
        return 0.0
    match = re.search(r"fee_([0-9]+(?:p[0-9]+)?)", name)
    if not match:
        return None
    try:
        return float(match.group(1).replace("p", "."))
    except ValueError:
        return None


def _csv_cell(value: str) -> str:
    from io import StringIO

    handle = StringIO()
    writer = csv.writer(handle)
    writer.writerow([value])
    return handle.getvalue().strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check evidence gates for the remaining implementation TODOs.")
    parser.add_argument("--format", choices=["text", "csv", "json"], default="text")
    parser.add_argument("--output")
    parser.add_argument("--min-shadow-observations", type=int, default=20)
    parser.add_argument("--simulated-fills", default="results/shadow_validation/simulated_fills.csv")
    parser.add_argument("--shadow-decisions", default="results/shadow_validation/shadow_decisions.csv")
    parser.add_argument("--max-shadow-price-error", type=float, default=5.0)
    parser.add_argument("--max-shadow-size-error", type=float, default=0.01)
    parser.add_argument("--max-shadow-fill-rate-error", type=float, default=0.05)
    parser.add_argument(
        "--l2",
        dest="l2_paths",
        action="append",
        help="Normalized L2 candidate path. Repeatable; defaults to the OKX and Bybit BTC smoke paths.",
    )
    parser.add_argument("--baseline-audit", default="results/current/btc_full_day_edge_zero_fee_audit.csv")
    parser.add_argument("--kelly-artifact", default="results/current/btc_full_day_edge_zero_fee.csv")
    parser.add_argument("--kelly-min-cost-bps", type=float, default=0.05)
    parser.add_argument(
        "--kelly-candidate",
        dest="kelly_artifacts",
        action="append",
        help="Kelly candidate result artifact. Repeatable; defaults to local16 edge artifacts when present.",
    )
    args = parser.parse_args(argv)

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
        kelly_artifacts=args.kelly_artifacts,
        kelly_min_cost_bps=args.kelly_min_cost_bps,
    )
    if args.output:
        output_path = write_evidence_gate_report(report, args.output, output_format=args.format)
        print(f"evidence_gates={output_path}")
    print(format_evidence_gate_report(report, output_format=args.format))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
