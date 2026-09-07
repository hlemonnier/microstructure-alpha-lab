from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from lob_forge.holdout import sha256_file, verify_holdout_manifest_file, verify_final_holdout_result
from lob_forge.live_validation import has_observed_fill, read_shadow_decisions, validate_shadow_fill_predictions
from lob_forge.ml_models import (
    L2TensorReadiness,
    ModelReadinessReport,
    SEQUENCE_ECONOMICS_VERSION,
    evaluate_l2_tensor_readiness,
    evaluate_model_readiness,
)
from lob_forge.portfolio import evaluate_oos_variance_stability
from lob_forge.provider_order_ids import read_order_plan_client_id_map
from lob_forge.study_status import evaluate_expected_edge_study_status
from lob_forge.study_provenance import provenance_path_for, verify_recorded_expected_edge_provenance


DEFAULT_L2_PATHS = (
    "data/normalized_l2/okx/BTC-USDT-SWAP/2023-05-16.csv",
    "data/normalized_l2/bybit/BTCUSDT/2023-05-16.csv",
)
DEFAULT_KELLY_ARTIFACT = Path("results/current/btc_full_day_edge_zero_fee.csv")
DEFAULT_KELLY_CANDIDATE_GLOBS = (
    "results/kelly_candidate_search/*_edge.csv",
    "results/expected_edge_local16_20230516_20230714/*_edge.csv",
)
DEFAULT_FINAL_HOLDOUT_RESULT_NAME = "final_holdout_result.json"
DEFAULT_FINAL_HOLDOUT_EDGE_RESULT_NAME = "final_holdout_edge_result.json"
DEFAULT_FINAL_HOLDOUT_SEQUENCE_RESULT_NAME = "final_holdout_sequence_result.json"
DEFAULT_SHADOW_ORDER_PLAN_FILENAMES = (
    "bybit_order_plan.jsonl",
    "okx_order_plan.jsonl",
    "binance_usdm_order_plan.jsonl",
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
    order_plans: Sequence[Path | str] | None = None,
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
    final_holdout_result: Path | str = "results/final_holdout/final_holdout_result.json",
    research_manifest: Path | str = "artifacts/research_manifest.json",
) -> EvidenceGateReport:
    resolved_l2_paths = _resolve_l2_paths(l2_path=l2_path, l2_paths=l2_paths)
    resolved_kelly_artifacts = _resolve_kelly_artifacts(
        single_artifact=Path(kelly_artifact),
        artifacts=kelly_artifacts,
    )
    shadow_path = Path(shadow_decisions)
    order_plan_paths = (
        tuple(Path(path) for path in order_plans)
        if order_plans is not None
        else _default_shadow_order_plan_paths(shadow_path)
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
        _final_holdout_gate(
            result_paths=_final_holdout_result_candidates(Path(final_holdout_result)),
            research_manifest_path=Path(research_manifest),
        ),
        _shadow_gate(
            simulated_path=Path(simulated_fills),
            shadow_path=shadow_path,
            order_plan_paths=order_plan_paths,
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


def _final_holdout_result_candidates(result_path: Path) -> tuple[Path, ...]:
    paths = [result_path]
    if result_path.name == DEFAULT_FINAL_HOLDOUT_RESULT_NAME:
        paths.extend(
            [
                result_path.with_name(DEFAULT_FINAL_HOLDOUT_EDGE_RESULT_NAME),
                result_path.with_name(DEFAULT_FINAL_HOLDOUT_SEQUENCE_RESULT_NAME),
            ]
        )
    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    return tuple(unique_paths)


def _final_holdout_gate(*, result_paths: Sequence[Path], research_manifest_path: Path) -> EvidenceGate:
    todo = "Run and check in the declared immutable final holdout result artifact for the full selected candidate."
    existing_paths = tuple(path for path in result_paths if path.exists() and path.stat().st_size > 0)
    if not existing_paths:
        manifest_status = "unknown"
        manifest_reason = ""
        if research_manifest_path.exists():
            try:
                manifest_payload = json.loads(research_manifest_path.read_text())
                final_holdout = manifest_payload.get("final_holdout", {})
                if isinstance(final_holdout, dict):
                    manifest_status = str(final_holdout.get("status", "unknown"))
                    manifest_reason = str(final_holdout.get("reason", ""))
            except Exception as exc:
                manifest_status = f"research_manifest_error={type(exc).__name__}"
        result_paths_text = ",".join(str(path) for path in result_paths)
        evidence = (
            f"result_exists=0 result_paths={result_paths_text} research_manifest={research_manifest_path} "
            f"manifest_final_holdout_status={manifest_status} reason={manifest_reason or 'none'}"
        )
        return EvidenceGate(
            "immutable_final_holdout",
            todo,
            "not_ready",
            False,
            evidence,
            "after the full study selects one frozen candidate, run final-holdout-rule, final-holdout-edge, or final-holdout-sequence with a pre-registered candidate hash",
        )
    failed_gates: list[EvidenceGate] = []
    for result_path in existing_paths:
        gate = _final_holdout_result_gate_for_path(result_path=result_path, todo=todo)
        if gate.passed:
            return gate
        failed_gates.append(gate)
    evidence = f"result_candidates={','.join(str(path) for path in result_paths)} " + " | ".join(
        gate.evidence for gate in failed_gates
    )
    return EvidenceGate(
        "immutable_final_holdout",
        todo,
        "failed",
        False,
        evidence,
        "rerun final-holdout-rule, final-holdout-edge, or final-holdout-sequence from the verified manifest and frozen candidate, preserving the immutable result",
    )


def _final_holdout_result_gate_for_path(*, result_path: Path, todo: str) -> EvidenceGate:
    try:
        payload = json.loads(result_path.read_text())
    except Exception as exc:
        return EvidenceGate(
            "immutable_final_holdout",
            todo,
            "failed",
            False,
            f"result_path={result_path} parse_error={type(exc).__name__}",
            "fix the final holdout JSON artifact and rerun the evidence gate",
        )
    manifest = payload.get("manifest", {})
    metrics = payload.get("metrics", {})
    candidate_sha256 = str(payload.get("candidate_sha256", ""))
    manifest_candidate_sha256 = str(manifest.get("candidate_sha256", "")) if isinstance(manifest, dict) else ""
    metrics_candidate_sha256 = str(metrics.get("candidate_sha256", "")) if isinstance(metrics, dict) else ""
    final_evaluation = payload.get("final_evaluation") is True
    stateful_simulator = bool(metrics.get("stateful_simulator")) if isinstance(metrics, dict) else False
    rows = _numeric_metric(metrics, "rows")
    candidate_type = str(metrics.get("candidate_type", "")) if isinstance(metrics, dict) else ""
    sequence_contract = True
    if candidate_type.startswith("l2_sequence_torch_"):
        try:
            final_inventory = float(metrics["stateful_final_inventory"])
        except (KeyError, TypeError, ValueError):
            final_inventory = math.nan
        sequence_contract = (
            candidate_type == "l2_sequence_torch_v3"
            and metrics.get("economic_simulation_version") == SEQUENCE_ECONOMICS_VERSION
            and math.isfinite(final_inventory)
            and abs(final_inventory) <= 1e-9
        )
        candidate = metrics.get("candidate", {})
        sequence_contract = sequence_contract and isinstance(candidate, dict) and all(
            name in candidate for name in ("economic_latency_ms", "policy_json", "class_priors_json")
        )
    provenance_valid, provenance_errors = verify_final_holdout_result(result_path)
    checks = {
        "final_evaluation": final_evaluation,
        "candidate_sha256": bool(re.fullmatch(r"[0-9a-fA-F]{64}", candidate_sha256)),
        "manifest_candidate_match": bool(candidate_sha256 and candidate_sha256 == manifest_candidate_sha256),
        "metrics_candidate_match": bool(candidate_sha256 and candidate_sha256 == metrics_candidate_sha256),
        "stateful_simulator": stateful_simulator,
        "rows": rows > 0,
        "sequence_contract": sequence_contract,
        "holdout_provenance": provenance_valid,
    }
    evidence = (
        f"result_path={result_path} "
        f"final_evaluation={int(final_evaluation)} "
        f"candidate_sha256={int(checks['candidate_sha256'])} "
        f"manifest_candidate_match={int(checks['manifest_candidate_match'])} "
        f"metrics_candidate_match={int(checks['metrics_candidate_match'])} "
        f"stateful_simulator={int(stateful_simulator)} rows={rows:.12g} "
        f"sequence_contract={int(sequence_contract)} "
        f"holdout_provenance={int(provenance_valid)} errors={';'.join(provenance_errors) or 'none'}"
    )
    if all(checks.values()):
        return EvidenceGate(
            "immutable_final_holdout", todo, "passed", True, evidence, "checkbox can be marked complete"
        )
    failed = ",".join(name for name, passed in checks.items() if not passed)
    return EvidenceGate(
        "immutable_final_holdout",
        todo,
        "failed",
        False,
        f"{evidence} failed_checks={failed}",
        "rerun final-holdout-rule, final-holdout-edge, or final-holdout-sequence from the verified manifest and frozen candidate, preserving the immutable result",
    )


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
            f"present_audits={status.present_audit_files}/{status.expected_edge_jobs} "
            "candidate_registry=1"
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
    order_plan_paths: Sequence[Path],
    min_shadow_observations: int,
    max_price_error: float | None,
    max_size_error: float | None,
    max_fill_rate_error: float | None,
) -> EvidenceGate:
    todo = "Run simulated-vs-paper/live fill validation on real shadow or paper observations."
    if not simulated_path.exists() or not shadow_path.exists():
        (
            existing_order_plans,
            nonempty_order_plans,
            order_plan_rows,
            missing_order_plan_names,
            invalid_order_plan_names,
        ) = _order_plan_evidence(order_plan_paths)
        return EvidenceGate(
            "real_shadow_fill_validation",
            todo,
            "missing",
            False,
            f"simulated_exists={int(simulated_path.exists())} shadow_exists={int(shadow_path.exists())} "
            f"order_plan_files={existing_order_plans}/{len(order_plan_paths)} "
            f"nonempty_order_plans={nonempty_order_plans}/{len(order_plan_paths)} "
            f"order_plan_rows={order_plan_rows} missing_order_plans={missing_order_plan_names} "
            f"invalid_order_plans={invalid_order_plan_names}",
            "run edge-shadow-decisions, generate paper-order-plan, submit-paper-orders with --execute on demo/testnet, fetch order/fill history, normalize/import observations, then validate",
        )
    (
        existing_order_plans,
        nonempty_order_plans,
        order_plan_rows,
        missing_order_plan_names,
        invalid_order_plan_names,
    ) = _order_plan_evidence(order_plan_paths)
    try:
        decisions = read_shadow_decisions(shadow_path)
        observed = [decision for decision in decisions if has_observed_fill(decision)]
    except Exception as exc:
        return EvidenceGate(
            "real_shadow_fill_validation",
            todo,
            "failed",
            False,
            repr(exc),
            "fix shadow decision files, fetch order/fill history, normalize/import demo observations, and rerun validate-shadow-fills",
        )
    if len(observed) < min_shadow_observations:
        evidence = (
            f"observed_shadow_rows={len(observed)} required={min_shadow_observations} "
            "matched=0 validation_passed=0 "
            f"order_plan_files={existing_order_plans}/{len(order_plan_paths)} "
            f"nonempty_order_plans={nonempty_order_plans}/{len(order_plan_paths)} "
            f"order_plan_rows={order_plan_rows} missing_order_plans={missing_order_plan_names} "
            f"invalid_order_plans={invalid_order_plan_names}"
        )
        next_action = (
            "generate paper-order-plan, run submit-paper-orders with --execute on Bybit/OKX demo or Binance USD-M testnet, fetch order/fill history, normalize/import observations, then rerun validate-shadow-fills"
            if existing_order_plans < len(order_plan_paths) or nonempty_order_plans == 0
            else "run submit-paper-orders with --execute on demo/testnet, fetch order/fill history, normalize/import observations, then rerun validate-shadow-fills"
        )
        return EvidenceGate(
            "real_shadow_fill_validation",
            todo,
            "not_ready",
            False,
            evidence,
            next_action,
        )
    try:
        report = validate_shadow_fill_predictions(
            simulated_path=simulated_path,
            shadow_path=shadow_path,
            max_price_error=max_price_error,
            max_size_error=max_size_error,
            max_fill_rate_error=max_fill_rate_error,
            min_observed_coverage=1.0,
            min_observations=min_shadow_observations,
        )
    except Exception as exc:
        return EvidenceGate(
            "real_shadow_fill_validation",
            todo,
            "failed",
            False,
            repr(exc),
            "fix shadow/simulated files, fetch order/fill history, normalize/import demo observations, and rerun validate-shadow-fills",
        )
    evidence = (
        f"observed_shadow_rows={len(observed)} matched={report.matched_observations} "
        f"mean_abs_price_error={report.validation.mean_abs_price_error:.12g} "
        f"mean_abs_size_error={report.validation.mean_abs_size_error:.12g} "
        f"fill_rate_error={report.validation.fill_rate_error:.12g} "
        f"fill_mismatch_rate={report.validation.fill_mismatch_rate:.12g} "
        f"observed_coverage={report.observed_coverage:.12g} unobserved_decisions={len(report.unobserved_decisions)} "
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


def _default_shadow_order_plan_paths(shadow_path: Path) -> tuple[Path, ...]:
    root = shadow_path.parent
    return tuple(root / filename for filename in DEFAULT_SHADOW_ORDER_PLAN_FILENAMES)


def _order_plan_evidence(order_plan_paths: Sequence[Path]) -> tuple[int, int, int, str, str]:
    rows_and_errors = tuple(_count_order_plan_rows(path) for path in order_plan_paths)
    rows_by_path = tuple(rows for rows, _error in rows_and_errors)
    existing_order_plans = sum(1 for path in order_plan_paths if path.exists())
    nonempty_order_plans = sum(1 for rows in rows_by_path if rows > 0)
    missing_order_plan_names = ",".join(path.name for path in order_plan_paths if not path.exists()) or "none"
    invalid_order_plan_names = (
        ",".join(path.name for path, (_rows, error) in zip(order_plan_paths, rows_and_errors) if error is not None)
        or "none"
    )
    return (
        existing_order_plans,
        nonempty_order_plans,
        sum(rows_by_path),
        missing_order_plan_names,
        invalid_order_plan_names,
    )


def _count_order_plan_rows(path: Path) -> tuple[int, str | None]:
    if not path.exists() or not path.is_file():
        return 0, None
    try:
        return len(read_order_plan_client_id_map(path)), None
    except Exception as exc:
        return 0, f"{type(exc).__name__}: {exc}"


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
        provenance = verify_recorded_expected_edge_provenance(
            provenance_path_for(artifact_path),
            result_path=artifact_path,
            audit_path=audit_path,
            require_source_files=True,
        )
        fee_bps = _fee_bps_from_artifact_name(artifact_path)
        fee_eligible = fee_bps is not None and fee_bps >= min_cost_bps
        if report.passed:
            variance_passed_candidates += 1
        detail = (
            f"artifact={artifact_path} audit={audit_path} observations={report.observations} "
            f"window_count={len(report.window_variances)} variance_cv={report.variance_cv:.12g} "
            f"variance_passed={int(report.passed)} fee_bps={fee_bps if fee_bps is not None else 'unknown'} "
            f"min_cost_bps={min_cost_bps:.12g} fee_eligible={int(fee_eligible)} {audit_detail} "
            f"provenance_passed={int(provenance.passed)} "
            f"provenance_errors={';'.join(provenance.errors) or 'none'}"
        )
        if report.passed and audit_passed and fee_eligible and provenance.passed:
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
        "keep Kelly disabled until the same nonzero-cost strategy artifact passes provenance, audit acceptance, and variance stability",
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
    baseline_provenance_passed, baseline_provenance_detail = _baseline_provenance_status(baseline_audit)
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
        verify_l2_sequence_experiment_artifact(
            path,
            expected_model=model_name,
            selected_l2_path=selected_l2_path,
            baseline_audit=baseline_audit,
        )
        for model_name, path in zip(model_names, required_artifacts)
    )
    artifacts_passed = all(passed for passed, _ in artifact_checks)
    if ready and artifacts_passed and baseline_provenance_passed:
        evidence = (
            f"models={','.join(model_names)} l2_path={selected_l2_path} "
            f"baseline_provenance_passed=1 {baseline_provenance_detail} "
            f"{' '.join(detail for _, detail in artifact_checks)}"
        )
        return EvidenceGate(gate_id, todo_text, "passed", True, evidence, "checkbox can be marked complete")
    evidence = (
        f"l2_path={selected_l2_path} readiness_passed={int(ready)} "
        f"baseline_provenance_passed={int(baseline_provenance_passed)} {baseline_provenance_detail} "
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
        "produce a provenance-valid baseline, use a verified development holdout, and regenerate versioned sequence artifacts with hashed checkpoints and predictions",
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


def verify_l2_sequence_experiment_artifact(
    artifact_path: Path,
    *,
    expected_model: str,
    selected_l2_path: Path,
    baseline_audit: Path,
    expected_fields: Mapping[str, Any] | None = None,
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
    l2_hash_match = l2_match and selected_l2_path.is_file() and row.get("l2_sha256") == sha256_file(selected_l2_path)
    baseline_match = row.get("baseline_audit_path") == str(baseline_audit)
    baseline_hash_match = (
        baseline_match and baseline_audit.is_file() and row.get("baseline_audit_sha256") == sha256_file(baseline_audit)
    )
    pipeline_completed = row.get("pipeline_completed") in {"1", "true", "True"}
    readiness_passed = row.get("readiness_passed") in {"1", "true", "True"}
    dependency_available = row.get("dependency_available") in {"1", "true", "True"}
    semantics_match = row.get("economic_simulation_version") == SEQUENCE_ECONOMICS_VERSION
    try:
        final_inventory_flat = abs(float(row.get("test_stateful_final_inventory", "nan"))) <= 1e-9
        source_rows = int(row.get("source_rows_before_holdout_filter", "0"))
        development_rows = int(row.get("development_rows_after_holdout_filter", "0"))
        excluded_rows = int(row.get("holdout_rows_excluded", "0"))
    except ValueError:
        final_inventory_flat = False
        source_rows = development_rows = excluded_rows = 0
    holdout_verified = row.get("holdout_manifest_verified") in {"1", "true", "True"}
    holdout_hash_match = _artifact_hash_matches(
        row.get("holdout_manifest_path", ""),
        row.get("holdout_manifest_sha256", ""),
    )
    holdout_source_verified = holdout_hash_match and _holdout_manifest_verifies(row.get("holdout_manifest_path", ""))
    development_l2_present = _artifact_path_exists(row.get("development_l2_path", ""))
    development_l2_hash_match = _artifact_hash_matches(
        row.get("development_l2_path", ""),
        row.get("development_l2_sha256", ""),
    )
    holdout_partition_valid = excluded_rows > 0 and source_rows == development_rows + excluded_rows
    checkpoint_hash_match = _artifact_hash_matches(
        row.get("checkpoint_path", ""),
        row.get("checkpoint_sha256", ""),
    )
    prediction_hash_match = _artifact_hash_matches(
        row.get("prediction_output_path", ""),
        row.get("prediction_output_sha256", ""),
    )
    config_mismatches = sorted(
        field for field, expected in (expected_fields or {}).items() if not _csv_value_matches(row.get(field), expected)
    )
    config_match = not config_mismatches
    ok = (
        model_match
        and l2_hash_match
        and baseline_hash_match
        and pipeline_completed
        and readiness_passed
        and dependency_available
        and semantics_match
        and final_inventory_flat
        and holdout_verified
        and holdout_hash_match
        and holdout_source_verified
        and development_l2_present
        and development_l2_hash_match
        and holdout_partition_valid
        and checkpoint_hash_match
        and prediction_hash_match
        and config_match
    )
    detail = (
        f"{prefix} present=1 expected_model={expected_model} model_match={int(model_match)} "
        f"l2_match={int(l2_match)} l2_hash_match={int(l2_hash_match)} "
        f"baseline_hash_match={int(baseline_hash_match)} "
        f"readiness_passed={int(readiness_passed)} "
        f"dependency_available={int(dependency_available)} pipeline_completed={int(pipeline_completed)} "
        f"semantics_match={int(semantics_match)} final_inventory_flat={int(final_inventory_flat)} "
        f"holdout_verified={int(holdout_verified)} holdout_hash_match={int(holdout_hash_match)} "
        f"holdout_source_verified={int(holdout_source_verified)} "
        f"holdout_partition_valid={int(holdout_partition_valid)} "
        f"development_l2_present={int(development_l2_present)} "
        f"development_l2_hash_match={int(development_l2_hash_match)} "
        f"checkpoint_hash_match={int(checkpoint_hash_match)} prediction_hash_match={int(prediction_hash_match)} "
        f"config_match={int(config_match)} "
        f"config_mismatches={','.join(config_mismatches) or 'none'}"
    )
    return ok, detail


def _csv_value_matches(actual: str | None, expected: Any) -> bool:
    if actual is None:
        return False
    try:
        if isinstance(expected, bool):
            return _truthy_csv_value(actual) is expected
        if isinstance(expected, int):
            return int(float(actual)) == expected
        if isinstance(expected, float):
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False
    if isinstance(expected, Path):
        return Path(actual) == expected
    return actual == str(expected)


def _holdout_manifest_verifies(path_value: str) -> bool:
    try:
        return bool(path_value) and verify_holdout_manifest_file(Path(path_value))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _baseline_provenance_status(audit_path: Path) -> tuple[bool, str]:
    result_path: Path | None = None
    try:
        with audit_path.open(newline="") as handle:
            row = next(csv.DictReader(handle), None)
        if row is not None and row.get("artifact_path"):
            result_path = Path(str(row["artifact_path"]))
    except OSError:
        pass
    if result_path is None:
        suffix = "_audit.csv"
        result_path = Path(f"{str(audit_path)[:-len(suffix)]}.csv") if str(audit_path).endswith(suffix) else audit_path
    report = verify_recorded_expected_edge_provenance(
        provenance_path_for(result_path),
        result_path=result_path,
        audit_path=audit_path,
        require_source_files=True,
    )
    detail = f"baseline_result={result_path} baseline_provenance_errors={';'.join(report.errors) or 'none'}"
    return report.passed, detail


def _artifact_path_exists(value: str) -> bool:
    path = Path(value)
    return bool(value) and path.is_file() and path.stat().st_size > 0


def _artifact_hash_matches(path_value: str, expected_sha256: str) -> bool:
    if not expected_sha256 or not _artifact_path_exists(path_value):
        return False
    return sha256_file(Path(path_value)) == expected_sha256


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
    learned = _truthy_csv_value(row.get("learned_pretraining"))
    model_kind = row.get("model_kind", "legacy_unverified")
    learned_contract = (
        learned and model_kind == "masked_l2_encoder_v1"
        and _artifact_hash_matches(row.get("l2_path", ""), row.get("l2_sha256", ""))
        and _artifact_hash_matches(row.get("checkpoint_path", ""), row.get("checkpoint_sha256", ""))
        and _artifact_hash_matches(row.get("training_manifest_path", ""), row.get("training_manifest_sha256", ""))
    )
    evidence = (
        f"artifact_present=1 pipeline_completed={int(pipeline_completed)} "
        f"artifact_l2_match={int(l2_match)} learned_pretraining={int(learned)} "
        f"model_kind={model_kind} learned_contract={int(learned_contract)} artifact={artifact_path}"
    )
    return pipeline_completed and l2_match and learned_contract, evidence


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


def _numeric_metric(metrics: object, key: str) -> float:
    if not isinstance(metrics, dict):
        return 0.0
    try:
        return float(metrics.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


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
    parser.add_argument(
        "--order-plan",
        action="append",
        dest="order_plans",
        help="Provider paper-order-plan artifact path. Repeatable; defaults to Bybit/OKX/Binance plans beside the shadow decisions.",
    )
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
        order_plans=args.order_plans,
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
