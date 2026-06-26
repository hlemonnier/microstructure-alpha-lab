import csv
import json
from pathlib import Path

from lob_forge.evidence_gates import evaluate_remaining_evidence_gates, format_evidence_gate_report
from lob_forge.study_plan import build_expected_edge_run_plan, write_expected_edge_run_plan


def test_evidence_gates_report_missing_remaining_artifacts(tmp_path: Path) -> None:
    capped_plan = _write_plan(tmp_path / "capped" / "run_plan.json", profile="local16_60day")
    full_plan = _write_plan(tmp_path / "full" / "run_plan.json", profile="cloud_full")
    audit = tmp_path / "audit.csv"
    kelly = tmp_path / "kelly.csv"
    l2 = tmp_path / "missing_l2.csv"
    _write_audit(audit, fold_count=4, acceptance_passed=0, rejection_reasons="fold count too low")
    _write_kelly(kelly, [1.0, -10.0, 2.0, 0.0])

    report = evaluate_remaining_evidence_gates(
        capped_plan=capped_plan,
        capped_result_dir=capped_plan.parent,
        full_plan=full_plan,
        full_result_dir=full_plan.parent,
        simulated_fills=tmp_path / "simulated.csv",
        shadow_decisions=tmp_path / "shadow.csv",
        kelly_artifact=kelly,
        kelly_min_observations=4,
        kelly_window_size=2,
        baseline_audit=audit,
        l2_path=l2,
        min_fold_count=20,
        min_l2_rows=10,
    )
    text = format_evidence_gate_report(report)
    json_text = format_evidence_gate_report(report, output_format="json")

    assert not report.passed
    assert [gate.gate_id for gate in report.gates] == [
        "capped_60day_btc_eth",
        "full_60_90day_cloud",
        "immutable_final_holdout",
        "real_shadow_fill_validation",
        "kelly_variance_stability",
        "sequence_transformer_tcn_experiments",
        "self_supervised_l2_pretraining",
    ]
    assert "passed=0" in text
    assert '"passed": false' in json_text
    assert any(gate.status in {"missing", "not_ready"} for gate in report.gates)
    shadow_gate = next(gate for gate in report.gates if gate.gate_id == "real_shadow_fill_validation")
    assert "paper-order-plan" in shadow_gate.next_action
    assert "submit-paper-orders" in shadow_gate.next_action
    assert "fetch order/fill history" in shadow_gate.next_action
    final_gate = next(gate for gate in report.gates if gate.gate_id == "immutable_final_holdout")
    assert final_gate.status == "not_ready"
    assert "result_exists=0" in final_gate.evidence
    assert "final_holdout_edge_result.json" in final_gate.evidence
    assert "final_holdout_sequence_result.json" in final_gate.evidence
    assert "final-holdout-rule" in final_gate.next_action
    assert "final-holdout-edge" in final_gate.next_action
    assert "final-holdout-sequence" in final_gate.next_action


def test_evidence_gate_csv_format_quotes_commas(tmp_path: Path) -> None:
    capped_plan = _write_plan(tmp_path / "capped" / "run_plan.json", profile="local16_60day")
    full_plan = _write_plan(tmp_path / "full" / "run_plan.json", profile="cloud_full")
    audit = tmp_path / "audit.csv"
    kelly = tmp_path / "kelly.csv"
    _write_audit(audit, fold_count=4, acceptance_passed=0, rejection_reasons="fold count, too low")
    _write_kelly(kelly, [1.0, -10.0, 2.0, 0.0])

    report = evaluate_remaining_evidence_gates(
        capped_plan=capped_plan,
        capped_result_dir=capped_plan.parent,
        full_plan=full_plan,
        full_result_dir=full_plan.parent,
        simulated_fills=tmp_path / "simulated.csv",
        shadow_decisions=tmp_path / "shadow.csv",
        kelly_artifact=kelly,
        kelly_min_observations=4,
        kelly_window_size=2,
        baseline_audit=audit,
        l2_path=tmp_path / "missing_l2.csv",
    )
    csv_text = format_evidence_gate_report(report, output_format="csv")

    assert csv_text.splitlines()[0] == "gate_id,status,passed,evidence,next_action,todo_text"
    assert len(list(csv.DictReader(csv_text.splitlines()))) == 7


def test_shadow_evidence_gate_reports_order_plan_readiness(tmp_path: Path) -> None:
    capped_plan = _write_plan(tmp_path / "capped" / "run_plan.json", profile="local16_60day")
    full_plan = _write_plan(tmp_path / "full" / "run_plan.json", profile="cloud_full")
    audit = tmp_path / "audit.csv"
    kelly = tmp_path / "kelly.csv"
    shadow_dir = tmp_path / "results" / "shadow_validation"
    simulated = shadow_dir / "simulated_fills.csv"
    shadow = shadow_dir / "shadow_decisions.csv"
    bybit_plan = shadow_dir / "bybit_order_plan.jsonl"
    okx_plan = shadow_dir / "okx_order_plan.jsonl"
    _write_audit(audit, fold_count=4, acceptance_passed=0, rejection_reasons="fold count too low")
    _write_kelly(kelly, [1.0, -10.0, 2.0, 0.0])
    _write_simulated_fills(
        simulated, [{"decision_id": "d1", "simulated_fill_price": "100.0", "simulated_fill_size": "1.0"}]
    )
    _write_shadow_decisions(
        shadow,
        [
            {
                "decision_id": "d1",
                "timestamp_ms": "1700000000000",
                "venue": "bybit",
                "symbol": "BTCUSDT",
                "model_name": "ridge_expected_edge",
                "predicted_side": "1",
                "predicted_edge_bps": "0.8",
                "order_type": "paper_limit",
                "intended_price": "100.0",
                "intended_size": "1.0",
                "observed_fill_price": "",
                "observed_fill_size": "",
                "realized_pnl": "",
                "notes": "",
            }
        ],
    )
    bybit_plan.write_text('{"decision_id":"d1"}\n{"decision_id":"d2"}\n')
    okx_plan.write_text('{"decision_id":"d1"}\n')

    report = evaluate_remaining_evidence_gates(
        capped_plan=capped_plan,
        capped_result_dir=capped_plan.parent,
        full_plan=full_plan,
        full_result_dir=full_plan.parent,
        simulated_fills=simulated,
        shadow_decisions=shadow,
        kelly_artifact=kelly,
        kelly_min_observations=4,
        kelly_window_size=2,
        baseline_audit=audit,
        l2_path=tmp_path / "missing_l2.csv",
    )

    gate = next(gate for gate in report.gates if gate.gate_id == "real_shadow_fill_validation")
    assert gate.status == "not_ready"
    assert "order_plan_files=2/3" in gate.evidence
    assert "nonempty_order_plans=2/3" in gate.evidence
    assert "order_plan_rows=3" in gate.evidence
    assert "missing_order_plans=binance_usdm_order_plan.jsonl" in gate.evidence
    assert gate.next_action.startswith("generate paper-order-plan")
    assert "submit-paper-orders" in gate.next_action


def test_immutable_final_holdout_gate_passes_verified_payload(tmp_path: Path) -> None:
    capped_plan = _write_plan(tmp_path / "capped" / "run_plan.json", profile="local16_60day")
    full_plan = _write_plan(tmp_path / "full" / "run_plan.json", profile="cloud_full")
    audit = tmp_path / "audit.csv"
    kelly = tmp_path / "kelly.csv"
    final_holdout = tmp_path / "final_holdout.json"
    _write_audit(audit, fold_count=4, acceptance_passed=0, rejection_reasons="fold count too low")
    _write_kelly(kelly, [1.0, -10.0, 2.0, 0.0])
    _write_final_holdout_result(final_holdout)

    report = evaluate_remaining_evidence_gates(
        capped_plan=capped_plan,
        capped_result_dir=capped_plan.parent,
        full_plan=full_plan,
        full_result_dir=full_plan.parent,
        final_holdout_result=final_holdout,
        simulated_fills=tmp_path / "simulated.csv",
        shadow_decisions=tmp_path / "shadow.csv",
        kelly_artifact=kelly,
        kelly_min_observations=4,
        kelly_window_size=2,
        baseline_audit=audit,
        l2_path=tmp_path / "missing_l2.csv",
    )

    gate = next(gate for gate in report.gates if gate.gate_id == "immutable_final_holdout")
    assert gate.passed
    assert "final_evaluation=1" in gate.evidence
    assert "stateful_simulator=1" in gate.evidence


def test_immutable_final_holdout_gate_accepts_default_edge_artifact(tmp_path: Path) -> None:
    capped_plan = _write_plan(tmp_path / "capped" / "run_plan.json", profile="local16_60day")
    full_plan = _write_plan(tmp_path / "full" / "run_plan.json", profile="cloud_full")
    audit = tmp_path / "audit.csv"
    kelly = tmp_path / "kelly.csv"
    default_threshold_path = tmp_path / "results" / "final_holdout" / "final_holdout_result.json"
    default_edge_path = tmp_path / "results" / "final_holdout" / "final_holdout_edge_result.json"
    _write_audit(audit, fold_count=4, acceptance_passed=0, rejection_reasons="fold count too low")
    _write_kelly(kelly, [1.0, -10.0, 2.0, 0.0])
    _write_final_holdout_result(default_edge_path)

    report = evaluate_remaining_evidence_gates(
        capped_plan=capped_plan,
        capped_result_dir=capped_plan.parent,
        full_plan=full_plan,
        full_result_dir=full_plan.parent,
        final_holdout_result=default_threshold_path,
        simulated_fills=tmp_path / "simulated.csv",
        shadow_decisions=tmp_path / "shadow.csv",
        kelly_artifact=kelly,
        kelly_min_observations=4,
        kelly_window_size=2,
        baseline_audit=audit,
        l2_path=tmp_path / "missing_l2.csv",
    )

    gate = next(gate for gate in report.gates if gate.gate_id == "immutable_final_holdout")
    assert gate.passed
    assert f"result_path={default_edge_path}" in gate.evidence
    assert "final_evaluation=1" in gate.evidence


def test_immutable_final_holdout_gate_accepts_default_sequence_artifact(tmp_path: Path) -> None:
    capped_plan = _write_plan(tmp_path / "capped" / "run_plan.json", profile="local16_60day")
    full_plan = _write_plan(tmp_path / "full" / "run_plan.json", profile="cloud_full")
    audit = tmp_path / "audit.csv"
    kelly = tmp_path / "kelly.csv"
    default_threshold_path = tmp_path / "results" / "final_holdout" / "final_holdout_result.json"
    default_sequence_path = tmp_path / "results" / "final_holdout" / "final_holdout_sequence_result.json"
    _write_audit(audit, fold_count=4, acceptance_passed=0, rejection_reasons="fold count too low")
    _write_kelly(kelly, [1.0, -10.0, 2.0, 0.0])
    _write_final_holdout_result(default_sequence_path)

    report = evaluate_remaining_evidence_gates(
        capped_plan=capped_plan,
        capped_result_dir=capped_plan.parent,
        full_plan=full_plan,
        full_result_dir=full_plan.parent,
        final_holdout_result=default_threshold_path,
        simulated_fills=tmp_path / "simulated.csv",
        shadow_decisions=tmp_path / "shadow.csv",
        kelly_artifact=kelly,
        kelly_min_observations=4,
        kelly_window_size=2,
        baseline_audit=audit,
        l2_path=tmp_path / "missing_l2.csv",
    )

    gate = next(gate for gate in report.gates if gate.gate_id == "immutable_final_holdout")
    assert gate.passed
    assert f"result_path={default_sequence_path}" in gate.evidence
    assert "final_evaluation=1" in gate.evidence


def test_evidence_gates_accept_bybit_l2_candidate_when_okx_missing(tmp_path: Path) -> None:
    capped_plan = _write_plan(tmp_path / "capped" / "run_plan.json", profile="local16_60day")
    full_plan = _write_plan(tmp_path / "full" / "run_plan.json", profile="cloud_full")
    audit = tmp_path / "audit.csv"
    kelly = tmp_path / "kelly.csv"
    okx_missing = tmp_path / "normalized_l2" / "okx" / "BTC-USDT-SWAP" / "2023-05-16.csv"
    bybit_l2 = tmp_path / "normalized_l2" / "bybit" / "BTCUSDT" / "2023-05-16.csv"
    _write_audit(audit, fold_count=4, acceptance_passed=0, rejection_reasons="fold count too low")
    _write_kelly(kelly, [1.0, -10.0, 2.0, 0.0])
    _write_l2(bybit_l2)

    report = evaluate_remaining_evidence_gates(
        capped_plan=capped_plan,
        capped_result_dir=capped_plan.parent,
        full_plan=full_plan,
        full_result_dir=full_plan.parent,
        simulated_fills=tmp_path / "simulated.csv",
        shadow_decisions=tmp_path / "shadow.csv",
        kelly_artifact=kelly,
        kelly_min_observations=4,
        kelly_window_size=2,
        baseline_audit=audit,
        l2_paths=[okx_missing, bybit_l2],
        min_l2_rows=4,
    )

    model_gate = next(gate for gate in report.gates if gate.gate_id == "sequence_transformer_tcn_experiments")
    pretraining_gate = next(gate for gate in report.gates if gate.gate_id == "self_supervised_l2_pretraining")
    assert f"l2_path={bybit_l2}" in model_gate.evidence
    assert f"l2_path={bybit_l2}" in pretraining_gate.evidence
    assert "missing l2_path" not in model_gate.evidence


def test_shadow_evidence_gate_uses_default_error_thresholds(tmp_path: Path) -> None:
    capped_plan = _write_plan(tmp_path / "capped" / "run_plan.json", profile="local16_60day")
    full_plan = _write_plan(tmp_path / "full" / "run_plan.json", profile="cloud_full")
    audit = tmp_path / "audit.csv"
    kelly = tmp_path / "kelly.csv"
    simulated = tmp_path / "simulated.csv"
    shadow = tmp_path / "shadow.csv"
    _write_audit(audit, fold_count=4, acceptance_passed=0, rejection_reasons="fold count too low")
    _write_kelly(kelly, [1.0, -10.0, 2.0, 0.0])
    _write_simulated_fills(
        simulated, [{"decision_id": "d1", "simulated_fill_price": "100.0", "simulated_fill_size": "1.0"}]
    )
    _write_shadow_decisions(
        shadow,
        [
            {
                "decision_id": "d1",
                "timestamp_ms": "1700000000000",
                "venue": "bybit",
                "symbol": "BTCUSDT",
                "model_name": "ridge_expected_edge",
                "predicted_side": "1",
                "predicted_edge_bps": "0.8",
                "order_type": "paper_limit",
                "intended_price": "100.0",
                "intended_size": "1.0",
                "observed_fill_price": "",
                "observed_fill_size": "0.0",
                "realized_pnl": "",
                "notes": "",
            }
        ],
    )

    report = evaluate_remaining_evidence_gates(
        capped_plan=capped_plan,
        capped_result_dir=capped_plan.parent,
        full_plan=full_plan,
        full_result_dir=full_plan.parent,
        simulated_fills=simulated,
        shadow_decisions=shadow,
        min_shadow_observations=1,
        kelly_artifact=kelly,
        kelly_min_observations=4,
        kelly_window_size=2,
        baseline_audit=audit,
        l2_path=tmp_path / "missing_l2.csv",
    )

    gate = next(gate for gate in report.gates if gate.gate_id == "real_shadow_fill_validation")
    assert not gate.passed
    assert gate.status == "failed"
    assert "fill_rate_error=1" in gate.evidence
    assert "max_fill_rate_error=0.05" in gate.evidence


def test_shadow_evidence_gate_validates_only_explicit_observations(tmp_path: Path) -> None:
    capped_plan = _write_plan(tmp_path / "capped" / "run_plan.json", profile="local16_60day")
    full_plan = _write_plan(tmp_path / "full" / "run_plan.json", profile="cloud_full")
    audit = tmp_path / "audit.csv"
    kelly = tmp_path / "kelly.csv"
    simulated = tmp_path / "simulated.csv"
    shadow = tmp_path / "shadow.csv"
    _write_audit(audit, fold_count=4, acceptance_passed=0, rejection_reasons="fold count too low")
    _write_kelly(kelly, [1.0, -10.0, 2.0, 0.0])
    _write_simulated_fills(
        simulated,
        [
            {"decision_id": "observed", "simulated_fill_price": "100.0", "simulated_fill_size": "1.0"},
            {"decision_id": "shadow_only", "simulated_fill_price": "", "simulated_fill_size": "0.0"},
        ],
    )
    _write_shadow_decisions(
        shadow,
        [
            {
                "decision_id": "observed",
                "timestamp_ms": "1700000000000",
                "venue": "bybit",
                "symbol": "BTCUSDT",
                "model_name": "ridge_expected_edge",
                "predicted_side": "1",
                "predicted_edge_bps": "0.8",
                "order_type": "paper_limit",
                "intended_price": "100.0",
                "intended_size": "1.0",
                "observed_fill_price": "100.0",
                "observed_fill_size": "1.0",
                "realized_pnl": "",
                "notes": "",
            },
            {
                "decision_id": "shadow_only",
                "timestamp_ms": "1700000000100",
                "venue": "bybit",
                "symbol": "BTCUSDT",
                "model_name": "ridge_expected_edge",
                "predicted_side": "1",
                "predicted_edge_bps": "0.2",
                "order_type": "paper_limit",
                "intended_price": "100.0",
                "intended_size": "1.0",
                "observed_fill_price": "",
                "observed_fill_size": "",
                "realized_pnl": "",
                "notes": "",
            },
        ],
    )

    report = evaluate_remaining_evidence_gates(
        capped_plan=capped_plan,
        capped_result_dir=capped_plan.parent,
        full_plan=full_plan,
        full_result_dir=full_plan.parent,
        simulated_fills=simulated,
        shadow_decisions=shadow,
        min_shadow_observations=1,
        kelly_artifact=kelly,
        kelly_min_observations=4,
        kelly_window_size=2,
        baseline_audit=audit,
        l2_path=tmp_path / "missing_l2.csv",
    )

    gate = next(gate for gate in report.gates if gate.gate_id == "real_shadow_fill_validation")
    assert gate.passed
    assert "observed_shadow_rows=1 matched=1" in gate.evidence


def test_evidence_gates_pass_pretraining_when_artifact_matches_l2(tmp_path: Path) -> None:
    capped_plan = _write_plan(tmp_path / "capped" / "run_plan.json", profile="local16_60day")
    full_plan = _write_plan(tmp_path / "full" / "run_plan.json", profile="cloud_full")
    audit = tmp_path / "audit.csv"
    kelly = tmp_path / "kelly.csv"
    bybit_l2 = tmp_path / "normalized_l2" / "bybit" / "BTCUSDT" / "2023-05-16.csv"
    pretraining = tmp_path / "self_supervised_pretraining.csv"
    _write_audit(audit, fold_count=4, acceptance_passed=0, rejection_reasons="fold count too low")
    _write_kelly(kelly, [1.0, -10.0, 2.0, 0.0])
    _write_l2(bybit_l2)
    _write_pretraining_artifact(pretraining, l2_path=bybit_l2)

    report = evaluate_remaining_evidence_gates(
        capped_plan=capped_plan,
        capped_result_dir=capped_plan.parent,
        full_plan=full_plan,
        full_result_dir=full_plan.parent,
        simulated_fills=tmp_path / "simulated.csv",
        shadow_decisions=tmp_path / "shadow.csv",
        kelly_artifact=kelly,
        kelly_min_observations=4,
        kelly_window_size=2,
        baseline_audit=audit,
        l2_paths=[bybit_l2],
        min_l2_rows=4,
        pretraining_artifact=pretraining,
    )

    pretraining_gate = next(gate for gate in report.gates if gate.gate_id == "self_supervised_l2_pretraining")
    assert pretraining_gate.passed
    assert "pipeline_completed=1" in pretraining_gate.evidence
    assert "artifact_l2_match=1" in pretraining_gate.evidence


def test_sequence_model_gate_requires_pipeline_completed_artifacts(tmp_path: Path) -> None:
    capped_plan = _write_plan(tmp_path / "capped" / "run_plan.json", profile="local16_60day")
    full_plan = _write_plan(tmp_path / "full" / "run_plan.json", profile="cloud_full")
    audit = tmp_path / "audit.csv"
    kelly = tmp_path / "kelly.csv"
    bybit_l2 = tmp_path / "normalized_l2" / "bybit" / "BTCUSDT" / "2023-05-16.csv"
    transformer = tmp_path / "sequence_transformer_results.csv"
    tcn = tmp_path / "sequence_tcn_results.csv"
    pretraining = tmp_path / "self_supervised_pretraining.csv"
    legacy_transformer = tmp_path / "legacy_sequence_transformer_results.csv"
    legacy_tcn = tmp_path / "legacy_sequence_tcn_results.csv"
    _write_audit(audit, fold_count=20, acceptance_passed=1, rejection_reasons="")
    _write_kelly(kelly, [1.0, -10.0, 2.0, 0.0])
    _write_l2(bybit_l2)
    _write_sequence_model_artifact(transformer, model_name="sequence_transformer", l2_path=bybit_l2)
    _write_sequence_model_artifact(tcn, model_name="sequence_tcn", l2_path=bybit_l2)

    report = evaluate_remaining_evidence_gates(
        capped_plan=capped_plan,
        capped_result_dir=capped_plan.parent,
        full_plan=full_plan,
        full_result_dir=full_plan.parent,
        simulated_fills=tmp_path / "simulated.csv",
        shadow_decisions=tmp_path / "shadow.csv",
        kelly_artifact=kelly,
        kelly_min_observations=4,
        kelly_window_size=2,
        baseline_audit=audit,
        l2_paths=[bybit_l2],
        min_l2_rows=4,
        transformer_artifact=transformer,
        tcn_artifact=tcn,
        pretraining_artifact=pretraining,
    )
    gate = next(gate for gate in report.gates if gate.gate_id == "sequence_transformer_tcn_experiments")
    assert gate.passed
    assert "pipeline_completed=1" in gate.evidence

    _write_sequence_model_artifact(
        legacy_transformer,
        model_name="sequence_transformer",
        l2_path=bybit_l2,
        legacy_passed_only=True,
    )
    _write_sequence_model_artifact(
        legacy_tcn,
        model_name="sequence_tcn",
        l2_path=bybit_l2,
        legacy_passed_only=True,
    )
    legacy_report = evaluate_remaining_evidence_gates(
        capped_plan=capped_plan,
        capped_result_dir=capped_plan.parent,
        full_plan=full_plan,
        full_result_dir=full_plan.parent,
        simulated_fills=tmp_path / "simulated.csv",
        shadow_decisions=tmp_path / "shadow.csv",
        kelly_artifact=kelly,
        kelly_min_observations=4,
        kelly_window_size=2,
        baseline_audit=audit,
        l2_paths=[bybit_l2],
        min_l2_rows=4,
        transformer_artifact=legacy_transformer,
        tcn_artifact=legacy_tcn,
        pretraining_artifact=pretraining,
    )
    legacy_gate = next(gate for gate in legacy_report.gates if gate.gate_id == "sequence_transformer_tcn_experiments")
    assert not legacy_gate.passed
    assert "pipeline_completed=0" in legacy_gate.evidence


def test_kelly_gate_requires_variance_and_audit_acceptance(tmp_path: Path) -> None:
    capped_plan = _write_plan(tmp_path / "capped" / "run_plan.json", profile="local16_60day")
    full_plan = _write_plan(tmp_path / "full" / "run_plan.json", profile="cloud_full")
    audit = tmp_path / "audit.csv"
    kelly = tmp_path / "strategy_fee_0p05_edge.csv"
    kelly_audit = tmp_path / "strategy_fee_0p05_edge_audit.csv"
    _write_audit(audit, fold_count=4, acceptance_passed=0, rejection_reasons="fold count too low")
    _write_audit(kelly_audit, fold_count=20, acceptance_passed=0, rejection_reasons="cost safety failed")
    _write_kelly(kelly, [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0])

    report = evaluate_remaining_evidence_gates(
        capped_plan=capped_plan,
        capped_result_dir=capped_plan.parent,
        full_plan=full_plan,
        full_result_dir=full_plan.parent,
        simulated_fills=tmp_path / "simulated.csv",
        shadow_decisions=tmp_path / "shadow.csv",
        kelly_artifacts=[kelly],
        kelly_min_observations=9,
        kelly_window_size=3,
        kelly_max_variance_cv=0.01,
        baseline_audit=audit,
        l2_path=tmp_path / "missing_l2.csv",
    )

    gate = next(gate for gate in report.gates if gate.gate_id == "kelly_variance_stability")
    assert not gate.passed
    assert "variance_passed=1" in gate.evidence
    assert "audit_passed=0" in gate.evidence


def test_kelly_gate_passes_when_variance_and_audit_acceptance_match(tmp_path: Path) -> None:
    capped_plan = _write_plan(tmp_path / "capped" / "run_plan.json", profile="local16_60day")
    full_plan = _write_plan(tmp_path / "full" / "run_plan.json", profile="cloud_full")
    audit = tmp_path / "audit.csv"
    kelly = tmp_path / "strategy_fee_0p05_edge.csv"
    kelly_audit = tmp_path / "strategy_fee_0p05_edge_audit.csv"
    _write_audit(audit, fold_count=4, acceptance_passed=0, rejection_reasons="fold count too low")
    _write_audit(kelly_audit, fold_count=20, acceptance_passed=1, rejection_reasons="")
    _write_kelly(kelly, [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0])

    report = evaluate_remaining_evidence_gates(
        capped_plan=capped_plan,
        capped_result_dir=capped_plan.parent,
        full_plan=full_plan,
        full_result_dir=full_plan.parent,
        simulated_fills=tmp_path / "simulated.csv",
        shadow_decisions=tmp_path / "shadow.csv",
        kelly_artifacts=[kelly],
        kelly_min_observations=9,
        kelly_window_size=3,
        kelly_max_variance_cv=0.01,
        baseline_audit=audit,
        l2_path=tmp_path / "missing_l2.csv",
    )

    gate = next(gate for gate in report.gates if gate.gate_id == "kelly_variance_stability")
    assert gate.passed
    assert "variance_passed=1" in gate.evidence
    assert "audit_passed=1" in gate.evidence
    assert "fee_eligible=1" in gate.evidence


def test_kelly_gate_rejects_zero_fee_even_with_variance_and_audit_acceptance(tmp_path: Path) -> None:
    capped_plan = _write_plan(tmp_path / "capped" / "run_plan.json", profile="local16_60day")
    full_plan = _write_plan(tmp_path / "full" / "run_plan.json", profile="cloud_full")
    audit = tmp_path / "audit.csv"
    kelly = tmp_path / "strategy_fee_0_edge.csv"
    kelly_audit = tmp_path / "strategy_fee_0_edge_audit.csv"
    _write_audit(audit, fold_count=4, acceptance_passed=0, rejection_reasons="fold count too low")
    _write_audit(kelly_audit, fold_count=20, acceptance_passed=1, rejection_reasons="")
    _write_kelly(kelly, [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0])

    report = evaluate_remaining_evidence_gates(
        capped_plan=capped_plan,
        capped_result_dir=capped_plan.parent,
        full_plan=full_plan,
        full_result_dir=full_plan.parent,
        simulated_fills=tmp_path / "simulated.csv",
        shadow_decisions=tmp_path / "shadow.csv",
        kelly_artifacts=[kelly],
        kelly_min_observations=9,
        kelly_window_size=3,
        kelly_max_variance_cv=0.01,
        baseline_audit=audit,
        l2_path=tmp_path / "missing_l2.csv",
    )

    gate = next(gate for gate in report.gates if gate.gate_id == "kelly_variance_stability")
    assert not gate.passed
    assert "variance_passed=1" in gate.evidence
    assert "audit_passed=1" in gate.evidence
    assert "fee_eligible=0" in gate.evidence


def _write_plan(path: Path, *, profile: str) -> Path:
    plan = build_expected_edge_run_plan(
        profile=profile,
        start_date="2023-05-16",
        end_date="2023-05-16",
        symbols=["BTCUSDT"],
        horizons_ms=[5000],
        fees_bps=[0.0],
        latency_ms=1000,
        max_quote_buckets=10 if profile != "cloud_full" else None,
        train_size=1,
        validation_size=1,
        test_size=1,
        step_size=1,
        edge_streaming=True,
        min_ram_gb=4,
        max_csv_load_memory_gb=4,
        out_dir=str(path.parent),
        processed_root=str(path.parent / "processed"),
        raw_root=str(path.parent / "raw"),
        physical_ram_gb_value=128 if profile == "cloud_full" else 16,
        with_book_depth=profile == "cloud_full",
    )
    return write_expected_edge_run_plan(plan, path)


def _write_audit(path: Path, *, fold_count: int, acceptance_passed: int, rejection_reasons: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold_count", "acceptance_passed", "rejection_reasons"])
        writer.writeheader()
        writer.writerow(
            {
                "fold_count": fold_count,
                "acceptance_passed": acceptance_passed,
                "rejection_reasons": rejection_reasons,
            }
        )


def _write_kelly(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold", "test_net_pnl"])
        writer.writeheader()
        for index, value in enumerate(values, start=1):
            writer.writerow({"fold": index, "test_net_pnl": value})


def _write_final_holdout_result(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate_sha256 = "a" * 64
    path.write_text(
        json.dumps(
            {
                "candidate_sha256": candidate_sha256,
                "final_evaluation": True,
                "manifest": {"candidate_sha256": candidate_sha256},
                "metrics": {
                    "candidate_sha256": candidate_sha256,
                    "rows": 10,
                    "stateful_simulator": True,
                    "net_pnl": -1.25,
                },
            },
            sort_keys=True,
        )
        + "\n"
    )


def _write_simulated_fills(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["decision_id", "simulated_fill_price", "simulated_fill_size"])
        writer.writeheader()
        writer.writerows(rows)


def _write_shadow_decisions(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "decision_id",
                "timestamp_ms",
                "venue",
                "symbol",
                "model_name",
                "predicted_side",
                "predicted_edge_bps",
                "order_type",
                "intended_price",
                "intended_size",
                "observed_fill_price",
                "observed_fill_size",
                "realized_pnl",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_l2(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"event_type": "snapshot", "side": "bid", "price": "100.0", "size": "1.0", "sequence": "1", "venue": "bybit"},
        {"event_type": "snapshot", "side": "ask", "price": "101.0", "size": "1.0", "sequence": "1", "venue": "bybit"},
        {"event_type": "delta", "side": "bid", "price": "100.0", "size": "0.8", "sequence": "2", "venue": "bybit"},
        {"event_type": "delta", "side": "ask", "price": "101.0", "size": "0.6", "sequence": "2", "venue": "bybit"},
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_pretraining_artifact(path: Path, *, l2_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "l2_path": str(l2_path),
            "depth": "1",
            "window": "2",
            "rows_checked": "4",
            "snapshots": "2",
            "sequence_count": "1",
            "feature_count": "4",
            "mask_probability": "0.15",
            "masked_values": "2",
            "mean_reconstruction_mse": "0.1",
            "zero_reconstruction_mse": "1.0",
            "mean_abs_error": "0.2",
            "pipeline_completed": "1",
        }
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_sequence_model_artifact(
    path: Path,
    *,
    model_name: str,
    l2_path: Path,
    legacy_passed_only: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "model_name": model_name,
        "l2_path": str(l2_path),
        "readiness_passed": "1",
        "dependency_available": "1",
        "test_macro_f1": "0.0",
    }
    if legacy_passed_only:
        row["passed"] = "1"
    else:
        row["pipeline_completed"] = "1"
        row["acceptance_passed"] = "0"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
