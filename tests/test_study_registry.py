from __future__ import annotations

from pathlib import Path

from lob_forge.study_plan import build_expected_edge_run_plan, write_expected_edge_run_plan
from lob_forge.study_registry import (
    build_expected_edge_candidate_registry,
    main as study_registry_main,
    read_expected_edge_candidate_registry,
    write_expected_edge_candidate_registry,
)


def test_expected_edge_candidate_registry_expands_run_plan_grid(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path / "run_plan.json", symbols=["BTCUSDT", "ETHUSDT"])

    attempts = build_expected_edge_candidate_registry(plan_path=plan_path, result_dir=tmp_path / "results")

    assert len(attempts) == 4
    assert {attempt.symbol for attempt in attempts} == {"BTCUSDT", "ETHUSDT"}
    assert {attempt.edge_threshold_bps for attempt in attempts} == {0.0, 0.1}
    assert {attempt.status for attempt in attempts} == {"planned"}
    assert all(attempt.config_sha256 for attempt in attempts)


def test_expected_edge_candidate_registry_marks_selected_artifact_rows(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path / "run_plan.json")
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    (result_dir / "BTCUSDT_5000ms_fee_0p1_edge.csv").write_text(
        "\n".join(
            [
                "fold,edge_threshold_bps,val_net_pnl,test_net_pnl",
                "1,0.1,2.5,-0.5",
                "2,0.1,1.5,0.25",
                "summary,,,",
            ]
        )
        + "\n"
    )
    (result_dir / "BTCUSDT_5000ms_fee_0p1_edge_audit.csv").write_text(
        "\n".join(
            [
                "acceptance_passed,rejection_reasons",
                "0,total OOS net PnL is not positive",
            ]
        )
        + "\n"
    )

    attempts = build_expected_edge_candidate_registry(plan_path=plan_path, result_dir=result_dir)
    selected = next(attempt for attempt in attempts if attempt.edge_threshold_bps == 0.1)
    unselected = next(attempt for attempt in attempts if attempt.edge_threshold_bps == 0.0)

    assert selected.status == "selected_in_artifact"
    assert selected.selected
    assert selected.selected_fold_count == 2
    assert selected.fold_count == 2
    assert selected.validation_net_pnl == 4.0
    assert selected.test_net_pnl == -0.25
    assert selected.audit_acceptance_passed is False
    assert selected.audit_rejection_reasons == "total OOS net PnL is not positive"
    assert selected.artifact_path.endswith("BTCUSDT_5000ms_fee_0p1_edge.csv")
    assert unselected.status == "evaluated_unselected"
    assert not unselected.selected


def test_expected_edge_candidate_registry_cli_writes_jsonl(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path / "run_plan.json")
    output = tmp_path / "candidate_registry.jsonl"

    exit_code = study_registry_main(
        ["--plan", str(plan_path), "--result-dir", str(tmp_path / "results"), "--output", str(output)]
    )
    attempts = read_expected_edge_candidate_registry(output)

    assert exit_code == 0
    assert len(attempts) == 2
    assert output.read_text().count("\n") == 2
    assert write_expected_edge_candidate_registry(attempts, tmp_path / "roundtrip.jsonl").exists()


def _write_plan(path: Path, *, symbols: list[str] | None = None) -> Path:
    plan = build_expected_edge_run_plan(
        profile="laptop_tiny",
        start_date="2023-05-16",
        end_date="2023-05-16",
        symbols=symbols or ["BTCUSDT"],
        horizons_ms=[5000],
        fees_bps=[0.1],
        latency_ms=1000,
        max_quote_buckets=1200,
        train_size=900,
        validation_size=450,
        test_size=450,
        step_size=450,
        edge_streaming=True,
        min_ram_gb=4,
        max_csv_load_memory_gb=4,
        out_dir=str(path.parent / "results"),
        processed_root=str(path.parent / "processed"),
        raw_root=str(path.parent / "raw"),
        physical_ram_gb_value=16,
        with_book_depth=False,
        max_feature_build_memory_gb=4,
        edge_thresholds_bps=[0.0, 0.1],
    )
    return write_expected_edge_run_plan(plan, path)
