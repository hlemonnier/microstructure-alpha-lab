import csv
import zipfile
from pathlib import Path

from lob_forge.external_gate_readiness import evaluate_external_gate_readiness, format_external_gate_readiness
from lob_forge.study_plan import build_expected_edge_run_plan, write_expected_edge_run_plan


def test_external_readiness_reports_missing_modal_and_external_evidence(tmp_path: Path) -> None:
    package = tmp_path / "dist" / "microstructure-alpha-lab-cloud-handoff-test.zip"
    _write_package(package, {"lob-forge/README.md": "# ok\n"})
    full_plan = _write_plan(tmp_path / "results" / "expected_edge_60day_20230516_20230714" / "run_plan.json")
    shadow = tmp_path / "results" / "shadow_validation" / "shadow_decisions.csv"
    simulated = tmp_path / "results" / "shadow_validation" / "simulated_fills.csv"
    _write_shadow(shadow, observed_size="")
    _write_simulated(simulated)

    report = evaluate_external_gate_readiness(
        project_root=tmp_path,
        modal_binary=str(tmp_path / "missing-modal"),
        cloud_package=package,
        full_plan=full_plan,
        full_result_dir=full_plan.parent,
        shadow_decisions=shadow,
        simulated_fills=simulated,
        observed_template=tmp_path / "results" / "shadow_validation" / "observed_fills_template.csv",
    )
    text = format_external_gate_readiness(report)

    checks = {check.check_id: check for check in report.checks}
    assert not report.passed
    assert checks["modal_cli"].status == "missing"
    assert checks["modal_auth"].status == "missing"
    assert checks["cloud_handoff_package"].passed
    assert checks["full_cloud_study"].status == "not_ready"
    assert checks["paper_live_fill_validation"].status == "not_ready"
    assert "template_exists=0" in checks["paper_live_fill_validation"].evidence
    assert "order_plan_files=0/3" in checks["paper_live_fill_validation"].evidence
    assert (
        "missing_order_plans=bybit_order_plan.jsonl,okx_order_plan.jsonl,binance_usdm_order_plan.jsonl"
        in checks["paper_live_fill_validation"].evidence
    )
    assert "paper-order-plan" in checks["paper_live_fill_validation"].next_action
    assert "fetch-observed-fills" in checks["paper_live_fill_validation"].next_action
    assert "check=modal_cli" in text


def test_external_readiness_counts_shadow_order_plan_files(tmp_path: Path) -> None:
    package = tmp_path / "dist" / "microstructure-alpha-lab-cloud-handoff-test.zip"
    _write_package(package, {"lob-forge/README.md": "# ok\n"})
    full_plan = _write_plan(tmp_path / "results" / "expected_edge_60day_20230516_20230714" / "run_plan.json")
    shadow_dir = tmp_path / "results" / "shadow_validation"
    shadow = shadow_dir / "shadow_decisions.csv"
    simulated = shadow_dir / "simulated_fills.csv"
    template = shadow_dir / "observed_fills_template.csv"
    bybit_plan = shadow_dir / "bybit_order_plan.jsonl"
    okx_plan = shadow_dir / "okx_order_plan.jsonl"
    _write_shadow(shadow, observed_size="")
    _write_simulated(simulated)
    template.write_text("decision_id,client_order_id\n")
    bybit_plan.write_text('{"decision_id":"d1"}\n{"decision_id":"d2"}\n')
    okx_plan.write_text('{"decision_id":"d1"}\n')

    report = evaluate_external_gate_readiness(
        project_root=tmp_path,
        modal_binary=str(tmp_path / "missing-modal"),
        cloud_package=package,
        full_plan=full_plan,
        full_result_dir=full_plan.parent,
        shadow_decisions=shadow,
        simulated_fills=simulated,
        observed_template=template,
    )

    check = {item.check_id: item for item in report.checks}["paper_live_fill_validation"]
    assert check.status == "not_ready"
    assert "template_exists=1" in check.evidence
    assert "order_plan_files=2/3" in check.evidence
    assert "nonempty_order_plans=2/3" in check.evidence
    assert "order_plan_rows=3" in check.evidence
    assert "missing_order_plans=binance_usdm_order_plan.jsonl" in check.evidence


def test_external_readiness_rejects_packages_with_excluded_nested_entries(tmp_path: Path) -> None:
    fake_modal = tmp_path / "modal"
    fake_modal.write_text('#!/bin/sh\nprintf \'{"token_id": null, "token_secret": null}\\n\'\n')
    fake_modal.chmod(fake_modal.stat().st_mode | 0o111)
    package = tmp_path / "dist" / "microstructure-alpha-lab-cloud-handoff-test.zip"
    _write_package(package, {"lob-forge/data/raw.csv": "bad\n"})

    report = evaluate_external_gate_readiness(
        project_root=tmp_path,
        modal_binary=str(fake_modal),
        cloud_package=package,
        full_plan=tmp_path / "missing_plan.json",
        shadow_decisions=tmp_path / "missing_shadow.csv",
        simulated_fills=tmp_path / "missing_simulated.csv",
    )

    checks = {check.check_id: check for check in report.checks}
    assert checks["modal_cli"].passed
    assert checks["modal_auth"].status == "not_ready"
    assert checks["cloud_handoff_package"].status == "failed"
    assert "first_excluded=lob-forge/data/raw.csv" in checks["cloud_handoff_package"].evidence


def test_external_readiness_passes_modal_auth_when_token_fields_are_present(tmp_path: Path) -> None:
    fake_modal = tmp_path / "modal"
    fake_modal.write_text(
        '#!/bin/sh\nprintf \'{"token_id": "ak-test-token", "token_secret": "raw-modal-secret-value"}\\n\'\n'
    )
    fake_modal.chmod(fake_modal.stat().st_mode | 0o111)
    package = tmp_path / "dist" / "microstructure-alpha-lab-cloud-handoff-test.zip"
    _write_package(package, {"lob-forge/README.md": "# ok\n"})

    report = evaluate_external_gate_readiness(
        project_root=tmp_path,
        modal_binary=str(fake_modal),
        cloud_package=package,
        full_plan=tmp_path / "missing_plan.json",
        shadow_decisions=tmp_path / "missing_shadow.csv",
        simulated_fills=tmp_path / "missing_simulated.csv",
    )

    checks = {check.check_id: check for check in report.checks}
    assert checks["modal_auth"].passed
    assert "token_secret_present=1" in checks["modal_auth"].evidence
    assert "raw-modal-secret-value" not in checks["modal_auth"].evidence


def _write_package(path: Path, files: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def _write_plan(path: Path) -> Path:
    plan = build_expected_edge_run_plan(
        profile="cloud_full",
        start_date="2023-05-16",
        end_date="2023-05-16",
        symbols=["BTCUSDT"],
        horizons_ms=[5000],
        fees_bps=[0.0],
        latency_ms=1000,
        max_quote_buckets=None,
        train_size=1,
        validation_size=1,
        test_size=1,
        step_size=1,
        edge_streaming=True,
        min_ram_gb=64,
        max_csv_load_memory_gb=48,
        max_feature_build_memory_gb=48,
        out_dir=str(path.parent),
        processed_root=str(path.parent / "processed"),
        raw_root=str(path.parent / "raw"),
        physical_ram_gb_value=128,
        with_book_depth=True,
    )
    return write_expected_edge_run_plan(plan, path)


def _write_simulated(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["decision_id", "simulated_fill_price", "simulated_fill_size"])
        writer.writeheader()
        writer.writerow({"decision_id": "d1", "simulated_fill_price": "100.0", "simulated_fill_size": "1.0"})


def _write_shadow(path: Path, *, observed_size: str) -> None:
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
        writer.writerow(
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
                "observed_fill_size": observed_size,
                "realized_pnl": "",
                "notes": "",
            }
        )
