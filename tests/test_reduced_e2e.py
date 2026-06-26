from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace


def test_reduced_e2e_git_commit_env_override_accepts_full_hash() -> None:
    module = _load_reduced_e2e_module()
    previous = os.environ.get(module.SOURCE_GIT_COMMIT_ENV)
    try:
        os.environ[module.SOURCE_GIT_COMMIT_ENV] = "a" * 40
        assert module._git_commit() == "a" * 40
    finally:
        _restore_env(module.SOURCE_GIT_COMMIT_ENV, previous)


def test_reduced_e2e_git_commit_env_override_rejects_non_hash() -> None:
    module = _load_reduced_e2e_module()
    previous = os.environ.get(module.SOURCE_GIT_COMMIT_ENV)
    try:
        os.environ[module.SOURCE_GIT_COMMIT_ENV] = "not-a-commit"
        try:
            module._git_commit()
        except RuntimeError as exc:
            assert module.SOURCE_GIT_COMMIT_ENV in str(exc)
        else:
            raise AssertionError("expected invalid source commit override to fail")
    finally:
        _restore_env(module.SOURCE_GIT_COMMIT_ENV, previous)


def test_reduced_e2e_git_commit_requires_git_or_explicit_hash() -> None:
    module = _load_reduced_e2e_module()
    previous_env = os.environ.get(module.SOURCE_GIT_COMMIT_ENV)
    original_check_output = module.subprocess.check_output

    def fail_check_output(*args: object, **kwargs: object) -> str:
        raise subprocess.CalledProcessError(128, args[0] if args else "git")

    try:
        os.environ.pop(module.SOURCE_GIT_COMMIT_ENV, None)
        module.subprocess.check_output = fail_check_output
        try:
            module._git_commit()
        except RuntimeError as exc:
            assert "requires Git metadata" in str(exc)
            assert module.SOURCE_GIT_COMMIT_ENV in str(exc)
        else:
            raise AssertionError("expected missing Git metadata to fail")
    finally:
        module.subprocess.check_output = original_check_output
        _restore_env(module.SOURCE_GIT_COMMIT_ENV, previous_env)


def test_reduced_e2e_git_commit_uses_git_archive_substitution_without_git(tmp_path: Path) -> None:
    module = _load_reduced_e2e_module()
    previous_env = os.environ.get(module.SOURCE_GIT_COMMIT_ENV)
    original_check_output = module.subprocess.check_output
    original_root = module.ROOT

    def fail_check_output(*args: object, **kwargs: object) -> str:
        raise subprocess.CalledProcessError(128, args[0] if args else "git")

    try:
        os.environ.pop(module.SOURCE_GIT_COMMIT_ENV, None)
        module.subprocess.check_output = fail_check_output
        module.ROOT = tmp_path
        (tmp_path / module.SOURCE_ARCHIVE_COMMIT_FILE).write_text("b" * 40 + "\n")
        assert module._git_commit() == "b" * 40
    finally:
        module.ROOT = original_root
        module.subprocess.check_output = original_check_output
        _restore_env(module.SOURCE_GIT_COMMIT_ENV, previous_env)


def test_reduced_e2e_git_commit_ignores_unexpanded_archive_placeholder(tmp_path: Path) -> None:
    module = _load_reduced_e2e_module()
    original_root = module.ROOT
    try:
        module.ROOT = tmp_path
        (tmp_path / module.SOURCE_ARCHIVE_COMMIT_FILE).write_text("$Format:%H$\n")
        assert module._source_archive_git_commit() is None
    finally:
        module.ROOT = original_root


def test_reduced_e2e_statistics_use_ledger_increments_and_fees() -> None:
    module = _load_reduced_e2e_module()
    positions = [
        module.PositionLedgerRow(
            timestamp_ms=1000,
            cash=1001.0,
            inventory=0.0,
            avg_entry_price=0.0,
            mark_price=100.0,
            equity=1001.0,
            realized_pnl=1.0,
            turnover=100.0,
            kill_switch_triggered=False,
        ),
        module.PositionLedgerRow(
            timestamp_ms=1500,
            cash=1001.0,
            inventory=0.0,
            avg_entry_price=0.0,
            mark_price=100.0,
            equity=1001.0,
            realized_pnl=1.0,
            turnover=100.0,
            kill_switch_triggered=False,
        ),
        module.PositionLedgerRow(
            timestamp_ms=2000,
            cash=1003.0,
            inventory=0.0,
            avg_entry_price=0.0,
            mark_price=100.0,
            equity=1003.0,
            realized_pnl=3.0,
            turnover=150.0,
            kill_switch_triggered=False,
        ),
    ]
    fills = [
        module.FillLedgerRow("o1", 1000, 1, 1.0, 100.0, 0.1, "taker", 1.0, False),
        module.FillLedgerRow("o2", 2000, -1, 0.5, 100.0, 0.2, "taker", 0.5, False),
    ]

    net_pnls, net_returns, days, gross_pnls, turnovers = module._ledger_statistics_inputs(
        positions,
        fills,
        timestamp_days={1000: "d1", 2000: "d2"},
        initial_cash=1000.0,
    )

    assert net_pnls == [1.0, 2.0]
    assert net_returns == [0.001, 0.002]
    assert days == ["d1", "d2"]
    assert gross_pnls == [1.1, 2.2]
    assert turnovers == [100.0, 50.0]


def test_reduced_e2e_signals_are_exported_from_walk_forward_test_slice_only() -> None:
    module = _load_reduced_e2e_module()
    rows = [
        {"event_time": str(index * 1000), "microprice_deviation": "1.0"}
        for index in range(1, module.TRAIN_SIZE + module.VALIDATION_SIZE + module.TEST_SIZE + 1)
    ]
    fold = SimpleNamespace(
        fold=1,
        result=SimpleNamespace(feature="microprice_deviation", threshold=0.5),
    )

    signals = module._signals_from_walk_forward(rows, [fold])

    assert [signal.decision_time_ms for signal in signals] == [7000, 8000]
    assert [signal.signal_id for signal in signals] == ["selected-rule-oos-1", "selected-rule-oos-2"]
    assert all(signal.target_side == 1 for signal in signals)


def test_reduced_e2e_sequence_smokes_use_l2_holdout_manifest(tmp_path: Path) -> None:
    module = _load_reduced_e2e_module()
    previous_root = module.ROOT
    original_runner = module.run_l2_torch_sequence_experiment
    calls: list[dict[str, object]] = []

    def fake_runner(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("model_name,pipeline_completed\nfixture,1\n")
        development_l2_path = Path(kwargs["development_l2_output_path"])
        development_l2_path.parent.mkdir(parents=True, exist_ok=True)
        development_l2_path.write_text("exchange_timestamp\n1000\n")
        return SimpleNamespace(
            passed=True,
            validation_macro_f1=0.25,
            test_macro_f1=0.5,
            test_stateful_net_pnl=-0.1,
            purge_gap=3,
            holdout_manifest_sha256="a" * 64,
            development_l2_path=str(development_l2_path),
            holdout_rows_excluded=1,
        )

    try:
        module.ROOT = tmp_path
        module.run_l2_torch_sequence_experiment = fake_runner
        baseline_audit_path = tmp_path / "baseline.csv"
        tcn_path = tmp_path / "sequence_tcn.csv"
        transformer_path = tmp_path / "sequence_transformer.csv"
        holdout_manifest_path = tmp_path / "l2_holdout.json"
        development_l2_path = tmp_path / "development_l2.csv"

        outputs = module._run_sequence_smokes(
            baseline_audit_path=baseline_audit_path,
            sequence_tcn_path=tcn_path,
            sequence_transformer_path=transformer_path,
            checkpoint_dir=tmp_path / "checkpoints",
            prediction_dir=tmp_path / "predictions",
            holdout_manifest_path=holdout_manifest_path,
            development_l2_path=development_l2_path,
        )
    finally:
        module.ROOT = previous_root
        module.run_l2_torch_sequence_experiment = original_runner

    assert len(calls) == 2
    assert all(call["holdout_manifest_path"] == holdout_manifest_path for call in calls)
    assert all(call["development_l2_output_path"] == development_l2_path for call in calls)
    assert outputs["sequence_tcn"]["holdout_manifest"] == "l2_holdout.json"
    assert outputs["sequence_tcn"]["development_l2"] == "development_l2.csv"
    assert outputs["sequence_tcn"]["holdout_manifest_sha256"] == "a" * 64
    assert outputs["sequence_transformer"]["holdout_rows_excluded"] == 1


def _load_reduced_e2e_module() -> object:
    path = Path("scripts/run_reduced_e2e.py")
    spec = importlib.util.spec_from_file_location("_test_run_reduced_e2e", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _restore_env(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous
