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
            assert "requires a Git checkout" in str(exc)
            assert module.SOURCE_GIT_COMMIT_ENV in str(exc)
        else:
            raise AssertionError("expected missing Git metadata to fail")
    finally:
        module.subprocess.check_output = original_check_output
        _restore_env(module.SOURCE_GIT_COMMIT_ENV, previous_env)


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
