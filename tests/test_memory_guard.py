import os
import builtins
from types import SimpleNamespace
from pathlib import Path

from lob_forge.memory_guard import (
    apply_process_memory_limit,
    assert_csv_load_budget,
    assert_feature_build_budget,
    process_memory_limit_gb_from_env,
    estimate_csv_load_memory_gb,
    estimate_feature_build_memory_gb,
)


def test_csv_memory_estimate_uses_file_size_and_multiplier(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    path.write_text("a,b\n1,2\n")

    estimate = estimate_csv_load_memory_gb(path, multiplier=10.0)

    assert estimate == path.stat().st_size * 10.0 / 1_000_000_000


def test_csv_memory_budget_rejects_oversized_input(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    path.write_text("a,b\n1,2\n")

    try:
        assert_csv_load_budget(path, max_memory_gb=0.000000001, multiplier=10.0)
    except ValueError as exc:
        assert "estimated CSV load memory exceeds budget" in str(exc)
    else:
        raise AssertionError("expected memory budget rejection")


def test_csv_memory_budget_can_be_disabled(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    path.write_text("a,b\n1,2\n")

    assert_csv_load_budget(path, max_memory_gb=0.0, multiplier=10.0)


def test_feature_build_memory_estimate_sums_input_files(tmp_path: Path) -> None:
    first = tmp_path / "bookTicker.zip"
    second = tmp_path / "aggTrades.zip"
    first.write_text("a" * 10)
    second.write_text("b" * 20)

    estimate = estimate_feature_build_memory_gb([first, second, None], multiplier=5.0)

    assert estimate == 30 * 5.0 / 1_000_000_000


def test_feature_build_memory_budget_rejects_oversized_inputs(tmp_path: Path) -> None:
    first = tmp_path / "bookTicker.zip"
    second = tmp_path / "aggTrades.zip"
    first.write_text("a" * 10)
    second.write_text("b" * 20)

    try:
        assert_feature_build_budget([first, second], max_memory_gb=0.000000001, multiplier=5.0)
    except ValueError as exc:
        assert "estimated feature-build memory exceeds budget" in str(exc)
        assert "WITH_BOOK_DEPTH" in str(exc)
    else:
        raise AssertionError("expected feature-build memory budget rejection")


def test_process_memory_limit_env_is_optional() -> None:
    previous = os.environ.pop("LOB_FORGE_MAX_PROCESS_MEMORY_GB", None)
    try:
        assert process_memory_limit_gb_from_env() is None
    finally:
        if previous is not None:
            os.environ["LOB_FORGE_MAX_PROCESS_MEMORY_GB"] = previous


def test_process_memory_limit_env_can_be_disabled() -> None:
    previous = os.environ.get("LOB_FORGE_MAX_PROCESS_MEMORY_GB")
    os.environ["LOB_FORGE_MAX_PROCESS_MEMORY_GB"] = "0"
    try:
        assert process_memory_limit_gb_from_env() is None
    finally:
        if previous is None:
            os.environ.pop("LOB_FORGE_MAX_PROCESS_MEMORY_GB", None)
        else:
            os.environ["LOB_FORGE_MAX_PROCESS_MEMORY_GB"] = previous


def test_process_memory_limit_env_rejects_invalid_value() -> None:
    previous = os.environ.get("LOB_FORGE_MAX_PROCESS_MEMORY_GB")
    os.environ["LOB_FORGE_MAX_PROCESS_MEMORY_GB"] = "sixteen"
    try:
        try:
            process_memory_limit_gb_from_env()
        except ValueError as exc:
            assert "LOB_FORGE_MAX_PROCESS_MEMORY_GB must be a number" in str(exc)
        else:
            raise AssertionError("expected invalid process memory limit rejection")
    finally:
        if previous is None:
            os.environ.pop("LOB_FORGE_MAX_PROCESS_MEMORY_GB", None)
        else:
            os.environ["LOB_FORGE_MAX_PROCESS_MEMORY_GB"] = previous


def test_process_memory_limit_returns_none_when_platform_rejects_limits() -> None:
    original_import = builtins.__import__
    fake_resource = SimpleNamespace(
        RLIM_INFINITY=999999999,
        RLIMIT_AS=1,
        RLIMIT_DATA=2,
        RLIMIT_RSS=3,
        getrlimit=lambda _kind: (999999999, 999999999),
        setrlimit=lambda _kind, _limits: (_ for _ in ()).throw(ValueError("unsupported")),
    )

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "resource":
            return fake_resource
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = fake_import
    try:
        assert apply_process_memory_limit(1.0) is None
    finally:
        builtins.__import__ = original_import
