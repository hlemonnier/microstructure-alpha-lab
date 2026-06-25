from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path


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
