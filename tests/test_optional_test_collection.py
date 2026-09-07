import os
from pathlib import Path
import subprocess
import sys


def test_direct_runner_handles_optional_dependency_module_skips(tmp_path):
    runner = Path(__file__).resolve().parents[1] / "scripts/run_tests.sh"
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "scripts/run_tests.sh").write_text(runner.read_text())
    (tmp_path / "tests/test_optional.py").write_text(
        "import pytest\npytest.skip('optional dependency unavailable', allow_module_level=True)\n"
    )
    (tmp_path / "tests/test_regular.py").write_text("def test_regular():\n    assert 2 + 2 == 4\n")
    result = subprocess.run(
        ["bash", str(tmp_path / "scripts/run_tests.sh"), "direct"],
        cwd=tmp_path,
        env={**os.environ, "PYTHON_BIN": sys.executable, "PYTHONPATH": str(tmp_path / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SKIP MODULE tests/test_optional.py" in result.stdout
    assert "passed 1 direct test cases" in result.stdout
    assert "skipped 1 optional test modules" in result.stdout
