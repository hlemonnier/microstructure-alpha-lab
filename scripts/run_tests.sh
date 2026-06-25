#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-direct}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONPATH
if [[ -z "${PYTHON_BIN:-}" && -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

run_direct_tests() {
  "$PYTHON_BIN" -m compileall -q src tests

  "$PYTHON_BIN" - <<'PY'
import importlib.util
import inspect
import tempfile
from pathlib import Path

root = Path("tests")
failures = []
count = 0

for path in sorted(root.glob("test_*.py")):
    module_name = f"_direct_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    for function_name, function in sorted(vars(module).items()):
        if not function_name.startswith("test_") or not callable(function):
            continue

        count += 1
        parameters = inspect.signature(function).parameters

        try:
            if "tmp_path" in parameters:
                with tempfile.TemporaryDirectory() as temp_dir:
                    function(Path(temp_dir))
            else:
                function()
        except Exception as exc:
            failures.append((str(path), function_name, repr(exc)))

if failures:
    for path, function_name, exc in failures:
        print(f"FAIL {path}::{function_name}: {exc}")
    raise SystemExit(1)

print(f"passed {count} direct test functions")
PY
}

case "$MODE" in
  direct)
    run_direct_tests
    ;;
  pytest)
    "$PYTHON_BIN" -m compileall -q src tests
    "$PYTHON_BIN" -m pytest -q
    ;;
  all)
    run_direct_tests
    "$PYTHON_BIN" -m pytest -q
    ;;
  *)
    echo "Usage: bash scripts/run_tests.sh [direct|pytest|all]" >&2
    exit 2
    ;;
esac
