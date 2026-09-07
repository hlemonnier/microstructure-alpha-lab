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
try:
    from _pytest.outcomes import Skipped as OptionalDependencySkip
except ImportError:
    class OptionalDependencySkip(BaseException):
        pass

root = Path("tests")
failures = []
skipped = []
skipped_modules = []
count = 0

for path in sorted(root.glob("test_*.py")):
    if not path.stem.isidentifier():
        continue  # Preserve local backup copies without executing them as tests.
    module_name = f"_direct_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    try:
        spec.loader.exec_module(module)
    except OptionalDependencySkip as exc:
        skipped_modules.append((str(path), str(exc)))
        continue

    for function_name, function in sorted(vars(module).items()):
        if not function_name.startswith("test_") or not callable(function):
            continue

        cases = [{}]
        for mark in getattr(function, "pytestmark", []):
            if mark.name != "parametrize":
                continue
            names = mark.args[0]
            names = [name.strip() for name in names.split(",")] if isinstance(names, str) else list(names)
            expanded = []
            for value in mark.args[1]:
                values = (value,) if len(names) == 1 else tuple(value)
                for case in cases:
                    expanded.append({**case, **dict(zip(names, values))})
            cases = expanded
        parameters = inspect.signature(function).parameters
        for case in cases:
            count += 1
            case_name = function_name + (f"{case!r}" if case else "")
            try:
                if "tmp_path" in parameters:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        function(tmp_path=Path(temp_dir), **case)
                else:
                    function(**case)
            except OptionalDependencySkip as exc:
                skipped.append((str(path), case_name, str(exc)))
            except Exception as exc:
                failures.append((str(path), case_name, repr(exc)))

if failures:
    for path, function_name, exc in failures:
        print(f"FAIL {path}::{function_name}: {exc}")
    raise SystemExit(1)

for path, function_name, reason in skipped:
    print(f"SKIP {path}::{function_name}: {reason}")
for path, reason in skipped_modules:
    print(f"SKIP MODULE {path}: {reason}")
print(f"passed {count - len(skipped)} direct test cases; skipped {len(skipped)} optional-dependency tests; skipped {len(skipped_modules)} optional test modules")
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
