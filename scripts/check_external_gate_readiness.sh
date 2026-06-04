#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONPATH
if [[ -z "${PYTHON_BIN:-}" && -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

OUTPUT_PATH="${OUTPUT_PATH:-results/current/external_gate_readiness.json}"
FORMAT="${FORMAT:-json}"
mkdir -p "$(dirname "$OUTPUT_PATH")"

"$PYTHON_BIN" -m lob_forge.external_gate_readiness \
  --format "$FORMAT" \
  --output "$OUTPUT_PATH"
