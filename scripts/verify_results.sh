#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONPATH

RESULT_DIR="${1:-results/current}"
if [[ -d "$RESULT_DIR" || "$#" -gt 0 || -d .git ]]; then
  python3 -m lob_forge.result_verifier "$RESULT_DIR"
else
  printf 'result_dir=%s present=0 skipped=1 source_package=1\n' "$RESULT_DIR"
fi
python3 scripts/verify_reduced_e2e_artifacts.py --project-root "$ROOT_DIR"
