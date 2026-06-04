#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONPATH

RESULT_DIR="${1:-results/expected_edge_laptop_quick_20230516_20230522}"
PLAN_PATH="${2:-$RESULT_DIR/run_plan.json}"
STATUS_PATH="${STATUS_PATH:-$RESULT_DIR/study_status.json}"
MIN_AUDIT_FOLD_COUNT="${MIN_AUDIT_FOLD_COUNT:-1}"

python3 -m lob_forge.study_status \
  --plan "$PLAN_PATH" \
  --result-dir "$RESULT_DIR" \
  --output "$STATUS_PATH" \
  --min-audit-fold-count "$MIN_AUDIT_FOLD_COUNT"
