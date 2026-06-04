#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONPATH

RESULT_DIR="${1:-results/expected_edge_laptop_quick_20230516_20230522}"
PLAN_PATH="${2:-$RESULT_DIR/run_plan.json}"
FEATURE_STATUS_PATH="${FEATURE_STATUS_PATH:-$RESULT_DIR/feature_status.json}"

python3 -m lob_forge.study_features \
  --plan "$PLAN_PATH" \
  --output "$FEATURE_STATUS_PATH"
