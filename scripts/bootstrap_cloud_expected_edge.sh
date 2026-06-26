#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${MODE:-plan}"
STUDY_PROFILE="${STUDY_PROFILE:-cloud_full}"
CONFIRM_HEAVY="${CONFIRM_HEAVY:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
RUN_TESTS="${RUN_TESTS:-1}"
MIN_AUDIT_FOLD_COUNT="${MIN_AUDIT_FOLD_COUNT:-20}"

export STUDY_PROFILE CONFIRM_HEAVY MIN_AUDIT_FOLD_COUNT

case "$MODE" in
  plan|run|verify|sequence)
    ;;
  *)
    echo "unknown MODE=$MODE; expected plan, run, verify, or sequence" >&2
    exit 2
    ;;
esac

if [[ "$SKIP_INSTALL" != "1" ]]; then
  "$PYTHON_BIN" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python3 -m pip install --upgrade pip
  python3 -m pip install -r requirements-research.txt
  python3 -m pip install -e . --no-deps
elif [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if [[ "$RUN_TESTS" == "1" ]]; then
  bash scripts/run_tests.sh
fi

case "$MODE" in
  plan)
    bash scripts/plan_expected_edge_study.sh
    ;;
  run)
    bash scripts/plan_expected_edge_study.sh
    bash scripts/run_60day_expected_edge_study.sh
    bash scripts/verify_expected_edge_features.sh "results/expected_edge_60day_20230516_20230714"
    bash scripts/verify_expected_edge_study.sh "results/expected_edge_60day_20230516_20230714"
    ;;
  verify)
    RESULT_DIR="${RESULT_DIR:-results/expected_edge_60day_20230516_20230714}"
    bash scripts/verify_expected_edge_features.sh "$RESULT_DIR"
    bash scripts/verify_expected_edge_study.sh "$RESULT_DIR"
    ;;
  sequence)
    DRY_RUN="${DRY_RUN:-0}" bash scripts/run_l2_sequence_experiments.sh
    ;;
esac
