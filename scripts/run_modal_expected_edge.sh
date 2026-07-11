#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${MODE:-plan}"
STUDY_PROFILE="${STUDY_PROFILE:-cloud_full}"
RUN_TESTS="${RUN_TESTS:-false}"
CONFIRM_HEAVY="${CONFIRM_HEAVY:-true}"
HOLDOUT_MANIFEST_PATH="${HOLDOUT_MANIFEST_PATH:-}"
if [[ -z "${MODAL_BIN:-}" && -x ".venv/bin/modal" ]]; then
  MODAL_BIN=".venv/bin/modal"
else
  MODAL_BIN="${MODAL_BIN:-modal}"
fi

case "$MODE" in
  plan|run|verify|sequence)
    ;;
  *)
    echo "unknown MODE=$MODE; expected plan, run, verify, or sequence" >&2
    exit 2
    ;;
esac

if [[ "$MODE" == "sequence" && -z "$HOLDOUT_MANIFEST_PATH" ]]; then
  echo "HOLDOUT_MANIFEST_PATH is required for MODE=sequence" >&2
  exit 2
fi

if [[ "$MODAL_BIN" != */* ]] && ! command -v "$MODAL_BIN" >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Modal CLI is not installed.
Install it with:
  python3 -m pip install ".[cloud]"
Then authenticate with:
  modal setup
EOF
  exit 2
fi
if [[ "$MODAL_BIN" == */* && ! -x "$MODAL_BIN" ]]; then
  cat >&2 <<EOF
Modal CLI is not executable: $MODAL_BIN
Install it with:
  python3 -m pip install ".[cloud]"
Then authenticate with:
  modal setup
EOF
  exit 2
fi

command=("$MODAL_BIN" run scripts/modal_expected_edge_job.py \
  --mode "$MODE" \
  --study-profile "$STUDY_PROFILE" \
  --run-tests "$RUN_TESTS" \
  --confirm-heavy "$CONFIRM_HEAVY")
if [[ -n "$HOLDOUT_MANIFEST_PATH" ]]; then
  command+=(--holdout-manifest-path "$HOLDOUT_MANIFEST_PATH")
fi
"${command[@]}"
