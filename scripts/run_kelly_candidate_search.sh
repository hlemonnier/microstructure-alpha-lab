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

SYMBOL="${SYMBOL:-BTCUSDT}"
HORIZON_MS="${HORIZON_MS:-5000}"
FEE_BPS="${FEE_BPS:-0.05}"
LATENCY_MS="${LATENCY_MS:-1000}"
START_DATE="${START_DATE:-2023-05-16}"
END_DATE="${END_DATE:-2023-07-14}"
SOURCE_PROCESSED_ROOT="${SOURCE_PROCESSED_ROOT:-data/processed/expected_edge_60day_20230516_20230714}"
OUT_DIR="${OUT_DIR:-results/kelly_candidate_search}"
DRY_RUN="${DRY_RUN:-1}"
MAX_LOAD_MEMORY_GB="${MAX_LOAD_MEMORY_GB:-10}"
LOB_FORGE_MAX_PROCESS_MEMORY_GB="${LOB_FORGE_MAX_PROCESS_MEMORY_GB:-8}"
TRAIN_SIZE="${TRAIN_SIZE:-43200}"
VALIDATION_SIZE="${VALIDATION_SIZE:-14400}"
TEST_SIZE="${TEST_SIZE:-14400}"
STEP_SIZE="${STEP_SIZE:-14400}"
MAX_FOLDS="${MAX_FOLDS:-20}"
EDGE_THRESHOLDS_BPS="${EDGE_THRESHOLDS_BPS:-0.06,0.065,0.07,0.075,0.08,0.09,0.1,0.125,0.15,0.175,0.2,0.25,0.3,0.4,0.5,0.75,1,1.5,2,3,5}"
MIN_AUDIT_FOLD_COUNT="${MIN_AUDIT_FOLD_COUNT:-20}"
MIN_POSITIVE_FOLD_RATE="${MIN_POSITIVE_FOLD_RATE:-0.70}"
MAX_FOLD_CONTRIBUTION="${MAX_FOLD_CONTRIBUTION:-0.40}"
KELLY_WINDOW_SIZE="${KELLY_WINDOW_SIZE:-5}"
MAX_VARIANCE_CV="${MAX_VARIANCE_CV:-0.5}"
SOURCE_BUCKET_MS="${SOURCE_BUCKET_MS:-1000}"
SOURCE_MAX_QUOTE_BUCKETS="${SOURCE_MAX_QUOTE_BUCKETS:-}"
SOURCE_WITH_BOOK_DEPTH="${SOURCE_WITH_BOOK_DEPTH:-1}"
SOURCE_FEATURE_THRESHOLD="${SOURCE_FEATURE_THRESHOLD:-half_spread}"
SOURCE_LARGE_TRADE_NOTIONAL="${SOURCE_LARGE_TRADE_NOTIONAL:-10000}"
SOURCE_EXECUTION_QUOTE_RESOLUTION="${SOURCE_EXECUTION_QUOTE_RESOLUTION:-raw}"
HOLDOUT_SPLIT_COLUMN="${HOLDOUT_SPLIT_COLUMN:-source_date}"
HOLDOUT_VALUES="${HOLDOUT_VALUES:-$END_DATE}"

export LOB_FORGE_MAX_PROCESS_MEMORY_GB
source scripts/holdout_manifest.sh
source scripts/source_provenance.sh

safe_fee="$(printf '%s' "$FEE_BPS" | tr '.' 'p')"
symbol_lower="$(printf '%s' "$SYMBOL" | tr '[:upper:]' '[:lower:]')"
combined="$SOURCE_PROCESSED_ROOT/${symbol_lower}_${HORIZON_MS}ms_latency_${LATENCY_MS}/${SYMBOL}-${START_DATE}_${END_DATE}-combined-features.csv"
result="$OUT_DIR/${SYMBOL}_${HORIZON_MS}ms_fee_${safe_fee}_balanced_edge.csv"
audit="$OUT_DIR/${SYMBOL}_${HORIZON_MS}ms_fee_${safe_fee}_balanced_edge_audit.csv"
plan="$OUT_DIR/run_plan.json"
provenance="${result}.provenance.json"
if [[ "$DRY_RUN" == "0" ]]; then
  holdout_manifest="$(holdout_manifest_for "$combined")"
else
  holdout_manifest="${HOLDOUT_MANIFEST_DIR:-results/holdout_manifests}/$(basename "$combined" .csv).holdout.json"
fi

mkdir -p "$OUT_DIR"

printf 'kelly_candidate symbol=%s horizon_ms=%s fee_bps=%s dry_run=%s\n' "$SYMBOL" "$HORIZON_MS" "$FEE_BPS" "$DRY_RUN"
printf 'combined=%s result=%s audit=%s\n' "$combined" "$result" "$audit"
printf 'edge_thresholds_bps=%s\n' "$EDGE_THRESHOLDS_BPS"

command=(
  "$PYTHON_BIN" -m lob_forge.cli edge-walk-forward "$combined"
  --holdout-manifest "$holdout_manifest"
  --train-size "$TRAIN_SIZE"
  --validation-size "$VALIDATION_SIZE"
  --test-size "$TEST_SIZE"
  --step-size "$STEP_SIZE"
  --max-folds "$MAX_FOLDS"
  --l2 1
  --taker-fee-bps "$FEE_BPS"
  --max-load-memory-gb "$MAX_LOAD_MEMORY_GB"
  --stream
  --edge-thresholds-bps "$EDGE_THRESHOLDS_BPS"
  --sort-by validation_net_pnl
)

if [[ "$DRY_RUN" != "0" ]]; then
  printf 'would_run command='
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

if [[ "$(source_worktree_dirty)" == "true" ]]; then
  echo "refusing certified Kelly search from a dirty working tree; commit the exact source first" >&2
  exit 2
fi

"$PYTHON_BIN" -m lob_forge.study_plan \
  --profile kelly_candidate_search \
  --start "$START_DATE" \
  --end "$END_DATE" \
  --symbols "$SYMBOL" \
  --horizons-ms "$HORIZON_MS" \
  --fees-bps "$FEE_BPS" \
  --latency-ms "$LATENCY_MS" \
  --bucket-ms "$SOURCE_BUCKET_MS" \
  --execution-quote-resolution "$SOURCE_EXECUTION_QUOTE_RESOLUTION" \
  --feature-threshold "$SOURCE_FEATURE_THRESHOLD" \
  --large-trade-notional "$SOURCE_LARGE_TRADE_NOTIONAL" \
  --holdout-split-column "$HOLDOUT_SPLIT_COLUMN" \
  --holdout-values "$HOLDOUT_VALUES" \
  --max-quote-buckets "$SOURCE_MAX_QUOTE_BUCKETS" \
  --with-book-depth "$SOURCE_WITH_BOOK_DEPTH" \
  --train-size "$TRAIN_SIZE" \
  --validation-size "$VALIDATION_SIZE" \
  --test-size "$TEST_SIZE" \
  --step-size "$STEP_SIZE" \
  --edge-streaming 1 \
  --edge-thresholds-bps "$EDGE_THRESHOLDS_BPS" \
  --min-ram-gb 1 \
  --max-load-memory-gb "$MAX_LOAD_MEMORY_GB" \
  --max-feature-build-memory-gb 0 \
  --out-dir "$OUT_DIR" \
  --processed-root "$SOURCE_PROCESSED_ROOT" \
  --raw-root data/raw \
  --output "$plan"

"${command[@]}" > "$result"
"$PYTHON_BIN" -m lob_forge.cli audit-results "$result" \
  --assumed-cost-bps "$FEE_BPS" \
  --cost-safety-multiple 2 \
  --min-fold-count "$MIN_AUDIT_FOLD_COUNT" \
  --min-positive-fold-rate "$MIN_POSITIVE_FOLD_RATE" \
  --max-fold-contribution "$MAX_FOLD_CONTRIBUTION" \
  > "$audit"
"$PYTHON_BIN" -m lob_forge.study_provenance write \
  --plan "$plan" \
  --feature "$combined" \
  --holdout-manifest "$holdout_manifest" \
  --result "$result" \
  --audit "$audit" \
  --symbol "$SYMBOL" \
  --horizon-ms "$HORIZON_MS" \
  --taker-fee-bps "$FEE_BPS" \
  --output "$provenance"
"$PYTHON_BIN" -m lob_forge.cli kelly-variance-gate "$result" \
  --column validation_net_pnl \
  --min-observations "$MIN_AUDIT_FOLD_COUNT" \
  --window-size "$KELLY_WINDOW_SIZE" \
  --max-variance-cv "$MAX_VARIANCE_CV"
