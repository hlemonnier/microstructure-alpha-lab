#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONPATH

SYMBOL="${SYMBOL:-BTCUSDT}"
START_DATE="${START_DATE:-2023-05-16}"
END_DATE="${END_DATE:-$START_DATE}"
MANIFEST_PATH="${MANIFEST_PATH:-data/manifests/bybit_l2_smoke_${SYMBOL}_${START_DATE//-/}_${END_DATE//-/}.csv}"
RAW_ROOT="${RAW_ROOT:-data/raw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data}"
MAX_IMPORT_ROWS="${MAX_IMPORT_ROWS:-5000}"
MIN_L2_ROWS="${MIN_L2_ROWS:-1000}"
THROTTLE_SECONDS="${THROTTLE_SECONDS:-0}"
DOWNLOAD="${DOWNLOAD:-1}"
IMPORT="${IMPORT:-1}"
RUN_MODEL_GATE="${RUN_MODEL_GATE:-1}"
BASELINE_AUDIT="${BASELINE_AUDIT:-results/current/btc_full_day_edge_zero_fee_audit.csv}"

printf 'bybit_l2_smoke symbol=%s date_range=%s..%s max_import_rows=%s\n' "$SYMBOL" "$START_DATE" "$END_DATE" "$MAX_IMPORT_ROWS"
printf 'manifest=%s raw_root=%s output_root=%s\n' "$MANIFEST_PATH" "$RAW_ROOT" "$OUTPUT_ROOT"

python3 -m lob_forge.cli l2-manifest \
  --source bybit \
  --symbols "$SYMBOL" \
  --start "$START_DATE" \
  --end "$END_DATE" \
  --output "$MANIFEST_PATH"

python3 -m lob_forge.cli l2-resolve-bybit \
  "$MANIFEST_PATH" \
  --throttle-seconds "$THROTTLE_SECONDS"

if [[ "$DOWNLOAD" == "1" ]]; then
  python3 -m lob_forge.cli l2-download-manifest \
    "$MANIFEST_PATH" \
    --raw-root "$RAW_ROOT"
else
  printf 'download_skipped=1\n'
fi

if [[ "$IMPORT" == "1" ]]; then
  python3 -m lob_forge.cli l2-import-manifest \
    "$MANIFEST_PATH" \
    --output-root "$OUTPUT_ROOT" \
    --format csv \
    --max-import-rows "$MAX_IMPORT_ROWS" \
    --require-sequence
else
  printf 'import_skipped=1\n'
fi

NORMALIZED_PATH="$OUTPUT_ROOT/normalized_l2/bybit/$SYMBOL/$START_DATE.csv"
printf 'normalized_path=%s\n' "$NORMALIZED_PATH"

if [[ "$RUN_MODEL_GATE" == "1" && -f "$NORMALIZED_PATH" ]]; then
  set +e
  python3 -m lob_forge.cli model-readiness-gate \
    --model sequence_transformer \
    --baseline-audit "$BASELINE_AUDIT" \
    --l2 "$NORMALIZED_PATH" \
    --min-fold-count 20 \
    --min-l2-rows "$MIN_L2_ROWS"
  gate_status=$?
  set -e
  printf 'model_readiness_exit_code=%s\n' "$gate_status"
fi
