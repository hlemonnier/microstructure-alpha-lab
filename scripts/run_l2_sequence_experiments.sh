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

MODELS="${MODELS:-sequence_transformer sequence_tcn}"
BASELINE_AUDIT_PATH="${BASELINE_AUDIT_PATH:-results/current/btc_full_day_edge_zero_fee_audit.csv}"
L2_PATH="${L2_PATH:-data/normalized_l2/bybit/BTCUSDT/2023-05-16.csv}"
OUT_DIR="${OUT_DIR:-results/model_experiments}"
DRY_RUN="${DRY_RUN:-1}"
RESUME="${RESUME:-1}"

MIN_FOLD_COUNT="${MIN_FOLD_COUNT:-20}"
MIN_L2_ROWS="${MIN_L2_ROWS:-1000}"
DEPTH="${DEPTH:-5}"
WINDOW="${WINDOW:-16}"
LABEL_HORIZON="${LABEL_HORIZON:-1}"
FLAT_THRESHOLD_BPS="${FLAT_THRESHOLD_BPS:-0}"
EPOCHS="${EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
MAX_ROWS="${MAX_ROWS:-100000}"
MAX_SNAPSHOTS="${MAX_SNAPSHOTS:-2000}"
SEED="${SEED:-7}"

mkdir -p "$OUT_DIR"

printf 'models="%s"\n' "$MODELS"
printf 'baseline_audit=%s l2=%s out_dir=%s dry_run=%s resume=%s\n' \
  "$BASELINE_AUDIT_PATH" "$L2_PATH" "$OUT_DIR" "$DRY_RUN" "$RESUME"
printf 'sequence_params=depth=%s window=%s label_horizon=%s min_fold_count=%s min_l2_rows=%s epochs=%s\n' \
  "$DEPTH" "$WINDOW" "$LABEL_HORIZON" "$MIN_FOLD_COUNT" "$MIN_L2_ROWS" "$EPOCHS"

for model in $MODELS; do
  case "$model" in
    sequence_transformer|sequence_tcn)
      ;;
    *)
      echo "unknown model=$model; expected sequence_transformer or sequence_tcn" >&2
      exit 2
      ;;
  esac

  output="$OUT_DIR/${model}_results.csv"
  if [[ "$RESUME" == "1" && -s "$output" ]]; then
    printf 'skip existing model=%s output=%s\n' "$model" "$output"
    continue
  fi

  command=(
    "$PYTHON_BIN" -m lob_forge.cli l2-sequence-experiment
    --model "$model"
    --baseline-audit "$BASELINE_AUDIT_PATH"
    --l2 "$L2_PATH"
    --output "$output"
    --depth "$DEPTH"
    --window "$WINDOW"
    --label-horizon "$LABEL_HORIZON"
    --flat-threshold-bps "$FLAT_THRESHOLD_BPS"
    --epochs "$EPOCHS"
    --learning-rate "$LEARNING_RATE"
    --max-rows "$MAX_ROWS"
    --max-snapshots "$MAX_SNAPSHOTS"
    --min-fold-count "$MIN_FOLD_COUNT"
    --min-l2-rows "$MIN_L2_ROWS"
    --seed "$SEED"
  )

  if [[ "$DRY_RUN" != "0" ]]; then
    printf 'would_run model=%s output=%s command=' "$model" "$output"
    printf '%q ' "${command[@]}"
    printf '\n'
    continue
  fi

  printf 'run model=%s output=%s\n' "$model" "$output"
  "${command[@]}"
done
