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
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-$RESUME}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUT_DIR/checkpoints}"
PREDICTION_DIR="${PREDICTION_DIR:-$OUT_DIR/predictions}"
HOLDOUT_MANIFEST_PATH="${HOLDOUT_MANIFEST_PATH:-${HOLDOUT_MANIFEST:-}}"
DEVELOPMENT_L2_DIR="${DEVELOPMENT_L2_DIR:-$OUT_DIR/development_l2}"

MIN_FOLD_COUNT="${MIN_FOLD_COUNT:-20}"
MIN_L2_ROWS="${MIN_L2_ROWS:-1000}"
DEPTH="${DEPTH:-5}"
WINDOW="${WINDOW:-16}"
LABEL_HORIZON="${LABEL_HORIZON:-1}"
FLAT_THRESHOLD_BPS="${FLAT_THRESHOLD_BPS:-0}"
EPOCHS="${EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-3}"
DEVICE="${DEVICE:-auto}"
CLASS_WEIGHTING="${CLASS_WEIGHTING:-balanced}"
LR_SCHEDULER_GAMMA="${LR_SCHEDULER_GAMMA:-1}"
ECONOMIC_TARGET_NOTIONAL="${ECONOMIC_TARGET_NOTIONAL:-100}"
ECONOMIC_TAKER_FEE_BPS="${ECONOMIC_TAKER_FEE_BPS:-1}"
ECONOMIC_SLIPPAGE_BPS="${ECONOMIC_SLIPPAGE_BPS:-0}"
MAX_ROWS="${MAX_ROWS:-100000}"
MAX_SNAPSHOTS="${MAX_SNAPSHOTS:-2000}"
SEED="${SEED:-7}"
SEEDS="${SEEDS:-$SEED}"

mkdir -p "$OUT_DIR" "$CHECKPOINT_DIR" "$PREDICTION_DIR"
if [[ -n "$HOLDOUT_MANIFEST_PATH" ]]; then
  mkdir -p "$DEVELOPMENT_L2_DIR"
fi
normalized_seeds="${SEEDS//,/ }"
read -r -a seed_values <<< "$normalized_seeds"
seed_count="${#seed_values[@]}"

printf 'models="%s"\n' "$MODELS"
printf 'baseline_audit=%s l2=%s out_dir=%s dry_run=%s resume=%s resume_checkpoint=%s seeds="%s"\n' \
  "$BASELINE_AUDIT_PATH" "$L2_PATH" "$OUT_DIR" "$DRY_RUN" "$RESUME" "$RESUME_CHECKPOINT" "$SEEDS"
if [[ -n "$HOLDOUT_MANIFEST_PATH" ]]; then
  printf 'holdout_manifest=%s development_l2_dir=%s\n' "$HOLDOUT_MANIFEST_PATH" "$DEVELOPMENT_L2_DIR"
fi
printf 'sequence_params=depth=%s window=%s label_horizon=%s min_fold_count=%s min_l2_rows=%s epochs=%s batch_size=%s patience=%s device=%s class_weighting=%s lr_gamma=%s\n' \
  "$DEPTH" "$WINDOW" "$LABEL_HORIZON" "$MIN_FOLD_COUNT" "$MIN_L2_ROWS" "$EPOCHS" "$BATCH_SIZE" "$EARLY_STOPPING_PATIENCE" "$DEVICE" "$CLASS_WEIGHTING" "$LR_SCHEDULER_GAMMA"

for model in $MODELS; do
  case "$model" in
    sequence_transformer|sequence_tcn)
      ;;
    *)
      echo "unknown model=$model; expected sequence_transformer or sequence_tcn" >&2
      exit 2
      ;;
  esac

  for seed in "${seed_values[@]}"; do
    suffix=""
    if [[ "$seed_count" -gt 1 ]]; then
      suffix="_seed_${seed}"
    fi
    output="$OUT_DIR/${model}${suffix}_results.csv"
    checkpoint="$CHECKPOINT_DIR/${model}${suffix}.pt"
    predictions="$PREDICTION_DIR/${model}${suffix}_predictions.csv"
    development_l2="$DEVELOPMENT_L2_DIR/${model}${suffix}_development_l2.csv"
    if [[ "$RESUME" == "1" && -s "$output" ]]; then
      printf 'skip existing model=%s seed=%s output=%s\n' "$model" "$seed" "$output"
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
      --batch-size "$BATCH_SIZE"
      --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
      --device "$DEVICE"
      --class-weighting "$CLASS_WEIGHTING"
      --lr-scheduler-gamma "$LR_SCHEDULER_GAMMA"
      --checkpoint-path "$checkpoint"
      --predictions-output "$predictions"
      --economic-target-notional "$ECONOMIC_TARGET_NOTIONAL"
      --economic-taker-fee-bps "$ECONOMIC_TAKER_FEE_BPS"
      --economic-slippage-bps "$ECONOMIC_SLIPPAGE_BPS"
      --max-rows "$MAX_ROWS"
      --max-snapshots "$MAX_SNAPSHOTS"
      --min-fold-count "$MIN_FOLD_COUNT"
      --min-l2-rows "$MIN_L2_ROWS"
      --seed "$seed"
    )
    if [[ -n "$HOLDOUT_MANIFEST_PATH" ]]; then
      command+=(--holdout-manifest "$HOLDOUT_MANIFEST_PATH" --development-l2-output "$development_l2")
    fi
    if [[ "$RESUME_CHECKPOINT" == "1" && -s "$checkpoint" ]]; then
      command+=(--resume-from-checkpoint)
    fi

    if [[ "$DRY_RUN" != "0" ]]; then
      printf 'would_run model=%s seed=%s output=%s checkpoint=%s predictions=%s command=' "$model" "$seed" "$output" "$checkpoint" "$predictions"
      printf '%q ' "${command[@]}"
      printf '\n'
      continue
    fi

    printf 'run model=%s seed=%s output=%s checkpoint=%s predictions=%s\n' "$model" "$seed" "$output" "$checkpoint" "$predictions"
    "${command[@]}"
  done
done
