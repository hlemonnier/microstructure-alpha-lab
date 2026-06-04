#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONPATH

STUDY_PROFILE="${STUDY_PROFILE:-laptop_tiny}"
PLAN_ONLY="${PLAN_ONLY:-0}"

case "$STUDY_PROFILE" in
  laptop_tiny)
    : "${START_DATE:=2023-05-16}"
    : "${END_DATE:=2023-05-17}"
    : "${SYMBOLS:=BTCUSDT}"
    : "${HORIZONS_MS:=5000}"
    : "${FEES_BPS:=0 0.1 0.5}"
    : "${LATENCY_MS:=1000}"
    : "${BUCKET_MS:=1000}"
    : "${MAX_QUOTE_BUCKETS:=1200}"
    : "${TRAIN_SIZE:=900}"
    : "${VALIDATION_SIZE:=450}"
    : "${TEST_SIZE:=450}"
    : "${STEP_SIZE:=450}"
    : "${STUDY_TAG:=expected_edge_laptop_tiny_${START_DATE//-/}_${END_DATE//-/}}"
    : "${MIN_RAM_GB:=4}"
    : "${MAX_LOAD_MEMORY_GB:=4}"
    : "${MAX_FEATURE_BUILD_MEMORY_GB:=4}"
    : "${LOB_FORGE_MAX_PROCESS_MEMORY_GB:=6}"
    : "${FEATURE_MEMORY_ESTIMATE_MULTIPLIER:=12}"
    : "${WITH_BOOK_DEPTH:=0}"
    : "${EDGE_STREAMING:=1}"
    : "${EDGE_THRESHOLDS_BPS:=0,0.1,0.25,0.5}"
    : "${MIN_AUDIT_FOLD_COUNT:=1}"
    ;;
  laptop_quick)
    : "${START_DATE:=2023-05-16}"
    : "${END_DATE:=2023-05-22}"
    : "${SYMBOLS:=BTCUSDT ETHUSDT}"
    : "${HORIZONS_MS:=5000}"
    : "${FEES_BPS:=0 0.05 0.1 0.25 0.5}"
    : "${LATENCY_MS:=1000}"
    : "${BUCKET_MS:=1000}"
    : "${MAX_QUOTE_BUCKETS:=3600}"
    : "${TRAIN_SIZE:=7200}"
    : "${VALIDATION_SIZE:=3600}"
    : "${TEST_SIZE:=3600}"
    : "${STEP_SIZE:=3600}"
    : "${STUDY_TAG:=expected_edge_laptop_quick_${START_DATE//-/}_${END_DATE//-/}}"
    : "${MIN_RAM_GB:=8}"
    : "${MAX_LOAD_MEMORY_GB:=8}"
    : "${MAX_FEATURE_BUILD_MEMORY_GB:=8}"
    : "${LOB_FORGE_MAX_PROCESS_MEMORY_GB:=10}"
    : "${FEATURE_MEMORY_ESTIMATE_MULTIPLIER:=12}"
    : "${WITH_BOOK_DEPTH:=1}"
    : "${EDGE_STREAMING:=1}"
    : "${EDGE_THRESHOLDS_BPS:=0,0.05,0.1,0.2,0.5}"
    : "${MIN_AUDIT_FOLD_COUNT:=1}"
    ;;
  local16|local16_60day)
    if [[ "$PLAN_ONLY" != "1" && "${CONFIRM_LOCAL16:-0}" != "1" ]]; then
      cat >&2 <<'EOF'
Refusing to start local16_60day without CONFIRM_LOCAL16=1.
This capped 60-day profile is still too heavy to treat as a normal 16 GB laptop run.
Use `bash scripts/run_60day_expected_edge_study.sh` for the safe laptop smoke,
or run `PLAN_ONLY=1 STUDY_PROFILE=local16_60day bash scripts/run_60day_expected_edge_study.sh` first.
For serious evidence, use the cloud path in docs/full_study_cloud_run.md.
EOF
      exit 2
    fi
    : "${START_DATE:=2023-05-16}"
    : "${END_DATE:=2023-07-14}"
    : "${SYMBOLS:=BTCUSDT ETHUSDT}"
    : "${HORIZONS_MS:=5000}"
    : "${FEES_BPS:=0 0.05 0.1 0.25 0.5}"
    : "${LATENCY_MS:=1000}"
    : "${BUCKET_MS:=1000}"
    : "${MAX_QUOTE_BUCKETS:=3600}"
    : "${TRAIN_SIZE:=21600}"
    : "${VALIDATION_SIZE:=7200}"
    : "${TEST_SIZE:=7200}"
    : "${STEP_SIZE:=7200}"
    : "${STUDY_TAG:=expected_edge_local16_${START_DATE//-/}_${END_DATE//-/}}"
    : "${MIN_RAM_GB:=16}"
    : "${MAX_LOAD_MEMORY_GB:=8}"
    : "${MAX_FEATURE_BUILD_MEMORY_GB:=6}"
    : "${LOB_FORGE_MAX_PROCESS_MEMORY_GB:=8}"
    : "${FEATURE_MEMORY_ESTIMATE_MULTIPLIER:=12}"
    : "${WITH_BOOK_DEPTH:=0}"
    : "${EDGE_STREAMING:=1}"
    : "${EDGE_THRESHOLDS_BPS:=0,0.05,0.1,0.2,0.5}"
    : "${MIN_AUDIT_FOLD_COUNT:=20}"
    ;;
  full|cloud_full)
    if [[ "${CONFIRM_HEAVY:-0}" != "1" ]]; then
      cat >&2 <<'EOF'
Refusing to start the full study on the default profile.
The full profile is intended for a deliberate long data run, not accidental local execution.
Run with CONFIRM_HEAVY=1 STUDY_PROFILE=cloud_full only when you accept the disk, network, and runtime cost.
EOF
      exit 2
    fi
    : "${START_DATE:=2023-05-16}"
    : "${END_DATE:=2023-07-14}"
    : "${SYMBOLS:=BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT}"
    : "${HORIZONS_MS:=5000 10000}"
    : "${FEES_BPS:=0 0.05 0.1 0.25 0.5 1 2 5}"
    : "${LATENCY_MS:=1000}"
    : "${BUCKET_MS:=1000}"
    : "${MAX_QUOTE_BUCKETS:=}"
    : "${TRAIN_SIZE:=72000}"
    : "${VALIDATION_SIZE:=18000}"
    : "${TEST_SIZE:=18000}"
    : "${STEP_SIZE:=18000}"
    : "${STUDY_TAG:=expected_edge_60day_${START_DATE//-/}_${END_DATE//-/}}"
    : "${MIN_RAM_GB:=64}"
    : "${MAX_LOAD_MEMORY_GB:=48}"
    : "${MAX_FEATURE_BUILD_MEMORY_GB:=48}"
    : "${LOB_FORGE_MAX_PROCESS_MEMORY_GB:=0}"
    : "${FEATURE_MEMORY_ESTIMATE_MULTIPLIER:=12}"
    : "${WITH_BOOK_DEPTH:=1}"
    : "${EDGE_STREAMING:=1}"
    : "${EDGE_THRESHOLDS_BPS:=0,0.025,0.05,0.075,0.1,0.15,0.2,0.3,0.5,0.75,1,2,5}"
    : "${MIN_AUDIT_FOLD_COUNT:=20}"
    ;;
  *)
    echo "unknown STUDY_PROFILE=$STUDY_PROFILE; expected laptop_tiny, laptop_quick, local16_60day, or cloud_full" >&2
    exit 2
    ;;
esac

OUT_DIR="${OUT_DIR:-results/${STUDY_TAG}}"
RAW_ROOT="${RAW_ROOT:-data/raw}"
PROCESSED_ROOT="${PROCESSED_ROOT:-data/processed/${STUDY_TAG}}"
RESUME="${RESUME:-1}"
PREFLIGHT="${PREFLIGHT:-quick}"
WRITE_PVALUES="${WRITE_PVALUES:-1}"
VERIFY_CHECKSUM="${VERIFY_CHECKSUM:-1}"
EXISTING_FEATURES_ONLY="${EXISTING_FEATURES_ONLY:-0}"
MAX_FOLDS="${MAX_FOLDS:-}"
MIN_AUDIT_FOLD_COUNT="${MIN_AUDIT_FOLD_COUNT:-1}"

export WITH_BOOK_DEPTH LOB_FORGE_MAX_PROCESS_MEMORY_GB

mkdir -p "$OUT_DIR" "$PROCESSED_ROOT"
PLAN_PATH="${PLAN_PATH:-$OUT_DIR/run_plan.json}"

printf 'study_profile=%s\n' "$STUDY_PROFILE"
printf 'date_range=%s..%s symbols="%s" horizons_ms="%s" fees_bps="%s" max_quote_buckets="%s" with_book_depth=%s\n' \
  "$START_DATE" "$END_DATE" "$SYMBOLS" "$HORIZONS_MS" "$FEES_BPS" "$MAX_QUOTE_BUCKETS" "$WITH_BOOK_DEPTH"
printf 'memory_guard=min_ram_gb=%s max_csv_load_gb=%s max_feature_build_gb=%s edge_streaming=%s\n' \
  "$MIN_RAM_GB" "$MAX_LOAD_MEMORY_GB" "$MAX_FEATURE_BUILD_MEMORY_GB" "$EDGE_STREAMING"
printf 'runtime_memory_limit_gb=%s\n' "$LOB_FORGE_MAX_PROCESS_MEMORY_GB"
printf 'edge_thresholds_bps=%s\n' "$EDGE_THRESHOLDS_BPS"
printf 'out_dir=%s processed_root=%s verify_checksum=%s existing_features_only=%s max_folds=%s min_audit_fold_count=%s\n' "$OUT_DIR" "$PROCESSED_ROOT" "$VERIFY_CHECKSUM" "$EXISTING_FEATURES_ONLY" "$MAX_FOLDS" "$MIN_AUDIT_FOLD_COUNT"

python3 -m lob_forge.study_plan \
  --profile "$STUDY_PROFILE" \
  --start "$START_DATE" \
  --end "$END_DATE" \
  --symbols "$SYMBOLS" \
  --horizons-ms "$HORIZONS_MS" \
  --fees-bps "$FEES_BPS" \
  --latency-ms "$LATENCY_MS" \
  --max-quote-buckets "$MAX_QUOTE_BUCKETS" \
  --with-book-depth "$WITH_BOOK_DEPTH" \
  --train-size "$TRAIN_SIZE" \
  --validation-size "$VALIDATION_SIZE" \
  --test-size "$TEST_SIZE" \
  --step-size "$STEP_SIZE" \
  --edge-streaming "$EDGE_STREAMING" \
  --min-ram-gb "$MIN_RAM_GB" \
  --max-load-memory-gb "$MAX_LOAD_MEMORY_GB" \
  --max-feature-build-memory-gb "$MAX_FEATURE_BUILD_MEMORY_GB" \
  --out-dir "$OUT_DIR" \
  --processed-root "$PROCESSED_ROOT" \
  --raw-root "$RAW_ROOT" \
  --output "$PLAN_PATH"

if [[ "$PLAN_ONLY" == "1" ]]; then
  printf 'plan_only=1; not starting downloads or edge evaluation\n'
  exit 0
fi

if [[ "${ALLOW_LOW_RAM:-0}" != "1" ]]; then
  python3 - "$MIN_RAM_GB" <<'PY'
import sys

from lob_forge.memory_guard import assert_physical_memory

minimum = float(sys.argv[1])
try:
    ram_gb = assert_physical_memory(min_memory_gb=minimum)
except ValueError as exc:
    print(str(exc), file=sys.stderr)
    print("Set ALLOW_LOW_RAM=1 only if you intentionally want to bypass this preflight.", file=sys.stderr)
    raise SystemExit(2)
if ram_gb is None:
    print(f"physical_ram=unknown min_required_gb={minimum:.1f}; continuing", file=sys.stderr)
else:
    print(f"physical_ram_gb={ram_gb:.1f} min_required_gb={minimum:.1f}")
PY
fi

if [[ "$PREFLIGHT" != "0" ]]; then
  python3 - "$START_DATE" "$END_DATE" "$SYMBOLS" <<'PY'
import socket
import sys
import time
import urllib.error
import urllib.request

from lob_forge.binance_vision import archive_key, iter_dates, url_for_key

start, end, symbols_arg = sys.argv[1:4]
symbols = symbols_arg.split()
datasets = ["bookTicker", "aggTrades"]
if __import__("os").environ.get("WITH_BOOK_DEPTH", "1") == "1":
    datasets.append("bookDepth")
preflight = __import__("os").environ.get("PREFLIGHT", "quick")
head_retries = int(__import__("os").environ.get("PREFLIGHT_HEAD_RETRIES", "3"))
dates = list(iter_dates(start, end))
dates_to_check = dates if preflight == "full" else sorted({dates[0], dates[-1]})
missing = []
transient = []
checked = 0
for symbol in symbols:
    for date_value in dates_to_check:
        for dataset in datasets:
            key = archive_key(
                market="futures/um",
                frequency="daily",
                dataset=dataset,
                symbol=symbol,
                date_value=date_value,
            )
            request = urllib.request.Request(url_for_key(key), method="HEAD")
            for attempt in range(1, head_retries + 1):
                try:
                    with urllib.request.urlopen(request, timeout=30):
                        checked += 1
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code == 404:
                        missing.append(key)
                        break
                    if attempt == head_retries:
                        raise
                    time.sleep(2 * attempt)
                except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionResetError, OSError) as exc:
                    if attempt == head_retries:
                        transient.append((key, repr(exc)))
                        break
                    time.sleep(2 * attempt)
if missing:
    print("missing required Binance Vision archives:", file=sys.stderr)
    for key in missing[:20]:
        print(key, file=sys.stderr)
    if len(missing) > 20:
        print(f"... {len(missing) - 20} more", file=sys.stderr)
    raise SystemExit(1)
if transient:
    print("preflight transient warnings; continuing because downloads have their own retries:", file=sys.stderr)
    for key, error in transient[:10]:
        print(f"{key}: {error}", file=sys.stderr)
    if len(transient) > 10:
        print(f"... {len(transient) - 10} more", file=sys.stderr)
print(f"preflight_ok mode={preflight} archives={checked}")
PY
fi

min_tick_for_symbol() {
  case "$1" in
    BTCUSDT) echo "0.1" ;;
    ETHUSDT) echo "0.01" ;;
    BNBUSDT) echo "0.01" ;;
    SOLUSDT) echo "0.001" ;;
    XRPUSDT) echo "0.0001" ;;
    *) echo "0.01" ;;
  esac
}

build_args_for_max_buckets() {
  if [[ -n "$MAX_QUOTE_BUCKETS" ]]; then
    printf -- '--max-quote-buckets %s' "$MAX_QUOTE_BUCKETS"
  fi
}

build_args_for_book_depth() {
  if [[ "$WITH_BOOK_DEPTH" == "1" ]]; then
    printf -- '--with-book-depth'
  fi
}

build_args_for_verify() {
  if [[ "$VERIFY_CHECKSUM" == "0" ]]; then
    printf -- '--no-verify'
  fi
}

edge_stream_args_for_cli() {
  if [[ "$EDGE_STREAMING" == "1" ]]; then
    printf -- '--stream'
  else
    printf -- '--no-stream'
  fi
}

edge_threshold_args_for_cli() {
  if [[ -n "$EDGE_THRESHOLDS_BPS" ]]; then
    printf -- '--edge-thresholds-bps %s' "$EDGE_THRESHOLDS_BPS"
  fi
}

max_fold_args_for_cli() {
  if [[ -n "$MAX_FOLDS" ]]; then
    printf -- '--max-folds %s' "$MAX_FOLDS"
  fi
}

run_one() {
  local symbol="$1"
  local horizon_ms="$2"
  local symbol_lower
  symbol_lower="$(printf '%s' "$symbol" | tr '[:upper:]' '[:lower:]')"
  local symbol_dir="$PROCESSED_ROOT/${symbol_lower}_${horizon_ms}ms_latency_${LATENCY_MS}"
  local combined="$symbol_dir/${symbol}-${START_DATE}_${END_DATE}-combined-features.csv"
  local min_tick
  min_tick="$(min_tick_for_symbol "$symbol")"

  mkdir -p "$symbol_dir"

  if [[ "$RESUME" == "1" && -s "$combined" ]]; then
    printf 'skip existing combined features: %s\n' "$combined"
  elif [[ "$EXISTING_FEATURES_ONLY" == "1" ]]; then
    python3 - "$symbol" "$START_DATE" "$END_DATE" "$symbol_dir" "$combined" <<'PY'
import sys
from pathlib import Path

from lob_forge.binance_vision import iter_dates
from lob_forge.features import combine_feature_csvs

symbol, start, end, symbol_dir_arg, combined_arg = sys.argv[1:6]
symbol_dir = Path(symbol_dir_arg)
combined = Path(combined_arg)
inputs = []
missing = []
for date_value in iter_dates(start, end):
    feature_csv = symbol_dir / f"{symbol.upper()}-{date_value}-quote-trade-features.csv"
    done_marker = feature_csv.with_suffix(feature_csv.suffix + ".done")
    if done_marker.exists() and feature_csv.exists() and feature_csv.stat().st_size > 0:
        inputs.append((feature_csv, {"source_symbol": symbol.upper(), "source_date": date_value}))
    else:
        missing.append(date_value)

if not inputs:
    raise SystemExit(f"no existing daily feature files found in {symbol_dir}")

combine_feature_csvs(inputs, combined)
print(
    f"combined_existing_features symbol={symbol.upper()} available_days={len(inputs)} "
    f"missing_days={len(missing)} output={combined}"
)
if missing:
    print("missing_existing_feature_dates=" + ",".join(missing[:20]))
    if len(missing) > 20:
        print(f"missing_existing_feature_dates_more={len(missing) - 20}")
PY
  else
    # shellcheck disable=SC2046
    python3 -m lob_forge.cli build-range \
      --symbol "$symbol" \
      --start "$START_DATE" \
      --end "$END_DATE" \
      --raw-root "$RAW_ROOT" \
      --output-dir "$symbol_dir" \
      --combined-output "$combined" \
      --bucket-ms "$BUCKET_MS" \
      --horizon-ms "$horizon_ms" \
      --execution-latency-ms "$LATENCY_MS" \
      --threshold half_spread \
      --min-tick "$min_tick" \
      --max-feature-build-memory-gb "$MAX_FEATURE_BUILD_MEMORY_GB" \
      --feature-memory-estimate-multiplier "$FEATURE_MEMORY_ESTIMATE_MULTIPLIER" \
      $(build_args_for_book_depth) \
      $(build_args_for_max_buckets) \
      $(build_args_for_verify)
  fi

  for fee_bps in $FEES_BPS; do
    local safe_fee
    safe_fee="$(printf '%s' "$fee_bps" | tr '.' 'p')"
    local result="$OUT_DIR/${symbol}_${horizon_ms}ms_fee_${safe_fee}_edge.csv"
    local audit="$OUT_DIR/${symbol}_${horizon_ms}ms_fee_${safe_fee}_edge_audit.csv"

    if [[ "$RESUME" == "1" && -s "$result" && -s "$audit" ]]; then
      printf 'skip existing result/audit: %s %s\n' "$result" "$audit"
    else
      python3 -m lob_forge.cli edge-walk-forward "$combined" \
        --train-size "$TRAIN_SIZE" \
        --validation-size "$VALIDATION_SIZE" \
        --test-size "$TEST_SIZE" \
        --step-size "$STEP_SIZE" \
        --l2 1 \
        --taker-fee-bps "$fee_bps" \
        --max-load-memory-gb "$MAX_LOAD_MEMORY_GB" \
        $(edge_stream_args_for_cli) \
        $(edge_threshold_args_for_cli) \
        $(max_fold_args_for_cli) \
        --sort-by validation_net_pnl \
        > "$result"

      python3 -m lob_forge.cli audit-results "$result" \
        --assumed-cost-bps "$fee_bps" \
        --cost-safety-multiple 2 \
        > "$audit"
    fi
  done
}

for symbol in $SYMBOLS; do
  for horizon_ms in $HORIZONS_MS; do
    run_one "$symbol" "$horizon_ms"
  done
done

if [[ "$WRITE_PVALUES" == "1" ]]; then
python3 - "$OUT_DIR" <<'PY'
import csv
import glob
import os
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
with (out_dir / "pvalues.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["hypothesis_id", "metric", "p_value"])
    writer.writeheader()
    for path in sorted(glob.glob(str(out_dir / "*_audit.csv"))):
        with open(path, newline="") as audit_handle:
            row = next(csv.DictReader(audit_handle))
        writer.writerow(
            {
                "hypothesis_id": os.path.basename(path).replace("_audit.csv", ""),
                "metric": "fold_mean_net_pnl",
                "p_value": row["one_sided_p_value_mean_le_zero"],
            }
        )
PY

python3 -m lob_forge.cli pvalue-correction "$OUT_DIR/pvalues.csv" > "$OUT_DIR/pvalue_corrections.csv"
fi

study_status_pvalue_args=()
if [[ "$WRITE_PVALUES" != "1" ]]; then
  study_status_pvalue_args+=(--no-pvalues)
fi

python3 -m lob_forge.study_status \
  --plan "$PLAN_PATH" \
  --result-dir "$OUT_DIR" \
  --output "$OUT_DIR/study_status.json" \
  --min-audit-fold-count "$MIN_AUDIT_FOLD_COUNT" \
  "${study_status_pvalue_args[@]}"

printf 'wrote %s expected-edge study artifacts to %s\n' "$STUDY_PROFILE" "$OUT_DIR"
