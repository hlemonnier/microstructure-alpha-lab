#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONPATH

RESULT_DIR="${RESULT_DIR:-results/expected_edge_local16_20230516_20230714}"
PLAN_PATH="${PLAN_PATH:-$RESULT_DIR/run_plan.json}"
SOURCE_PROCESSED_ROOT="${SOURCE_PROCESSED_ROOT:-data/processed/expected_edge_60day_20230516_20230714}"
DRY_RUN="${DRY_RUN:-1}"
MAX_JOBS="${MAX_JOBS:-0}"
MAX_FOLDS="${MAX_FOLDS:-}"
MIN_AUDIT_FOLD_COUNT="${MIN_AUDIT_FOLD_COUNT:-20}"
RERUN_UNDERFOLDED="${RERUN_UNDERFOLDED:-1}"
MAX_LOAD_MEMORY_GB="${MAX_LOAD_MEMORY_GB:-10}"
LOB_FORGE_MAX_PROCESS_MEMORY_GB="${LOB_FORGE_MAX_PROCESS_MEMORY_GB:-8}"
EDGE_THRESHOLDS_BPS="${EDGE_THRESHOLDS_BPS:-0,0.05,0.1,0.2,0.5}"
FEES_BPS="${FEES_BPS:-}"

export LOB_FORGE_MAX_PROCESS_MEMORY_GB

read -r START_DATE END_DATE LATENCY_MS TRAIN_SIZE VALIDATION_SIZE TEST_SIZE STEP_SIZE DEFAULT_FEES <<< "$(
python3 - "$PLAN_PATH" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    plan = json.load(handle)
fees = " ".join(format(float(fee), "g") for fee in plan.get("fees_bps", []))
print(
    plan["start_date"],
    plan["end_date"],
    int(plan.get("latency_ms", 1000)),
    int(plan["train_size"]),
    int(plan["validation_size"]),
    int(plan["test_size"]),
    int(plan["step_size"]),
    fees,
)
PY
)"

if [[ -z "$FEES_BPS" ]]; then
  FEES_BPS="$DEFAULT_FEES"
fi

mkdir -p "$RESULT_DIR"
cleanup() {
  rm -f "$RESULT_DIR"/*.tmp.$$
}
trap cleanup EXIT
job_count=0

audit_meets_min_fold_count() {
  local audit_path="$1"
  python3 - "$audit_path" "$MIN_AUDIT_FOLD_COUNT" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
minimum = int(sys.argv[2])
try:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("empty audit")
    fold_count = int(float(rows[0].get("fold_count", "0") or 0))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if fold_count >= minimum else 1)
PY
}

while IFS=$'\t' read -r symbol horizon_ms combined; do
  if [[ "$MAX_JOBS" != "0" && "$job_count" -ge "$MAX_JOBS" ]]; then
    break
  fi
  job_count=$((job_count + 1))
  for fee in $FEES_BPS; do
    safe_fee="$(printf '%s' "$fee" | tr '.' 'p')"
    result="$RESULT_DIR/${symbol}_${horizon_ms}ms_fee_${safe_fee}_edge.csv"
    audit="$RESULT_DIR/${symbol}_${horizon_ms}ms_fee_${safe_fee}_edge_audit.csv"
    if [[ -s "$result" && -s "$audit" ]]; then
      if [[ "$RERUN_UNDERFOLDED" == "1" ]] && ! audit_meets_min_fold_count "$audit"; then
        printf 'rerun underfolded symbol=%s horizon_ms=%s fee=%s audit=%s min_audit_fold_count=%s\n' \
          "$symbol" "$horizon_ms" "$fee" "$audit" "$MIN_AUDIT_FOLD_COUNT"
      else
        printf 'skip existing symbol=%s horizon_ms=%s fee=%s\n' "$symbol" "$horizon_ms" "$fee"
        continue
      fi
    fi
    if [[ "$DRY_RUN" != "0" ]]; then
      printf 'would_run symbol=%s horizon_ms=%s fee=%s combined=%s result=%s\n' \
        "$symbol" "$horizon_ms" "$fee" "$combined" "$result"
      continue
    fi
    printf 'run symbol=%s horizon_ms=%s fee=%s combined=%s result=%s\n' \
      "$symbol" "$horizon_ms" "$fee" "$combined" "$result"
    result_tmp="$result.tmp.$$"
    audit_tmp="$audit.tmp.$$"
    max_folds_args=()
    if [[ -n "$MAX_FOLDS" ]]; then
      max_folds_args=(--max-folds "$MAX_FOLDS")
    fi
    python3 -m lob_forge.cli edge-walk-forward "$combined" \
      --train-size "$TRAIN_SIZE" \
      --validation-size "$VALIDATION_SIZE" \
      --test-size "$TEST_SIZE" \
      --step-size "$STEP_SIZE" \
      "${max_folds_args[@]}" \
      --l2 1 \
      --taker-fee-bps "$fee" \
      --max-load-memory-gb "$MAX_LOAD_MEMORY_GB" \
      --stream \
      --edge-thresholds-bps "$EDGE_THRESHOLDS_BPS" \
      --sort-by validation_net_pnl \
      > "$result_tmp"
    mv "$result_tmp" "$result"
    python3 -m lob_forge.cli audit-results "$result" \
      --assumed-cost-bps "$fee" \
      --cost-safety-multiple 2 \
      > "$audit_tmp"
    mv "$audit_tmp" "$audit"
  done
done < <(
python3 - "$PLAN_PATH" "$SOURCE_PROCESSED_ROOT" "$START_DATE" "$END_DATE" "$LATENCY_MS" <<'PY'
import json
import sys
from pathlib import Path

plan_path, root_arg, start, end, latency = sys.argv[1:6]
root = Path(root_arg)
with open(plan_path) as handle:
    plan = json.load(handle)
for symbol in plan.get("symbols", []):
    symbol = str(symbol).upper()
    for horizon_ms in plan.get("horizons_ms", []):
        horizon_ms = int(horizon_ms)
        combined = root / f"{symbol.lower()}_{horizon_ms}ms_latency_{latency}" / f"{symbol}-{start}_{end}-combined-features.csv"
        if combined.exists() and combined.stat().st_size > 0:
            print("\t".join([symbol, str(horizon_ms), str(combined)]))
        else:
            print(f"missing_combined={combined}", file=sys.stderr)
PY
)

if [[ "$DRY_RUN" == "0" ]]; then
  python3 - "$RESULT_DIR" <<'PY'
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
  python3 -m lob_forge.cli pvalue-correction "$RESULT_DIR/pvalues.csv" > "$RESULT_DIR/pvalue_corrections.csv"
  python3 -m lob_forge.study_status \
    --plan "$PLAN_PATH" \
    --result-dir "$RESULT_DIR" \
    --output "$RESULT_DIR/study_status.json" \
    --min-audit-fold-count "$MIN_AUDIT_FOLD_COUNT" \
    || true
fi
