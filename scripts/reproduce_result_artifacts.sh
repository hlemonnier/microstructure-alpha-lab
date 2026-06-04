#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONPATH

OUT_DIR="${OUT_DIR:-results/current}"
mkdir -p "$OUT_DIR"

LEDGER="$OUT_DIR/experiment_ledger.jsonl"
GIT_REV="$(git rev-parse --short HEAD 2>/dev/null || echo no-git-commit)"

BTC_2D="data/processed/maker_horizon_5000_latency_1000/BTCUSDT-2023-05-16_2023-05-17-combined-features.csv"
ETH_2D="data/processed/eth_maker_horizon_5000_latency_1000/ETHUSDT-2023-05-16_2023-05-17-combined-features.csv"
BTC_MULTI="data/processed/multiday_20230516_20230519_btc_5s_latency_1000/BTCUSDT-2023-05-16_2023-05-19-combined-features.csv"
ETH_MULTI="data/processed/multiday_20230516_20230519_eth_5s_latency_1000/ETHUSDT-2023-05-16_2023-05-19-combined-features.csv"

record_experiment() {
  local experiment_id="$1"
  local hypothesis_id="$2"
  local artifact_path="$3"
  local data_path="$4"
  local candidate_count="$5"
  local notes="$6"
  local command="$7"
  python3 - "$LEDGER" "$experiment_id" "$hypothesis_id" "$artifact_path" "$data_path" "$candidate_count" "$GIT_REV" "$notes" "$command" <<'PY'
import sys
from lob_forge.alpha_factory import ExperimentRecord, append_experiment_record, utc_now_iso

append_experiment_record(
    sys.argv[1],
    ExperimentRecord(
        experiment_id=sys.argv[2],
        hypothesis_id=sys.argv[3],
        artifact_path=sys.argv[4],
        data_path=sys.argv[5],
        candidate_count=int(sys.argv[6]),
        git_rev=sys.argv[7],
        created_at_utc=utc_now_iso(),
        notes=sys.argv[8],
        command=sys.argv[9],
    ),
)
PY
}

run_fold_artifact() {
  local experiment_id="$1"
  local hypothesis_id="$2"
  local data_path="$3"
  local candidate_count="$4"
  local assumed_cost_bps="$5"
  local notes="$6"
  shift 6
  local artifact="$OUT_DIR/${experiment_id}.csv"
  "$@" > "$artifact"
  python3 -m lob_forge.cli audit-results "$artifact" \
    --assumed-cost-bps "$assumed_cost_bps" \
    --cost-safety-multiple 2 \
    > "$OUT_DIR/${experiment_id}_audit.csv"
  record_experiment "$experiment_id" "$hypothesis_id" "$artifact" "$data_path" "$candidate_count" "$notes" "$*"
}

run_artifact() {
  local experiment_id="$1"
  local hypothesis_id="$2"
  local data_path="$3"
  local candidate_count="$4"
  local notes="$5"
  shift 5
  local artifact="$OUT_DIR/${experiment_id}.csv"
  "$@" > "$artifact"
  record_experiment "$experiment_id" "$hypothesis_id" "$artifact" "$data_path" "$candidate_count" "$notes" "$*"
}

rm -f "$LEDGER"

run_fold_artifact btc_2d_threshold_zero_fee H_TOPBOOK_5S_TAKER_THRESHOLD "$BTC_2D" 131 0 \
  "two-day BTC threshold sanity check, zero explicit taker fee" \
  python3 -m lob_forge.cli walk-forward "$BTC_2D" \
    --train-size 2400 --validation-size 1200 --test-size 1200 --step-size 1200 \
    --taker-fee-bps 0 --sort-by validation_net_pnl

run_fold_artifact btc_2d_threshold_5bps H_TOPBOOK_5S_TAKER_THRESHOLD "$BTC_2D" 131 5 \
  "two-day BTC threshold negative-control, 5 bps taker fee" \
  python3 -m lob_forge.cli walk-forward "$BTC_2D" \
    --train-size 2400 --validation-size 1200 --test-size 1200 --step-size 1200 \
    --taker-fee-bps 5 --sort-by validation_net_pnl

run_fold_artifact eth_2d_threshold_zero_fee H_TOPBOOK_5S_TAKER_THRESHOLD "$ETH_2D" 131 0 \
  "two-day ETH threshold sanity check, zero explicit taker fee" \
  python3 -m lob_forge.cli walk-forward "$ETH_2D" \
    --train-size 2400 --validation-size 1200 --test-size 1200 --step-size 1200 \
    --taker-fee-bps 0 --sort-by validation_net_pnl

run_fold_artifact eth_2d_threshold_5bps H_TOPBOOK_5S_TAKER_THRESHOLD "$ETH_2D" 131 5 \
  "two-day ETH threshold negative-control, 5 bps taker fee" \
  python3 -m lob_forge.cli walk-forward "$ETH_2D" \
    --train-size 2400 --validation-size 1200 --test-size 1200 --step-size 1200 \
    --taker-fee-bps 5 --sort-by validation_net_pnl

run_fold_artifact btc_multiday_threshold_zero_fee H_TOPBOOK_5S_TAKER_THRESHOLD "$BTC_MULTI" 131 0 \
  "four-day BTC threshold robustness check, zero explicit taker fee" \
  python3 -m lob_forge.cli walk-forward "$BTC_MULTI" \
    --train-size 3600 --validation-size 1800 --test-size 1800 --step-size 1800 \
    --taker-fee-bps 0 --sort-by validation_net_pnl

run_fold_artifact eth_multiday_threshold_zero_fee H_TOPBOOK_5S_TAKER_THRESHOLD "$ETH_MULTI" 131 0 \
  "four-day ETH threshold robustness check, zero explicit taker fee" \
  python3 -m lob_forge.cli walk-forward "$ETH_MULTI" \
    --train-size 3600 --validation-size 1800 --test-size 1800 --step-size 1800 \
    --taker-fee-bps 0 --sort-by validation_net_pnl

run_fold_artifact btc_calendar_threshold_zero_fee H_TOPBOOK_5S_TAKER_THRESHOLD "$BTC_MULTI" 131 0 \
  "four-day BTC calendar-split threshold sanity check with 2 train days, 1 validation day, 1 test day" \
  python3 -m lob_forge.cli calendar-walk-forward "$BTC_MULTI" \
    --train-days 2 --validation-days 1 --test-days 1 \
    --taker-fee-bps 0 --sort-by validation_net_pnl

run_fold_artifact eth_calendar_threshold_zero_fee H_TOPBOOK_5S_TAKER_THRESHOLD "$ETH_MULTI" 131 0 \
  "four-day ETH calendar-split threshold sanity check with 2 train days, 1 validation day, 1 test day" \
  python3 -m lob_forge.cli calendar-walk-forward "$ETH_MULTI" \
    --train-days 2 --validation-days 1 --test-days 1 \
    --taker-fee-bps 0 --sort-by validation_net_pnl

run_fold_artifact btc_multiday_edge_zero_fee H_EXPECTED_EDGE_5S_TAKER_RIDGE "$BTC_MULTI" 14 0 \
  "four-day BTC expected-edge ridge check, zero explicit taker fee" \
  python3 -m lob_forge.cli edge-walk-forward "$BTC_MULTI" \
    --train-size 3600 --validation-size 1800 --test-size 1800 --step-size 1800 \
    --l2 1 --taker-fee-bps 0 --sort-by validation_net_pnl

run_fold_artifact eth_multiday_edge_zero_fee H_EXPECTED_EDGE_5S_TAKER_RIDGE "$ETH_MULTI" 14 0 \
  "four-day ETH expected-edge ridge check, zero explicit taker fee" \
  python3 -m lob_forge.cli edge-walk-forward "$ETH_MULTI" \
    --train-size 3600 --validation-size 1800 --test-size 1800 --step-size 1800 \
    --l2 1 --taker-fee-bps 0 --sort-by validation_net_pnl

run_artifact btc_multiday_passive_fill H_PASSIVE_FILL_DIAGNOSTIC "$BTC_MULTI" 1 \
  "four-day BTC conservative passive-entry fill diagnostic" \
  python3 -m lob_forge.cli fill-diagnostics "$BTC_MULTI" \
    --feature microprice_deviation --threshold 0.1 --by-source-date \
    --maker-fee-bps 0 --taker-fee-bps 0

run_artifact eth_multiday_passive_fill H_PASSIVE_FILL_DIAGNOSTIC "$ETH_MULTI" 1 \
  "four-day ETH conservative passive-entry fill diagnostic" \
  python3 -m lob_forge.cli fill-diagnostics "$ETH_MULTI" \
    --feature microprice_deviation --threshold 0.1 --by-source-date \
    --maker-fee-bps 0 --taker-fee-bps 0

run_artifact btc_multiday_capacity H_CAPACITY_DIAGNOSTIC "$BTC_MULTI" 1 \
  "four-day BTC conservative taker capacity diagnostic using 1 percent recent volume and 5 percent top-book caps" \
  python3 -m lob_forge.cli capacity "$BTC_MULTI" \
    --feature microprice_deviation --threshold 0.1 --by-source-date \
    --participation-rate 0.01 --top-book-fraction 0.05

run_artifact eth_multiday_capacity H_CAPACITY_DIAGNOSTIC "$ETH_MULTI" 1 \
  "four-day ETH conservative taker capacity diagnostic using 1 percent recent volume and 5 percent top-book caps" \
  python3 -m lob_forge.cli capacity "$ETH_MULTI" \
    --feature microprice_deviation --threshold 0.1 --by-source-date \
    --participation-rate 0.01 --top-book-fraction 0.05

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

printf 'wrote artifacts to %s\n' "$OUT_DIR"
