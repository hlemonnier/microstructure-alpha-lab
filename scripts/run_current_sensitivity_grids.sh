#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONPATH

OUT_DIR="${OUT_DIR:-results/current}"
mkdir -p "$OUT_DIR"

BTC_MULTI="data/processed/multiday_20230516_20230519_btc_5s_latency_1000/BTCUSDT-2023-05-16_2023-05-19-combined-features.csv"
ETH_MULTI="data/processed/multiday_20230516_20230519_eth_5s_latency_1000/ETHUSDT-2023-05-16_2023-05-19-combined-features.csv"

python3 - "$OUT_DIR" "$BTC_MULTI" "$ETH_MULTI" <<'PY'
import sys
from pathlib import Path

from lob_forge.regime_splits import format_regime_splits, run_regime_splits
from lob_forge.sensitivity import format_sensitivity_points, run_latency_fee_grid

out_dir = Path(sys.argv[1])
datasets = {
    "btc": Path(sys.argv[2]),
    "eth": Path(sys.argv[3]),
}
latencies = [0, 250, 500, 1000, 2000]
fees = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]

for name, path in datasets.items():
    points = run_latency_fee_grid(
        path,
        latencies_ms=latencies,
        fees_bps=fees,
        feature="microprice_deviation",
        threshold=0.1,
        horizon_ms=5000,
    )
    (out_dir / f"{name}_latency_fee_grid.csv").write_text(format_sensitivity_points(points) + "\n")
    zero_fee_points = [point for point in points if point.fee_bps == 0.0]
    (out_dir / f"{name}_latency_grid.csv").write_text(format_sensitivity_points(zero_fee_points) + "\n")
    one_second_points = [point for point in points if point.latency_ms == 1000]
    (out_dir / f"{name}_fee_grid.csv").write_text(format_sensitivity_points(one_second_points) + "\n")
    regimes = run_regime_splits(path)
    (out_dir / f"{name}_regime_splits.csv").write_text(format_regime_splits(regimes) + "\n")

print(f"wrote current sensitivity grids to {out_dir}")
PY
