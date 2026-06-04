from __future__ import annotations

import csv
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SensitivityPoint:
    latency_ms: int
    fee_bps: float
    signals: int
    trades: int
    net_pnl: float
    mean_net_bps: float
    win_rate: float


def run_latency_fee_grid(
    feature_csv: Path | str,
    *,
    latencies_ms: list[int],
    fees_bps: list[float],
    feature: str = "microprice_deviation",
    threshold: float = 0.1,
    horizon_ms: int = 5000,
) -> list[SensitivityPoint]:
    rows = _read_rows(Path(feature_csv))
    times = [int(float(row["event_time"])) for row in rows]
    points: list[SensitivityPoint] = []
    for latency_ms in latencies_ms:
        for fee_bps in fees_bps:
            pnls: list[float] = []
            returns_bps: list[float] = []
            signals = 0
            for index, row in enumerate(rows):
                side = _signal(row, feature=feature, threshold=threshold)
                if side == 0:
                    continue
                signals += 1
                entry_index = bisect_left(times, times[index] + latency_ms)
                if entry_index >= len(rows):
                    continue
                exit_index = bisect_left(times, times[entry_index] + horizon_ms)
                if exit_index >= len(rows):
                    continue
                pnl, return_bps = _trade_pnl(rows[entry_index], rows[exit_index], side=side, fee_bps=fee_bps)
                pnls.append(pnl)
                returns_bps.append(return_bps)
            wins = sum(1 for pnl in pnls if pnl > 0.0)
            points.append(
                SensitivityPoint(
                    latency_ms=latency_ms,
                    fee_bps=fee_bps,
                    signals=signals,
                    trades=len(pnls),
                    net_pnl=sum(pnls),
                    mean_net_bps=sum(returns_bps) / len(returns_bps) if returns_bps else 0.0,
                    win_rate=wins / len(pnls) if pnls else 0.0,
                )
            )
    return points


def format_sensitivity_points(points: list[SensitivityPoint]) -> str:
    lines = ["latency_ms,fee_bps,signals,trades,net_pnl,mean_net_bps,win_rate"]
    for point in points:
        lines.append(
            ",".join(
                [
                    str(point.latency_ms),
                    _fmt(point.fee_bps),
                    str(point.signals),
                    str(point.trades),
                    _fmt(point.net_pnl),
                    _fmt(point.mean_net_bps),
                    _fmt(point.win_rate),
                ]
            )
        )
    return "\n".join(lines)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("feature CSV has no rows")
    return rows


def _signal(row: dict[str, str], *, feature: str, threshold: float) -> int:
    value = float(row.get(feature) or 0.0)
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def _trade_pnl(entry: dict[str, str], exit_row: dict[str, str], *, side: int, fee_bps: float) -> tuple[float, float]:
    if side == 1:
        entry_price = float(entry["ask"])
        exit_price = float(exit_row["bid"])
        gross_return = (exit_price - entry_price) / entry_price
    else:
        entry_price = float(entry["bid"])
        exit_price = float(exit_row["ask"])
        gross_return = (entry_price - exit_price) / entry_price
    net_return_bps = gross_return * 10000.0 - 2.0 * fee_bps
    return net_return_bps, net_return_bps


def _fmt(value: float) -> str:
    return f"{value:.12g}"
