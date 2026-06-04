from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class RegimeSplit:
    regime_type: str
    regime: str
    rows: int
    trades: int
    mean_trade_notional: float
    mean_mid_return_5: float


def run_regime_splits(feature_csv: Path | str) -> list[RegimeSplit]:
    with Path(feature_csv).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("feature CSV has no rows")
    trade_values = [_float(row, "trade_notional") for row in rows]
    stress_cutoff = sorted(trade_values)[int(0.9 * (len(trade_values) - 1))]
    outputs: list[RegimeSplit] = []
    for regime_type in ["time_of_day", "weekday", "funding_window", "trend_chop", "stress_volume"]:
        buckets: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            regime = _regime(row, regime_type=regime_type, stress_cutoff=stress_cutoff)
            buckets.setdefault(regime, []).append(row)
        for regime, bucket_rows in sorted(buckets.items()):
            outputs.append(_summarize(regime_type, regime, bucket_rows))
    return outputs


def format_regime_splits(splits: list[RegimeSplit]) -> str:
    lines = ["regime_type,regime,rows,trades,mean_trade_notional,mean_mid_return_5"]
    for split in splits:
        lines.append(
            ",".join(
                [
                    split.regime_type,
                    split.regime,
                    str(split.rows),
                    str(split.trades),
                    _fmt(split.mean_trade_notional),
                    _fmt(split.mean_mid_return_5),
                ]
            )
        )
    return "\n".join(lines)


def _regime(row: dict[str, str], *, regime_type: str, stress_cutoff: float) -> str:
    timestamp_ms = int(float(row.get("event_time") or row.get("bucket_start_ms") or 0))
    dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
    if regime_type == "time_of_day":
        if 0 <= dt.hour < 8:
            return "asia"
        if 8 <= dt.hour < 16:
            return "europe"
        return "us"
    if regime_type == "weekday":
        return "weekend" if dt.weekday() >= 5 else "weekday"
    if regime_type == "funding_window":
        return "funding_hour" if dt.hour in {0, 8, 16} else "non_funding_hour"
    if regime_type == "trend_chop":
        return "trend" if abs(_float(row, "mid_return_5")) > 0.0005 else "chop"
    if regime_type == "stress_volume":
        return "stress_high_volume" if _float(row, "trade_notional") >= stress_cutoff else "normal_volume"
    raise ValueError(f"unknown regime_type {regime_type!r}")


def _summarize(regime_type: str, regime: str, rows: list[dict[str, str]]) -> RegimeSplit:
    trades = sum(int(float(row.get("trade_count") or 0)) for row in rows)
    return RegimeSplit(
        regime_type=regime_type,
        regime=regime,
        rows=len(rows),
        trades=trades,
        mean_trade_notional=sum(_float(row, "trade_notional") for row in rows) / len(rows),
        mean_mid_return_5=sum(_float(row, "mid_return_5") for row in rows) / len(rows),
    )


def _float(row: dict[str, str], column: str) -> float:
    return float(row.get(column) or 0.0)


def _fmt(value: float) -> str:
    return f"{value:.12g}"
