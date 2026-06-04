from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from lob_forge.baselines import predict_feature_threshold


@dataclass(frozen=True)
class CapacityDiagnostics:
    group: str
    side: str
    signals: int
    positive_capacity_signals: int
    mean_capacity_notional: float
    median_capacity_notional: float
    min_capacity_notional: float
    max_capacity_notional: float
    total_capacity_notional: float
    mean_volume_cap_notional: float
    mean_top_book_cap_notional: float


def run_capacity_diagnostics(
    feature_csv: Path | str,
    *,
    feature: str,
    threshold: float,
    participation_rate: float = 0.01,
    top_book_fraction: float = 0.05,
    max_notional: float | None = None,
    source_date: str | None = None,
    by_source_date: bool = False,
) -> list[CapacityDiagnostics]:
    if participation_rate < 0.0:
        raise ValueError("participation_rate must be non-negative")
    if top_book_fraction < 0.0:
        raise ValueError("top_book_fraction must be non-negative")
    if max_notional is not None and max_notional < 0.0:
        raise ValueError("max_notional must be non-negative")

    rows = _read_rows(Path(feature_csv))
    if source_date is not None:
        rows = [row for row in rows if row.get("source_date") == source_date]
    if not rows:
        raise ValueError("no rows available for capacity diagnostics")
    if feature not in rows[0]:
        raise ValueError(f"feature column not found: {feature}")

    if by_source_date:
        dates = sorted({row.get("source_date", "unknown") or "unknown" for row in rows})
        groups = [
            (date, [row for row in rows if (row.get("source_date", "unknown") or "unknown") == date])
            for date in dates
        ]
        groups.append(("all", rows))
    else:
        groups = [("all", rows)]

    diagnostics: list[CapacityDiagnostics] = []
    for group_name, group_rows in groups:
        for side_name, side_filter in [("all", 0), ("long", 1), ("short", -1)]:
            diagnostics.append(
                _compute_group(
                    group_name,
                    side_name,
                    group_rows,
                    feature=feature,
                    threshold=threshold,
                    side_filter=side_filter,
                    participation_rate=participation_rate,
                    top_book_fraction=top_book_fraction,
                    max_notional=max_notional,
                )
            )
    return diagnostics


def format_capacity_diagnostics(diagnostics: list[CapacityDiagnostics]) -> str:
    lines = [
        "group,side,signals,positive_capacity_signals,mean_capacity_notional,median_capacity_notional,min_capacity_notional,max_capacity_notional,total_capacity_notional,mean_volume_cap_notional,mean_top_book_cap_notional"
    ]
    for item in diagnostics:
        lines.append(
            ",".join(
                [
                    item.group,
                    item.side,
                    str(item.signals),
                    str(item.positive_capacity_signals),
                    _fmt(item.mean_capacity_notional),
                    _fmt(item.median_capacity_notional),
                    _fmt(item.min_capacity_notional),
                    _fmt(item.max_capacity_notional),
                    _fmt(item.total_capacity_notional),
                    _fmt(item.mean_volume_cap_notional),
                    _fmt(item.mean_top_book_cap_notional),
                ]
            )
        )
    return "\n".join(lines)


def _compute_group(
    group: str,
    side_name: str,
    rows: list[dict[str, str]],
    *,
    feature: str,
    threshold: float,
    side_filter: int,
    participation_rate: float,
    top_book_fraction: float,
    max_notional: float | None,
) -> CapacityDiagnostics:
    capacities: list[float] = []
    volume_caps: list[float] = []
    top_book_caps: list[float] = []
    signals = 0

    for row in rows:
        side = predict_feature_threshold(row, feature, threshold)
        if side == 0:
            continue
        if side_filter and side != side_filter:
            continue

        signals += 1
        trade_notional = _safe_float(row.get("trade_notional"))
        volume_cap = participation_rate * max(0.0, trade_notional)
        top_book_cap = top_book_fraction * _top_book_notional(row, side)
        capacity = min(volume_cap, top_book_cap)
        if max_notional is not None:
            capacity = min(capacity, max_notional)

        capacities.append(max(0.0, capacity))
        volume_caps.append(max(0.0, volume_cap))
        top_book_caps.append(max(0.0, top_book_cap))

    return CapacityDiagnostics(
        group=group,
        side=side_name,
        signals=signals,
        positive_capacity_signals=sum(1 for value in capacities if value > 0.0),
        mean_capacity_notional=_mean(capacities),
        median_capacity_notional=_median(capacities),
        min_capacity_notional=min(capacities) if capacities else 0.0,
        max_capacity_notional=max(capacities) if capacities else 0.0,
        total_capacity_notional=sum(capacities),
        mean_volume_cap_notional=_mean(volume_caps),
        mean_top_book_cap_notional=_mean(top_book_caps),
    )


def _top_book_notional(row: dict[str, str], side: int) -> float:
    if side > 0:
        price = _safe_float(row.get("entry_ask") or row.get("ask"))
        qty = _safe_float(row.get("ask_qty"))
    else:
        price = _safe_float(row.get("entry_bid") or row.get("bid"))
        qty = _safe_float(row.get("bid_qty"))
    return max(0.0, price) * max(0.0, qty)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _fmt(value: float) -> str:
    return f"{value:.6f}"
