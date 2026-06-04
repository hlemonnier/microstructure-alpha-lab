from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapacityPoint:
    notional: float
    gross_edge_bps: float
    net_edge_bps: float
    expected_pnl: float
    capacity_ok: bool


def pnl_by_notional_curve(
    *,
    base_edge_bps: float,
    fee_bps: float,
    notionals: list[float],
    decay_start: float,
    zero_edge_notional: float,
) -> list[CapacityPoint]:
    if decay_start <= 0.0 or zero_edge_notional <= decay_start:
        raise ValueError("zero_edge_notional must be greater than decay_start")
    if any(notional < 0.0 for notional in notionals):
        raise ValueError("notionals must be non-negative")
    points: list[CapacityPoint] = []
    for notional in notionals:
        gross_edge = decayed_edge_bps(
            base_edge_bps,
            notional=notional,
            decay_start=decay_start,
            zero_edge_notional=zero_edge_notional,
        )
        net_edge = gross_edge - fee_bps
        points.append(
            CapacityPoint(
                notional=notional,
                gross_edge_bps=gross_edge,
                net_edge_bps=net_edge,
                expected_pnl=notional * net_edge / 10000.0,
                capacity_ok=net_edge > 0.0,
            )
        )
    return points


def decayed_edge_bps(
    base_edge_bps: float,
    *,
    notional: float,
    decay_start: float,
    zero_edge_notional: float,
) -> float:
    if notional <= decay_start:
        return base_edge_bps
    decay_fraction = min(1.0, (notional - decay_start) / (zero_edge_notional - decay_start))
    return base_edge_bps * (1.0 - decay_fraction)


def format_capacity_curve(points: list[CapacityPoint]) -> str:
    lines = ["notional,gross_edge_bps,net_edge_bps,expected_pnl,capacity_ok"]
    for point in points:
        lines.append(
            ",".join(
                [
                    _fmt(point.notional),
                    _fmt(point.gross_edge_bps),
                    _fmt(point.net_edge_bps),
                    _fmt(point.expected_pnl),
                    str(int(point.capacity_ok)),
                ]
            )
        )
    return "\n".join(lines)


def _fmt(value: float) -> str:
    return f"{value:.12g}"
