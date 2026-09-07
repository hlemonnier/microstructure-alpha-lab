from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CapacityPoint:
    notional: float
    gross_edge_bps: float
    net_edge_bps: float
    expected_pnl: float
    capacity_ok: bool
    estimation_basis: str = "assumed_piecewise_linear"
    observations: int = 0
    net_edge_standard_error_bps: float | None = None


def pnl_by_notional_curve(
    *,
    base_edge_bps: float,
    fee_bps: float,
    notionals: list[float],
    decay_start: float,
    zero_edge_notional: float,
) -> list[CapacityPoint]:
    if not all(math.isfinite(value) for value in [base_edge_bps, fee_bps, decay_start, zero_edge_notional, *notionals]):
        raise ValueError("capacity curve inputs must be finite")
    if fee_bps < 0:
        raise ValueError("fee_bps must be non-negative")
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
    if (
        not all(math.isfinite(value) for value in (base_edge_bps, notional, decay_start, zero_edge_notional))
        or notional < 0
        or decay_start <= 0
        or zero_edge_notional <= decay_start
    ):
        raise ValueError("require finite inputs, non-negative notional and zero_edge_notional > decay_start > 0")
    if notional <= decay_start:
        return base_edge_bps
    decay_fraction = min(1.0, (notional - decay_start) / (zero_edge_notional - decay_start))
    return base_edge_bps * (1.0 - decay_fraction)


def empirical_capacity_curve(observations: list[tuple[float, float]], *, fee_bps: float = 0.0) -> list[CapacityPoint]:
    """Summarize observed (executed notional, gross edge bps) without extrapolation.

    Standard errors assume independent observations at each size; they are not
    valid significance tests for overlapping or adaptively selected trades.
    """
    if not math.isfinite(fee_bps) or fee_bps < 0:
        raise ValueError("fee_bps must be finite and non-negative")
    groups: dict[float, list[float]] = {}
    for notional, edge in observations:
        if not math.isfinite(notional) or notional <= 0 or not math.isfinite(edge):
            raise ValueError("observations require positive finite executed notionals and finite edges")
        groups.setdefault(notional, []).append(edge)
    points = []
    for notional, edges in sorted(groups.items()):
        mean = sum(edges) / len(edges)
        se = (
            math.sqrt(sum((edge - mean) ** 2 for edge in edges) / (len(edges) - 1) / len(edges))
            if len(edges) > 1
            else None
        )
        net = mean - fee_bps
        points.append(
            CapacityPoint(notional, mean, net, notional * net / 10000, net > 0, "observed_size_group", len(edges), se)
        )
    return points


def format_capacity_curve(points: list[CapacityPoint]) -> str:
    lines = [
        "notional,gross_edge_bps,net_edge_bps,expected_pnl,capacity_ok,estimation_basis,observations,net_edge_standard_error_bps"
    ]
    for point in points:
        lines.append(
            ",".join(
                [
                    _fmt(point.notional),
                    _fmt(point.gross_edge_bps),
                    _fmt(point.net_edge_bps),
                    _fmt(point.expected_pnl),
                    str(int(point.capacity_ok)),
                    point.estimation_basis,
                    str(point.observations),
                    _fmt(point.net_edge_standard_error_bps) if point.net_edge_standard_error_bps is not None else "",
                ]
            )
        )
    return "\n".join(lines)


def _fmt(value: float) -> str:
    return f"{value:.12g}"
