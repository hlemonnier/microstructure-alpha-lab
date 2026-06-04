from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PassiveCapacityEstimate:
    price: float
    order_size: float
    fill_probability: float
    expected_fill_size: float
    expected_notional: float
    capacity_ok: bool


def estimate_passive_capacity(
    *,
    price: float,
    displayed_size: float,
    queue_ahead_size: float,
    cancellation_ahead_size: float,
    trade_through_size: float,
    fill_probability: float,
    participation_cap: float = 0.1,
) -> PassiveCapacityEstimate:
    if price <= 0.0:
        raise ValueError("price must be positive")
    if displayed_size < 0.0 or queue_ahead_size < 0.0 or cancellation_ahead_size < 0.0 or trade_through_size < 0.0:
        raise ValueError("sizes must be non-negative")
    if not 0.0 <= fill_probability <= 1.0:
        raise ValueError("fill_probability must be between 0 and 1")
    if not 0.0 <= participation_cap <= 1.0:
        raise ValueError("participation_cap must be between 0 and 1")
    effective_queue_ahead = max(0.0, queue_ahead_size - cancellation_ahead_size)
    raw_fillable = max(0.0, trade_through_size - effective_queue_ahead)
    order_size = min(displayed_size * participation_cap, raw_fillable)
    expected_fill_size = order_size * fill_probability
    return PassiveCapacityEstimate(
        price=price,
        order_size=order_size,
        fill_probability=fill_probability,
        expected_fill_size=expected_fill_size,
        expected_notional=expected_fill_size * price,
        capacity_ok=expected_fill_size > 0.0,
    )


def format_passive_capacity(estimates: list[PassiveCapacityEstimate]) -> str:
    lines = ["price,order_size,fill_probability,expected_fill_size,expected_notional,capacity_ok"]
    for estimate in estimates:
        lines.append(
            ",".join(
                [
                    _fmt(estimate.price),
                    _fmt(estimate.order_size),
                    _fmt(estimate.fill_probability),
                    _fmt(estimate.expected_fill_size),
                    _fmt(estimate.expected_notional),
                    str(int(estimate.capacity_ok)),
                ]
            )
        )
    return "\n".join(lines)


def _fmt(value: float) -> str:
    return f"{value:.12g}"
