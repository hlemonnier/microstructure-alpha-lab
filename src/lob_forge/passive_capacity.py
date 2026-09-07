from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PassiveCapacityEstimate:
    price: float
    order_size: float
    fill_probability: float
    expected_fill_size: float
    expected_notional: float
    capacity_ok: bool
    estimation_basis: str = "conditional_queue_scenario"
    observations: int = 0
    expected_fill_lower: float | None = None
    expected_fill_upper: float | None = None


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
    if not all(
        math.isfinite(value)
        for value in (
            price,
            displayed_size,
            queue_ahead_size,
            cancellation_ahead_size,
            trade_through_size,
            fill_probability,
            participation_cap,
        )
    ):
        raise ValueError("passive capacity inputs must be finite")
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


def estimate_passive_capacity_from_scenarios(
    *,
    price: float,
    order_size: float,
    scenarios: list[tuple[float, float, float]],
    confidence: float = 0.95,
) -> PassiveCapacityEstimate:
    """Mean executable size from joint (queue, cancellations, trade-through) draws.

    No separate fill probability multiplier: zero-fill scenarios already enter
    the expectation. Bounds are Hoeffding bounds for independent bounded draws.
    """
    if (
        not all(math.isfinite(v) for v in (price, order_size, confidence))
        or price <= 0
        or order_size < 0
        or not 0 < confidence < 1
        or not scenarios
    ):
        raise ValueError("require positive price, non-negative order size, scenarios and confidence in (0,1)")
    fills = []
    for queue, canceled, through in scenarios:
        if any(not math.isfinite(v) or v < 0 for v in (queue, canceled, through)):
            raise ValueError("scenario sizes must be finite and non-negative")
        fills.append(min(order_size, max(0.0, through - max(0.0, queue - canceled))))
    mean = sum(fills) / len(fills)
    radius = order_size * math.sqrt(math.log(2.0 / (1.0 - confidence)) / (2.0 * len(fills)))
    return PassiveCapacityEstimate(
        price,
        order_size,
        sum(v > 0 for v in fills) / len(fills),
        mean,
        price * mean,
        mean > 0,
        "joint_independent_queue_scenarios",
        len(fills),
        max(0.0, mean - radius),
        min(order_size, mean + radius),
    )


def format_passive_capacity(estimates: list[PassiveCapacityEstimate]) -> str:
    lines = [
        "price,order_size,fill_probability,expected_fill_size,expected_notional,capacity_ok,estimation_basis,observations,expected_fill_lower,expected_fill_upper"
    ]
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
                    estimate.estimation_basis,
                    str(estimate.observations),
                    _fmt(estimate.expected_fill_lower) if estimate.expected_fill_lower is not None else "",
                    _fmt(estimate.expected_fill_upper) if estimate.expected_fill_upper is not None else "",
                ]
            )
        )
    return "\n".join(lines)


def _fmt(value: float) -> str:
    return f"{value:.12g}"
