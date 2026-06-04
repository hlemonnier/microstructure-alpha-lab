from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketEvent:
    timestamp_ms: int
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    trade_side: str | None = None
    trade_size: float = 0.0
    top_imbalance: float = 0.0
    volatility_bps: float = 0.0
    trade_intensity: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_bps(self) -> float:
        return (self.ask - self.bid) / self.mid * 10000.0 if self.mid else 0.0


@dataclass(frozen=True)
class OrderConstraints:
    tick_size: float
    lot_size: float
    min_quantity: float
    min_notional: float
    max_notional: float | None = None


@dataclass(frozen=True)
class LatencyAssumptions:
    order_latency_ms: int = 0
    websocket_delay_ms: int = 0
    rate_limit_interval_ms: int = 0


@dataclass(frozen=True)
class QueueAssumptions:
    queue_ahead_size: float
    cancellation_rate_per_second: float = 0.0
    max_wait_ms: int = 1000
    cancel_replace_edge_bps: float | None = None


@dataclass(frozen=True)
class OrderCheck:
    accepted: bool
    price: float
    quantity: float
    notional: float
    rejection_reason: str = ""


@dataclass(frozen=True)
class TakerFill:
    accepted: bool
    fill_time_ms: int | None
    fill_price: float | None
    quantity: float
    latency_ms: int
    rejection_reason: str = ""


@dataclass(frozen=True)
class PassiveFill:
    accepted: bool
    filled_size: float
    avg_fill_price: float | None
    partial: bool
    canceled: bool
    cancel_reason: str
    maker_exit_price: float | None
    inventory_carry: float
    events_seen: int
    rejection_reason: str = ""


@dataclass(frozen=True)
class FillObservation:
    spread_bps: float
    volatility_bps: float
    top_imbalance: float
    trade_intensity: float
    filled: bool


@dataclass(frozen=True)
class FillProbabilityBucket:
    spread_bucket: str
    volatility_bucket: str
    imbalance_bucket: str
    intensity_bucket: str
    observations: int
    fill_probability: float


@dataclass(frozen=True)
class FillValidation:
    observations: int
    mean_abs_price_error: float
    mean_abs_size_error: float
    fill_rate_error: float


class RateLimiter:
    def __init__(self, *, interval_ms: int) -> None:
        if interval_ms < 0:
            raise ValueError("interval_ms must be non-negative")
        self.interval_ms = interval_ms
        self._last_accepted_ms: int | None = None

    def allow(self, timestamp_ms: int) -> bool:
        if self._last_accepted_ms is None or timestamp_ms >= self._last_accepted_ms + self.interval_ms:
            self._last_accepted_ms = timestamp_ms
            return True
        return False


def apply_order_constraints(
    *,
    side: int,
    price: float,
    quantity: float,
    constraints: OrderConstraints,
) -> OrderCheck:
    if side not in {-1, 1}:
        raise ValueError("side must be -1 or 1")
    if constraints.tick_size <= 0.0 or constraints.lot_size <= 0.0:
        raise ValueError("tick_size and lot_size must be positive")
    rounded_price = _round_to_tick(price, constraints.tick_size)
    rounded_qty = _floor_to_lot(quantity, constraints.lot_size)
    notional = rounded_price * rounded_qty
    if rounded_qty < constraints.min_quantity:
        return OrderCheck(False, rounded_price, rounded_qty, notional, "min_quantity")
    if notional < constraints.min_notional:
        return OrderCheck(False, rounded_price, rounded_qty, notional, "min_notional")
    if constraints.max_notional is not None and notional > constraints.max_notional:
        return OrderCheck(False, rounded_price, rounded_qty, notional, "max_notional")
    return OrderCheck(True, rounded_price, rounded_qty, notional)


def simulate_taker_latency_order(
    events: list[MarketEvent],
    *,
    decision_time_ms: int,
    side: int,
    quantity: float,
    constraints: OrderConstraints,
    latency: LatencyAssumptions,
) -> TakerFill:
    target_time = decision_time_ms + latency.order_latency_ms + latency.websocket_delay_ms
    event = _first_event_at_or_after(events, target_time)
    if event is None:
        return TakerFill(False, None, None, 0.0, latency.order_latency_ms + latency.websocket_delay_ms, "no_market_event")
    price = event.ask if side == 1 else event.bid
    check = apply_order_constraints(side=side, price=price, quantity=quantity, constraints=constraints)
    if not check.accepted:
        return TakerFill(False, event.timestamp_ms, None, check.quantity, event.timestamp_ms - decision_time_ms, check.rejection_reason)
    return TakerFill(True, event.timestamp_ms, check.price, check.quantity, event.timestamp_ms - decision_time_ms)


def simulate_passive_limit_order(
    events: list[MarketEvent],
    *,
    side: int,
    price: float,
    quantity: float,
    constraints: OrderConstraints,
    queue: QueueAssumptions,
    signal_edges_bps: list[float] | None = None,
    maker_exit: bool = True,
) -> PassiveFill:
    check = apply_order_constraints(side=side, price=price, quantity=quantity, constraints=constraints)
    if not check.accepted:
        return PassiveFill(False, 0.0, None, False, True, "rejected", None, 0.0, 0, check.rejection_reason)
    if not events:
        return PassiveFill(True, 0.0, None, False, True, "no_market_event", None, check.quantity, 0)

    queue_ahead = max(0.0, queue.queue_ahead_size)
    remaining = check.quantity
    filled = 0.0
    notional = 0.0
    start_time = events[0].timestamp_ms
    previous_time = start_time
    canceled = False
    cancel_reason = ""
    events_seen = 0
    last_fill_index: int | None = None

    for index, event in enumerate(events):
        events_seen += 1
        elapsed = max(0, event.timestamp_ms - previous_time)
        previous_time = event.timestamp_ms
        queue_ahead = max(0.0, queue_ahead - queue.cancellation_rate_per_second * elapsed / 1000.0)
        if event.timestamp_ms - start_time > queue.max_wait_ms:
            canceled = True
            cancel_reason = "max_wait_ms"
            break
        if signal_edges_bps is not None and queue.cancel_replace_edge_bps is not None:
            edge = signal_edges_bps[min(index, len(signal_edges_bps) - 1)]
            if abs(edge) < queue.cancel_replace_edge_bps:
                canceled = True
                cancel_reason = "signal_decay_cancel_replace"
                break
        if _event_hits_passive_order(event, side=side, price=check.price):
            queue_ahead, remaining, filled_now = _consume_queue(queue_ahead, remaining, event.trade_size)
            filled += filled_now
            notional += filled_now * check.price
            if filled_now > 0.0:
                last_fill_index = index
            if remaining <= 0.0:
                break

    avg_price = notional / filled if filled else None
    maker_exit_price = None
    inventory_carry = remaining
    if maker_exit and filled > 0.0 and last_fill_index is not None and last_fill_index + 1 < len(events):
        exit_event = events[last_fill_index + 1]
        maker_exit_price = exit_event.bid if side == 1 else exit_event.ask
        inventory_carry = 0.0
    return PassiveFill(
        accepted=True,
        filled_size=filled,
        avg_fill_price=avg_price,
        partial=0.0 < filled < check.quantity,
        canceled=canceled,
        cancel_reason=cancel_reason,
        maker_exit_price=maker_exit_price,
        inventory_carry=inventory_carry,
        events_seen=events_seen,
    )


def calibrate_fill_probability(observations: list[FillObservation]) -> list[FillProbabilityBucket]:
    buckets: dict[tuple[str, str, str, str], list[FillObservation]] = {}
    for observation in observations:
        key = (
            _bucket(observation.spread_bps, 1.0, 5.0),
            _bucket(observation.volatility_bps, 1.0, 5.0),
            _bucket(observation.top_imbalance, -0.25, 0.25),
            _bucket(observation.trade_intensity, 1.0, 10.0),
        )
        buckets.setdefault(key, []).append(observation)
    return [
        FillProbabilityBucket(
            spread_bucket=key[0],
            volatility_bucket=key[1],
            imbalance_bucket=key[2],
            intensity_bucket=key[3],
            observations=len(rows),
            fill_probability=sum(1.0 for row in rows if row.filled) / len(rows),
        )
        for key, rows in sorted(buckets.items())
    ]


def validate_simulated_vs_live_fills(
    simulated: list[tuple[float | None, float]],
    live: list[tuple[float | None, float]],
) -> FillValidation:
    if len(simulated) != len(live):
        raise ValueError("simulated and live fills must have the same length")
    if not simulated:
        raise ValueError("need at least one fill observation")
    price_errors: list[float] = []
    size_errors: list[float] = []
    sim_fills = 0
    live_fills = 0
    for (sim_price, sim_size), (live_price, live_size) in zip(simulated, live):
        if sim_price is not None:
            sim_fills += 1
        if live_price is not None:
            live_fills += 1
        if sim_price is not None and live_price is not None:
            price_errors.append(abs(sim_price - live_price))
        size_errors.append(abs(sim_size - live_size))
    return FillValidation(
        observations=len(simulated),
        mean_abs_price_error=sum(price_errors) / len(price_errors) if price_errors else 0.0,
        mean_abs_size_error=sum(size_errors) / len(size_errors),
        fill_rate_error=abs(sim_fills / len(simulated) - live_fills / len(live)),
    )


def _first_event_at_or_after(events: list[MarketEvent], timestamp_ms: int) -> MarketEvent | None:
    for event in events:
        if event.timestamp_ms >= timestamp_ms:
            return event
    return None


def _event_hits_passive_order(event: MarketEvent, *, side: int, price: float) -> bool:
    if side == 1:
        return event.trade_side == "sell" and event.bid <= price
    return event.trade_side == "buy" and event.ask >= price


def _consume_queue(queue_ahead: float, remaining: float, trade_size: float) -> tuple[float, float, float]:
    available = max(0.0, trade_size - queue_ahead)
    queue_ahead = max(0.0, queue_ahead - trade_size)
    fill = min(remaining, available)
    return queue_ahead, remaining - fill, fill


def _round_to_tick(price: float, tick_size: float) -> float:
    return round(round(price / tick_size) * tick_size, 12)


def _floor_to_lot(quantity: float, lot_size: float) -> float:
    return round(int(quantity / lot_size) * lot_size, 12)


def _bucket(value: float, low: float, high: float) -> str:
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "mid"
