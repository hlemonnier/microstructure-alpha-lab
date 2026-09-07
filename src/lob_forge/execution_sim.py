from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_EVEN
import math


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
    bid_levels: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    ask_levels: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    trade_price: float | None = None
    trade_flow_kind: str = "quote_proxy"
    aggregate_buy_size: float = 0.0
    aggregate_sell_size: float = 0.0

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
    fill_time_ms: int | None = None


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
    fill_mismatch_rate: float = 0.0
    paired_fills: int = 0
    mean_relative_size_error: float = 0.0


@dataclass(frozen=True)
class SignalEvent:
    decision_time_ms: int
    target_side: int
    target_notional: float
    order_type: str = "market"
    limit_price: float | None = None
    signal_id: str = ""
    predicted_edge_bps: float = 0.0


@dataclass(frozen=True)
class StatefulExecutionConfig:
    initial_cash: float
    max_position_notional: float
    max_leverage: float
    taker_fee_bps: float = 0.0
    maker_fee_bps: float = 0.0
    slippage_bps: float = 0.0
    latency_ms: int = 0
    max_order_age_ms: int = 1000
    kill_switch_loss: float | None = None
    rate_limit_interval_ms: int = 0
    order_constraints: OrderConstraints | None = None
    queue_ahead_size: float = 0.0
    cancellation_rate_per_second: float = 0.0
    cancel_replace_edge_bps: float | None = None
    require_trade_price: bool = False


@dataclass(frozen=True)
class OrderLedgerRow:
    order_id: str
    signal_id: str
    decision_time_ms: int
    submit_time_ms: int
    side: int
    requested_quantity: float
    order_type: str
    limit_price: float | None
    status: str
    reason: str = ""


@dataclass(frozen=True)
class FillLedgerRow:
    order_id: str
    fill_time_ms: int
    side: int
    quantity: float
    avg_price: float
    fee: float
    liquidity: str
    available_quantity: float
    partial: bool


@dataclass(frozen=True)
class PositionLedgerRow:
    timestamp_ms: int
    cash: float
    inventory: float
    avg_entry_price: float
    mark_price: float
    equity: float
    realized_pnl: float
    turnover: float
    kill_switch_triggered: bool


@dataclass(frozen=True)
class StatefulSimulationResult:
    orders: list[OrderLedgerRow]
    fills: list[FillLedgerRow]
    positions: list[PositionLedgerRow]
    final_cash: float
    final_inventory: float
    final_equity: float
    realized_pnl: float
    turnover: float
    kill_switch_triggered: bool
    passive_fill_basis: str = "trade_price_or_quote_proxy"
    risk_stop_behavior: str = "cancel_pending_block_increases_allow_market_deleveraging"


@dataclass
class _PendingPassiveOrder:
    row_index: int
    order_id: str
    signal: SignalEvent
    side: int
    price: float
    requested_quantity: float
    remaining_quantity: float
    queue_ahead: float
    submit_time_ms: int
    activation_time_ms: int
    expiry_time_ms: int
    last_update_ms: int
    canceled: bool = False


@dataclass
class _EventLiquidity:
    bid_levels: list[list[float]]
    ask_levels: list[list[float]]

    @classmethod
    def from_event(cls, event: MarketEvent) -> _EventLiquidity:
        _validate_market_event(event)
        bid_levels = [[price, max(0.0, size)] for price, size in event.bid_levels]
        ask_levels = [[price, max(0.0, size)] for price, size in event.ask_levels]
        if not bid_levels:
            bid_levels = [[event.bid, max(0.0, event.bid_size)]]
        if not ask_levels:
            ask_levels = [[event.ask, max(0.0, event.ask_size)]]
        return cls(bid_levels=bid_levels, ask_levels=ask_levels)

    def best_price(self, side: int) -> float:
        levels = self.ask_levels if side == 1 else self.bid_levels
        return levels[0][0]

    def consume(self, *, side: int, quantity: float) -> tuple[float, float, float]:
        levels = self.ask_levels if side == 1 else self.bid_levels
        remaining = quantity
        filled = 0.0
        notional = 0.0
        available = sum(max(0.0, level[1]) for level in levels)
        for level in levels:
            level_size = max(0.0, level[1])
            take = min(remaining, level_size)
            if take <= 0.0:
                continue
            filled += take
            notional += take * level[0]
            remaining -= take
            level[1] = level_size - take
            if remaining <= 1e-12:
                break
        avg_price = notional / filled if filled else 0.0
        return avg_price, filled, available


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


def simulate_stateful_execution(
    events: list[MarketEvent],
    signals: list[SignalEvent],
    *,
    config: StatefulExecutionConfig,
) -> StatefulSimulationResult:
    if not all(
        math.isfinite(value)
        for value in (
            config.initial_cash,
            config.max_position_notional,
            config.max_leverage,
            config.taker_fee_bps,
            config.maker_fee_bps,
            config.slippage_bps,
            config.queue_ahead_size,
            config.cancellation_rate_per_second,
        )
    ):
        raise ValueError("execution config numeric values must be finite")
    if min(config.taker_fee_bps, config.slippage_bps, config.queue_ahead_size, config.cancellation_rate_per_second) < 0:
        raise ValueError("slippage, queue size and cancellation quantity rate must be non-negative")
    if config.initial_cash <= 0:
        raise ValueError("initial_cash must be positive")
    if config.max_position_notional < 0 or config.max_leverage <= 0:
        raise ValueError("position and leverage limits must be non-negative/positive")
    if config.latency_ms < 0 or config.max_order_age_ms < 0:
        raise ValueError("latency_ms and max_order_age_ms must be non-negative")
    if not events:
        raise ValueError("events cannot be empty")

    ordered_events = sorted(events, key=lambda event: event.timestamp_ms)
    ordered_signals: list[tuple[int | str, SignalEvent]] = list(
        sorted(enumerate(signals, start=1), key=lambda item: (item[1].decision_time_ms, item[0]))
    )
    for _, signal in ordered_signals:
        _validate_signal(signal)

    orders: list[OrderLedgerRow] = []
    fills: list[FillLedgerRow] = []
    positions: list[PositionLedgerRow] = []
    cash = config.initial_cash
    inventory = 0.0
    avg_entry_price = 0.0
    realized_pnl = 0.0
    turnover = 0.0
    kill_switch = False
    rate_limiter = RateLimiter(interval_ms=config.rate_limit_interval_ms)
    constraints = _effective_order_constraints(config)
    pending: list[_PendingPassiveOrder] = []
    next_signal_index = 0
    last_event = ordered_events[-1]

    for event in ordered_events:
        _validate_market_event(event)
        book = _EventLiquidity.from_event(event)
        for order in list(pending):
            if event.timestamp_ms > order.expiry_time_ms:
                _replace_order_status(orders, order.row_index, "expired", "max_order_age_ms")
                pending.remove(order)
        cash, inventory, avg_entry_price, realized_pnl, turnover, kill_switch = _refresh_kill_switch(
            event,
            cash,
            inventory,
            avg_entry_price,
            realized_pnl,
            turnover,
            kill_switch,
            config,
        )
        if kill_switch:
            _cancel_pending_orders(pending, orders, "kill_switch")

        while next_signal_index < len(ordered_signals):
            order_index, signal = ordered_signals[next_signal_index]
            submit_time = signal.decision_time_ms + config.latency_ms
            if submit_time > event.timestamp_ms:
                break
            next_signal_index += 1
            order_id = f"order-{order_index}"

            if event.timestamp_ms > submit_time + config.max_order_age_ms:
                orders.append(_order_row(order_id, signal, submit_time, 0, 0.0, "expired", "max_order_age_ms"))
                continue
            mark = event.mid
            target_inventory = 0.0 if signal.target_side == 0 else signal.target_side * signal.target_notional / mark
            risk_reducing = _reduces_inventory(inventory, target_inventory)
            if kill_switch and not (signal.order_type == "market" and risk_reducing):
                orders.append(_order_row(order_id, signal, submit_time, 0, 0.0, "rejected", "kill_switch"))
                continue

            if not rate_limiter.allow(submit_time):
                orders.append(_order_row(order_id, signal, submit_time, 0, 0.0, "rejected", "rate_limited"))
                continue
            if isinstance(order_index, int):
                # A new accepted parent target supersedes queued children of
                # earlier targets, including children waiting on a rate limit.
                ordered_signals[next_signal_index:] = [
                    item for item in ordered_signals[next_signal_index:] if isinstance(item[0], int)
                ]
            _reconcile_pending_orders_for_signal(
                pending,
                orders,
                signal=signal,
                inventory=inventory,
                target_inventory=target_inventory,
            )
            pending_inventory = _pending_inventory(pending)
            delta_qty = target_inventory - inventory - pending_inventory
            side = 1 if delta_qty > 0 else -1 if delta_qty < 0 else 0
            requested_qty = abs(delta_qty)
            if side == 0 or requested_qty <= 1e-12:
                orders.append(_order_row(order_id, signal, submit_time, 0, 0.0, "accepted", "already_at_target"))
                continue

            reference_price = (
                signal.limit_price
                if signal.limit_price is not None
                else book.best_price(-side if signal.order_type == "passive" else side)
            )
            # A target is a parent instruction. Venue per-order caps apply to
            # child orders, independently of the portfolio's position limit.
            unchunked_qty = requested_qty
            if constraints.max_notional is not None and signal.order_type == "market":
                levels = book.ask_levels if side == 1 else book.bid_levels
                worst_price = max((price for price, size in levels if size > 0.0), default=reference_price)
                requested_qty = min(requested_qty, constraints.max_notional / worst_price)
            check = apply_order_constraints(
                side=side,
                price=reference_price,
                quantity=requested_qty,
                constraints=constraints,
            )
            if not check.accepted:
                orders.append(
                    _order_row(order_id, signal, submit_time, side, check.quantity, "rejected", check.rejection_reason)
                )
                continue
            requested_qty = check.quantity

            if signal.order_type == "passive" and _post_only_would_cross(
                side=side,
                price=check.price,
                opposite_price=book.best_price(side),
            ):
                orders.append(
                    replace(
                        _order_row(
                            order_id,
                            signal,
                            submit_time,
                            side,
                            requested_qty,
                            "rejected",
                            "post_only_would_cross",
                        ),
                        limit_price=check.price,
                    )
                )
                continue

            projected_inventory = inventory + pending_inventory + side * requested_qty
            projected_notional = abs(projected_inventory) * mark
            current_equity = cash + inventory * mark
            reduces_risk = _reduces_inventory(inventory, projected_inventory)
            if projected_notional > config.max_position_notional and not reduces_risk:
                orders.append(
                    _order_row(order_id, signal, submit_time, side, requested_qty, "rejected", "max_position_notional")
                )
                continue
            if not reduces_risk and (current_equity <= 0 or projected_notional / current_equity > config.max_leverage):
                orders.append(
                    _order_row(order_id, signal, submit_time, side, requested_qty, "rejected", "max_leverage")
                )
                continue

            if signal.order_type == "market":
                preview = _EventLiquidity(
                    [level[:] for level in book.bid_levels], [level[:] for level in book.ask_levels]
                )
                preview_price, preview_qty, _ = preview.consume(side=side, quantity=requested_qty)
                future_inventory = inventory + side * preview_qty
                future_equity = current_equity - side * preview_qty * (preview_price - mark)
                future_equity -= preview_qty * preview_price * (config.taker_fee_bps + config.slippage_bps) / 10000.0
                future_equity -= _pending_cost_reserve(pending, mark, config.maker_fee_bps)
                if (
                    preview_qty > 0
                    and not _reduces_inventory(inventory, future_inventory)
                    and (
                        future_equity <= 0
                        or abs(future_inventory + pending_inventory) * mark > config.max_leverage * future_equity + 1e-9
                    )
                ):
                    orders.append(
                        _order_row(
                            order_id, signal, submit_time, side, requested_qty, "rejected", "max_leverage_after_costs"
                        )
                    )
                    continue
                avg_price, fill_qty, available_qty = book.consume(side=side, quantity=requested_qty)
                if fill_qty <= 0.0:
                    orders.append(
                        _order_row(
                            order_id, signal, submit_time, side, requested_qty, "canceled", "insufficient_liquidity"
                        )
                    )
                    continue
                partial_fill = fill_qty + 1e-12 < requested_qty
                cash, inventory, avg_entry_price, realized_pnl, turnover, fee = _apply_fill(
                    cash=cash,
                    inventory=inventory,
                    avg_entry_price=avg_entry_price,
                    realized_pnl=realized_pnl,
                    turnover=turnover,
                    side=side,
                    fill_qty=fill_qty,
                    avg_price=avg_price,
                    fee_bps=config.taker_fee_bps,
                    slippage_bps=config.slippage_bps,
                )
                orders.append(
                    _order_row(
                        order_id,
                        signal,
                        submit_time,
                        side,
                        requested_qty,
                        "partial" if partial_fill else "filled",
                        "insufficient_liquidity" if partial_fill else "",
                    )
                )
                fills.append(
                    FillLedgerRow(
                        order_id=order_id,
                        fill_time_ms=event.timestamp_ms,
                        side=side,
                        quantity=fill_qty,
                        avg_price=avg_price,
                        fee=fee,
                        liquidity="taker",
                        available_quantity=available_qty,
                        partial=partial_fill,
                    )
                )
                cash, inventory, avg_entry_price, realized_pnl, turnover, kill_switch = _refresh_kill_switch(
                    event,
                    cash,
                    inventory,
                    avg_entry_price,
                    realized_pnl,
                    turnover,
                    kill_switch,
                    config,
                )
                if kill_switch:
                    _cancel_pending_orders(pending, orders, "kill_switch")
                positions.append(
                    _position_row(event, cash, inventory, avg_entry_price, realized_pnl, turnover, kill_switch)
                )
                if requested_qty + 1e-12 < unchunked_qty and not partial_fill:
                    # Continue the parent target using another permitted child;
                    # preserve exchange rate limits and chronological decisions.
                    child_time = event.timestamp_ms + config.rate_limit_interval_ms
                    child = replace(signal, decision_time_ms=child_time - config.latency_ms)
                    remaining_signals = ordered_signals[next_signal_index:]
                    remaining_signals.append((f"{order_index}-child", child))
                    remaining_signals.sort(key=lambda item: item[1].decision_time_ms)
                    ordered_signals[next_signal_index:] = remaining_signals
                continue

            order_row = replace(
                _order_row(order_id, signal, submit_time, side, requested_qty, "open", ""),
                limit_price=check.price,
            )
            orders.append(order_row)
            pending.append(
                _PendingPassiveOrder(
                    row_index=len(orders) - 1,
                    order_id=order_id,
                    signal=signal,
                    side=side,
                    price=check.price,
                    requested_quantity=requested_qty,
                    remaining_quantity=requested_qty,
                    queue_ahead=max(0.0, config.queue_ahead_size),
                    submit_time_ms=submit_time,
                    activation_time_ms=event.timestamp_ms,
                    expiry_time_ms=submit_time + config.max_order_age_ms,
                    last_update_ms=event.timestamp_ms,
                )
            )

        remaining_trade_size = max(0.0, event.trade_size)
        for order in sorted(pending, key=lambda item: (item.side, -item.side * item.price, item.submit_time_ms)):
            if order.canceled:
                if order in pending:
                    pending.remove(order)
                continue
            if event.timestamp_ms > order.expiry_time_ms:
                _replace_order_status(orders, order.row_index, "expired", "max_order_age_ms")
                pending.remove(order)
                continue
            elapsed = max(0, event.timestamp_ms - order.last_update_ms)
            order.last_update_ms = event.timestamp_ms
            order.queue_ahead = max(0.0, order.queue_ahead - config.cancellation_rate_per_second * elapsed / 1000.0)
            if (
                config.cancel_replace_edge_bps is not None
                and abs(order.signal.predicted_edge_bps) < config.cancel_replace_edge_bps
            ):
                _replace_order_status(orders, order.row_index, "canceled", "signal_decay_cancel_replace")
                pending.remove(order)
                continue
            # A signal processed on this market event uses this event's book to
            # establish its resting price. It therefore cannot also consume
            # trade flow from the same event, even when its nominal network
            # submit time fell between two market events.
            if order.activation_time_ms >= event.timestamp_ms:
                continue
            if remaining_trade_size <= 0.0 or not _event_hits_passive_order(
                event, side=order.side, price=order.price, require_trade_price=config.require_trade_price
            ):
                continue

            queue_before = order.queue_ahead
            remaining_before = order.remaining_quantity
            projected_inventory = inventory + _pending_inventory(pending)
            projected_equity = (
                cash + inventory * event.mid - _pending_cost_reserve(pending, event.mid, config.maker_fee_bps)
            )
            if not _reduces_inventory(inventory, projected_inventory) and (
                abs(projected_inventory) * event.mid > config.max_position_notional + 1e-9
                or projected_equity <= 0
                or abs(projected_inventory) * event.mid > config.max_leverage * projected_equity + 1e-9
            ):
                _replace_order_status(orders, order.row_index, "canceled", "risk_limit_before_fill")
                pending.remove(order)
                continue
            order.queue_ahead, order.remaining_quantity, fill_qty = _consume_queue(
                order.queue_ahead,
                order.remaining_quantity,
                remaining_trade_size,
            )
            consumed_trade = min(
                remaining_trade_size, queue_before + remaining_before - order.queue_ahead - order.remaining_quantity
            )
            # External volume ahead at a common price is shared. Consume it
            # once, while own orders retain price/time priority via remaining
            # aggressive volume; otherwise every child gets a duplicate queue.
            external_consumed = queue_before - order.queue_ahead
            if external_consumed > 0:
                for later in pending:
                    if (
                        later is not order
                        and later.side == order.side
                        and later.price == order.price
                        and later.submit_time_ms >= order.submit_time_ms
                    ):
                        later.queue_ahead = max(0.0, later.queue_ahead - external_consumed)
            remaining_trade_size = max(0.0, remaining_trade_size - consumed_trade)
            if fill_qty <= 0.0:
                continue

            cash, inventory, avg_entry_price, realized_pnl, turnover, fee = _apply_fill(
                cash=cash,
                inventory=inventory,
                avg_entry_price=avg_entry_price,
                realized_pnl=realized_pnl,
                turnover=turnover,
                side=order.side,
                fill_qty=fill_qty,
                avg_price=order.price,
                fee_bps=config.maker_fee_bps,
                slippage_bps=0.0,
            )
            fills.append(
                FillLedgerRow(
                    order_id=order.order_id,
                    fill_time_ms=event.timestamp_ms,
                    side=order.side,
                    quantity=fill_qty,
                    avg_price=order.price,
                    fee=fee,
                    liquidity="maker",
                    available_quantity=consumed_trade,
                    partial=order.remaining_quantity > 1e-12,
                )
            )
            if order.remaining_quantity <= 1e-12:
                _replace_order_status(orders, order.row_index, "filled", "")
                pending.remove(order)
            else:
                _replace_order_status(orders, order.row_index, "partial", "")
            cash, inventory, avg_entry_price, realized_pnl, turnover, kill_switch = _refresh_kill_switch(
                event,
                cash,
                inventory,
                avg_entry_price,
                realized_pnl,
                turnover,
                kill_switch,
                config,
            )
            if kill_switch:
                _cancel_pending_orders(pending, orders, "kill_switch")
            positions.append(
                _position_row(event, cash, inventory, avg_entry_price, realized_pnl, turnover, kill_switch)
            )

        positions.append(_position_row(event, cash, inventory, avg_entry_price, realized_pnl, turnover, kill_switch))

    while next_signal_index < len(ordered_signals):
        order_index, signal = ordered_signals[next_signal_index]
        next_signal_index += 1
        submit_time = signal.decision_time_ms + config.latency_ms
        orders.append(_order_row(f"order-{order_index}", signal, submit_time, 0, 0.0, "rejected", "no_market_event"))
    _cancel_pending_orders(pending, orders, "no_market_event")

    final_equity = cash + inventory * last_event.mid
    return StatefulSimulationResult(
        orders=orders,
        fills=fills,
        positions=positions,
        final_cash=cash,
        final_inventory=inventory,
        final_equity=final_equity,
        realized_pnl=realized_pnl,
        turnover=turnover,
        kill_switch_triggered=kill_switch,
        passive_fill_basis="raw_trade_price_required" if config.require_trade_price else "trade_price_or_quote_proxy",
    )


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
    if min(constraints.min_quantity, constraints.min_notional) < 0 or (
        constraints.max_notional is not None
        and (not math.isfinite(constraints.max_notional) or constraints.max_notional <= 0)
    ):
        raise ValueError("minimum constraints must be non-negative and max_notional positive and finite")
    if (
        not all(
            math.isfinite(value)
            for value in (
                price,
                quantity,
                constraints.tick_size,
                constraints.lot_size,
                constraints.min_quantity,
                constraints.min_notional,
            )
        )
        or price <= 0
        or quantity < 0
    ):
        raise ValueError("prices, quantities and constraints must be finite and non-negative, with positive price")
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
    if min(latency.order_latency_ms, latency.websocket_delay_ms, latency.rate_limit_interval_ms) < 0:
        raise ValueError("latency intervals must be non-negative")
    events = sorted(events, key=lambda event: event.timestamp_ms)
    target_time = decision_time_ms + latency.order_latency_ms + latency.websocket_delay_ms
    event = _first_event_at_or_after(events, target_time)
    if event is None:
        return TakerFill(
            False, None, None, 0.0, latency.order_latency_ms + latency.websocket_delay_ms, "no_market_event"
        )
    price = event.ask if side == 1 else event.bid
    check = apply_order_constraints(side=side, price=price, quantity=quantity, constraints=constraints)
    if not check.accepted:
        return TakerFill(
            False,
            event.timestamp_ms,
            None,
            check.quantity,
            event.timestamp_ms - decision_time_ms,
            check.rejection_reason,
        )
    avg_price, filled, _ = _walk_book(event, side=side, quantity=check.quantity)
    return TakerFill(
        filled > 0.0,
        event.timestamp_ms,
        avg_price if filled else None,
        filled,
        event.timestamp_ms - decision_time_ms,
        "" if filled else "insufficient_liquidity",
    )


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
    require_trade_price: bool = False,
) -> PassiveFill:
    if (
        not math.isfinite(queue.queue_ahead_size)
        or not math.isfinite(queue.cancellation_rate_per_second)
        or min(queue.queue_ahead_size, queue.cancellation_rate_per_second, queue.max_wait_ms) < 0
    ):
        raise ValueError("queue assumptions must be finite and non-negative")
    if signal_edges_bps is not None and not signal_edges_bps:
        raise ValueError("signal_edges_bps must not be empty")
    if any(events[i].timestamp_ms > events[i + 1].timestamp_ms for i in range(len(events) - 1)):
        raise ValueError("passive replay events must be chronologically ordered")
    for event in events:
        _validate_market_event(event)
    check = apply_order_constraints(side=side, price=price, quantity=quantity, constraints=constraints)
    if not check.accepted:
        return PassiveFill(False, 0.0, None, False, True, "rejected", None, 0.0, 0, check.rejection_reason)
    if not events:
        return PassiveFill(True, 0.0, None, False, True, "no_market_event", None, 0.0, 0)
    if _post_only_would_cross(
        side=side,
        price=check.price,
        opposite_price=_EventLiquidity.from_event(events[0]).best_price(side),
    ):
        return PassiveFill(
            False,
            0.0,
            None,
            False,
            True,
            "rejected",
            None,
            0.0,
            0,
            "post_only_would_cross",
        )

    queue_ahead = max(0.0, queue.queue_ahead_size)
    remaining = check.quantity
    filled = 0.0
    notional = 0.0
    start_time = events[0].timestamp_ms
    previous_time = start_time
    canceled = False
    cancel_reason = ""
    events_seen = 0
    fill_time_ms: int | None = None

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
        # The first event is the placement snapshot, so its trade flow is not causally available.
        if event.timestamp_ms > start_time and _event_hits_passive_order(
            event, side=side, price=check.price, require_trade_price=require_trade_price
        ):
            queue_ahead, remaining, filled_now = _consume_queue(queue_ahead, remaining, event.trade_size)
            filled += filled_now
            notional += filled_now * check.price
            if filled_now > 0.0:
                fill_time_ms = event.timestamp_ms
            if remaining <= 0.0:
                break

    avg_price = notional / filled if filled else None
    maker_exit_price = None
    inventory_carry = filled
    # Only place an exit after the entry order has finished (filled, expired or
    # replay ended). Never retroactively close a partially working entry.
    if maker_exit and filled > 0.0 and events_seen < len(events):
        exit_events = events[events_seen:]
        exit_price = exit_events[0].ask if side == 1 else exit_events[0].bid
        exit_fill = simulate_passive_limit_order(
            exit_events,
            side=-side,
            price=exit_price,
            quantity=filled,
            constraints=constraints,
            queue=queue,
            maker_exit=False,
            require_trade_price=require_trade_price,
        )
        maker_exit_price = exit_fill.avg_fill_price
        inventory_carry = filled - exit_fill.filled_size
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
        fill_time_ms=fill_time_ms if filled > 0.0 else None,
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
    mismatches = 0
    relative_errors: list[float] = []
    for (sim_price, sim_size), (live_price, live_size) in zip(simulated, live):
        if any(price is not None and (not math.isfinite(price) or price <= 0) for price in (sim_price, live_price)):
            raise ValueError("fill prices must be finite and positive")
        if any(not math.isfinite(size) or size < 0 for size in (sim_size, live_size)):
            raise ValueError("fill sizes must be finite and non-negative")
        if (sim_price is None) != (sim_size == 0) or (live_price is None) != (live_size == 0):
            raise ValueError("fill price must be present exactly when size is positive")
        mismatches += (sim_size > 0) != (live_size > 0)
        relative_errors.append(
            abs(sim_size - live_size) / max(sim_size, live_size) if max(sim_size, live_size) else 0.0
        )
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
        fill_mismatch_rate=mismatches / len(simulated),
        paired_fills=len(price_errors),
        mean_relative_size_error=sum(relative_errors) / len(relative_errors),
    )


def _first_event_at_or_after(events: list[MarketEvent], timestamp_ms: int) -> MarketEvent | None:
    for event in events:
        if event.timestamp_ms >= timestamp_ms:
            return event
    return None


def _validate_signal(signal: SignalEvent) -> None:
    if not math.isfinite(signal.predicted_edge_bps):
        raise ValueError("predicted_edge_bps must be finite")
    if signal.target_side not in {-1, 0, 1}:
        raise ValueError("target_side must be -1, 0, or 1")
    if not math.isfinite(signal.target_notional) or signal.target_notional < 0:
        raise ValueError("target_notional must be non-negative")
    if signal.order_type not in {"market", "passive"}:
        raise ValueError("order_type must be market or passive")


def _effective_order_constraints(config: StatefulExecutionConfig) -> OrderConstraints:
    if config.order_constraints is not None:
        return config.order_constraints
    return OrderConstraints(
        tick_size=1e-12,
        lot_size=1e-12,
        min_quantity=0.0,
        min_notional=0.0,
        max_notional=None,
    )


def _replace_order_status(
    orders: list[OrderLedgerRow],
    row_index: int,
    status: str,
    reason: str,
) -> None:
    orders[row_index] = replace(orders[row_index], status=status, reason=reason)


def _cancel_pending_orders(
    pending: list[_PendingPassiveOrder],
    orders: list[OrderLedgerRow],
    reason: str,
) -> None:
    for order in pending:
        if not order.canceled:
            order.canceled = True
            _replace_order_status(orders, order.row_index, "canceled", reason)
    pending.clear()


def _reconcile_pending_orders_for_signal(
    pending: list[_PendingPassiveOrder],
    orders: list[OrderLedgerRow],
    *,
    signal: SignalEvent,
    inventory: float,
    target_inventory: float,
) -> None:
    if not pending:
        return
    pending_inventory = _pending_inventory(pending)
    if abs(pending_inventory) <= 1e-12:
        return
    target_side = 1 if target_inventory > 0.0 else -1 if target_inventory < 0.0 else 0
    pending_side = 1 if pending_inventory > 0.0 else -1
    projected_inventory = inventory + pending_inventory
    projected_overshoot = target_side != 0 and projected_inventory * target_side > abs(target_inventory) + 1e-12
    if target_side == 0 or pending_side != target_side or projected_overshoot:
        _cancel_pending_orders(pending, orders, "replaced_by_later_signal")


def _pending_inventory(pending: list[_PendingPassiveOrder]) -> float:
    return sum(order.side * order.remaining_quantity for order in pending if not order.canceled)


def _pending_cost_reserve(pending: list[_PendingPassiveOrder], mark: float, fee_bps: float) -> float:
    return sum(
        max(0.0, order.remaining_quantity * (order.side * (order.price - mark) + order.price * fee_bps / 10000.0))
        for order in pending
        if not order.canceled
    )


def _apply_fill(
    *,
    cash: float,
    inventory: float,
    avg_entry_price: float,
    realized_pnl: float,
    turnover: float,
    side: int,
    fill_qty: float,
    avg_price: float,
    fee_bps: float,
    slippage_bps: float,
) -> tuple[float, float, float, float, float, float]:
    notional = fill_qty * avg_price
    fee = notional * (fee_bps + slippage_bps) / 10_000.0
    realized_increment, next_avg_entry_price = _update_average_cost(
        inventory=inventory,
        avg_entry_price=avg_entry_price,
        side=side,
        fill_qty=fill_qty,
        fill_price=avg_price,
    )
    next_realized_pnl = realized_pnl + realized_increment - fee
    next_inventory = inventory + side * fill_qty
    next_cash = cash - side * notional - fee
    return next_cash, next_inventory, next_avg_entry_price, next_realized_pnl, turnover + notional, fee


def _refresh_kill_switch(
    event: MarketEvent,
    cash: float,
    inventory: float,
    avg_entry_price: float,
    realized_pnl: float,
    turnover: float,
    kill_switch: bool,
    config: StatefulExecutionConfig,
) -> tuple[float, float, float, float, float, bool]:
    if kill_switch or config.kill_switch_loss is None:
        return cash, inventory, avg_entry_price, realized_pnl, turnover, kill_switch
    equity = cash + inventory * event.mid
    if equity <= config.initial_cash - abs(config.kill_switch_loss):
        return cash, inventory, avg_entry_price, realized_pnl, turnover, True
    return cash, inventory, avg_entry_price, realized_pnl, turnover, False


def _order_row(
    order_id: str,
    signal: SignalEvent,
    submit_time_ms: int,
    side: int,
    requested_quantity: float,
    status: str,
    reason: str,
) -> OrderLedgerRow:
    return OrderLedgerRow(
        order_id=order_id,
        signal_id=signal.signal_id or order_id.replace("order", "signal"),
        decision_time_ms=signal.decision_time_ms,
        submit_time_ms=submit_time_ms,
        side=side,
        requested_quantity=requested_quantity,
        order_type=signal.order_type,
        limit_price=signal.limit_price,
        status=status,
        reason=reason,
    )


def _position_row(
    event: MarketEvent,
    cash: float,
    inventory: float,
    avg_entry_price: float,
    realized_pnl: float,
    turnover: float,
    kill_switch: bool,
) -> PositionLedgerRow:
    equity = cash + inventory * event.mid
    return PositionLedgerRow(
        timestamp_ms=event.timestamp_ms,
        cash=cash,
        inventory=inventory,
        avg_entry_price=avg_entry_price,
        mark_price=event.mid,
        equity=equity,
        realized_pnl=realized_pnl,
        turnover=turnover,
        kill_switch_triggered=kill_switch,
    )


def _walk_book(event: MarketEvent, *, side: int, quantity: float) -> tuple[float, float, float]:
    return _EventLiquidity.from_event(event).consume(side=side, quantity=quantity)


def _passive_available(event: MarketEvent, *, side: int, price: float, quantity: float) -> float:
    if side == 1:
        if event.trade_side != "sell" or event.bid > price:
            return 0.0
    else:
        if event.trade_side != "buy" or event.ask < price:
            return 0.0
    return min(quantity, max(0.0, event.trade_size))


def _update_average_cost(
    *,
    inventory: float,
    avg_entry_price: float,
    side: int,
    fill_qty: float,
    fill_price: float,
) -> tuple[float, float]:
    fill_inventory = side * fill_qty
    if abs(inventory) <= 1e-12:
        return 0.0, fill_price
    if inventory * fill_inventory > 0:
        new_abs = abs(inventory) + fill_qty
        new_avg = (abs(inventory) * avg_entry_price + fill_qty * fill_price) / new_abs
        return 0.0, new_avg

    closing_qty = min(abs(inventory), fill_qty)
    realized = closing_qty * (fill_price - avg_entry_price) * (1 if inventory > 0 else -1)
    remaining_inventory = inventory + fill_inventory
    if abs(remaining_inventory) <= 1e-12:
        return realized, 0.0
    if inventory * remaining_inventory < 0:
        return realized, fill_price
    return realized, avg_entry_price


def _event_hits_passive_order(
    event: MarketEvent, *, side: int, price: float, require_trade_price: bool = False
) -> bool:
    if event.trade_flow_kind == "aggregate" or (require_trade_price and event.trade_price is None):
        return False
    trade_price = event.trade_price
    if side == 1:
        return event.trade_side == "sell" and (trade_price if trade_price is not None else event.bid) <= price
    return event.trade_side == "buy" and (trade_price if trade_price is not None else event.ask) >= price


def _post_only_would_cross(*, side: int, price: float, opposite_price: float) -> bool:
    return price >= opposite_price if side == 1 else price <= opposite_price


def _consume_queue(queue_ahead: float, remaining: float, trade_size: float) -> tuple[float, float, float]:
    available = max(0.0, trade_size - queue_ahead)
    queue_ahead = max(0.0, queue_ahead - trade_size)
    fill = min(remaining, available)
    return queue_ahead, remaining - fill, fill


def _round_to_tick(price: float, tick_size: float) -> float:
    return float(
        (Decimal(str(price)) / Decimal(str(tick_size))).to_integral_value(rounding=ROUND_HALF_EVEN)
        * Decimal(str(tick_size))
    )


def _floor_to_lot(quantity: float, lot_size: float) -> float:
    return float(
        (Decimal(str(quantity)) / Decimal(str(lot_size))).to_integral_value(rounding=ROUND_FLOOR)
        * Decimal(str(lot_size))
    )


def _reduces_inventory(inventory: float, target_inventory: float) -> bool:
    return inventory != 0.0 and inventory * target_inventory >= 0.0 and abs(target_inventory) < abs(inventory)


def _validate_market_event(event: MarketEvent) -> None:
    if not all(
        math.isfinite(value) for value in (event.bid, event.ask, event.bid_size, event.ask_size, event.trade_size)
    ):
        raise ValueError("market prices and sizes must be finite")
    if event.bid <= 0 or event.ask < event.bid or min(event.bid_size, event.ask_size, event.trade_size) < 0:
        raise ValueError("market event requires positive uncrossed prices and non-negative sizes")
    if event.trade_price is not None and (not math.isfinite(event.trade_price) or event.trade_price <= 0):
        raise ValueError("trade_price must be positive and finite")
    for side, levels in ((-1, event.bid_levels), (1, event.ask_levels)):
        if any(not math.isfinite(price) or not math.isfinite(size) or price <= 0 or size < 0 for price, size in levels):
            raise ValueError("depth prices and sizes must be finite, with price>0 and size>=0")
        if any(side * levels[i][0] >= side * levels[i + 1][0] for i in range(len(levels) - 1)):
            raise ValueError("depth levels must be distinct and sorted from best to worst")


def _bucket(value: float, low: float, high: float) -> str:
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "mid"
