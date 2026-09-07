"""Comparable, timestamped taker-policy replay for performance experiments.

The scheduler observes fills causally and retries flattening residual inventory.
Reported PnL always comes from one authoritative persistent-account simulation.
This is a displayed-liquidity model: it does not infer trades or passive fills.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import heapq
import math
from pathlib import Path
from typing import Mapping, Sequence

from lob_forge.execution_sim import (
    MarketEvent, SignalEvent, StatefulExecutionConfig, StatefulSimulationResult,
    _EventLiquidity, _effective_order_constraints, _reduces_inventory,
    apply_order_constraints, simulate_stateful_execution,
)
from lob_forge.features import iter_quote_events


@dataclass(frozen=True)
class PolicyReplayResult:
    metrics: dict[str, float | int | str | None]
    signal_ledger: list[dict[str, float | int | str]]
    simulation: StatefulSimulationResult


def load_quote_tape(
    path: Path | str, *, start_ms: int, end_ms: int,
) -> list[MarketEvent]:
    """Read every raw quote in the inclusive interval, preserving update order."""
    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError("quote tape requires 0 <= start_ms < end_ms")
    events = []
    for quote in iter_quote_events(Path(path)):
        if quote.event_time < start_ms:
            continue
        if quote.event_time > end_ms:
            break
        events.append(MarketEvent(
            quote.event_time, quote.bid, quote.ask, quote.bid_qty, quote.ask_qty,
            trade_flow_kind="none",
        ))
    if not events:
        raise ValueError("requested raw quote tape is empty")
    return events


def evaluate_policy(
    rows: Sequence[Mapping[str, object]],
    gross_predictions: Sequence[Sequence[float]],
    tape: list[MarketEvent],
    *, threshold_bps: float, taker_fee_bps: float, latency_ms: int,
    target_notional: float = 100.0, initial_cash: float = 10000.0,
    horizon_ms: int = 5000, decision_stride_ms: int = 10000,
    slippage_bps: float = 0.0, policy_fee_bps: float | None = None,
    include_ledgers: bool = False,
) -> PolicyReplayResult:
    """Replay paired [long, short] gross-bps forecasts on one common account.

    Decisions occur on UTC-aligned stride boundaries, excluding the first minute
    of every UTC day. Each entry schedules a flat instruction H milliseconds
    later. Positive latency applies to both instructions. Residual positions
    suppress subsequent entries and receive flatten retries at eligible decisions.
    ``policy_fee_bps`` freezes the fee used for side/threshold selection while
    ``taker_fee_bps`` controls actual charged fees. ``include_ledgers`` retains the
    simulator's detailed orders, fills and equity rows when requested. Equity
    rows in flat intervals are compressed; every in-position quote is retained.
    Order constraints are the engine's fractional research defaults; this helper
    does not claim venue-specific lot/minimum-notional compliance.
    """
    if len(rows) != len(gross_predictions):
        raise ValueError("one pair of gross predictions is required per feature row")
    if not tape:
        raise ValueError("raw quote tape cannot be empty")
    if latency_ms <= 0 or horizon_ms <= 0 or decision_stride_ms <= horizon_ms + latency_ms:
        raise ValueError("positive latency/horizon and stride > horizon + latency are required")
    if math.isnan(threshold_bps) or threshold_bps < 0:
        raise ValueError("threshold must be non-negative or positive infinity for flat")
    if any(not math.isfinite(v) or v < 0 for v in (taker_fee_bps, slippage_bps)):
        raise ValueError("fees and slippage must be finite and non-negative")
    policy_fee_bps = taker_fee_bps if policy_fee_bps is None else policy_fee_bps
    if not math.isfinite(policy_fee_bps) or policy_fee_bps < 0:
        raise ValueError("policy fee must be finite and non-negative")
    if any(not math.isfinite(v) or v <= 0 for v in (target_notional, initial_cash)):
        raise ValueError("target notional and initial cash must be finite and positive")
    if any(tape[i].timestamp_ms > tape[i + 1].timestamp_ms for i in range(len(tape) - 1)):
        raise ValueError("raw quote tape must retain nondecreasing source timestamps")
    config = StatefulExecutionConfig(
        initial_cash=initial_cash, max_position_notional=2 * target_notional,
        max_leverage=1.0, taker_fee_bps=taker_fee_bps,
        slippage_bps=slippage_bps, latency_ms=latency_ms,
    )
    scheduler = _MarketScheduler(config)
    ledger: list[dict[str, float | int | str]] = []
    event_index = 0
    prior_decision = -1
    cost = policy_fee_bps + slippage_bps
    for row_number, (row, pair) in enumerate(zip(rows, gross_predictions)):
        raw_time = row.get("decision_time", row.get("event_time"))
        decision_time = _integer_timestamp(raw_time)
        if decision_time <= prior_decision:
            raise ValueError("feature decisions must be strictly increasing and unique")
        prior_decision = decision_time
        if len(pair) != 2 or any(not math.isfinite(float(value)) for value in pair):
            raise ValueError("gross predictions must be finite [long, short] pairs")
        long_gross, short_gross = (float(value) for value in pair)
        long_net = (1 - cost / 10000) * long_gross - 2 * cost
        short_net = (1 + cost / 10000) * short_gross - 2 * cost
        if not math.isfinite(long_net) or not math.isfinite(short_net):
            raise ValueError("predicted net edges overflowed")
        if not math.isinf(threshold_bps):
            while event_index < len(tape) and tape[event_index].timestamp_ms < decision_time:
                scheduler.observe(tape[event_index])
                event_index += 1
        if decision_time % decision_stride_ms:
            continue
        predicted_side = (
            1 if long_net > threshold_bps and long_net >= short_net
            else -1 if short_net > threshold_bps and short_net > long_net else 0
        )
        record: dict[str, float | int | str] = {
            "row": row_number, "decision_time": decision_time,
            "long_gross_bps": long_gross, "short_gross_bps": short_gross,
            "long_net_bps": long_net, "short_net_bps": short_net,
            "predicted_side": predicted_side, "scheduled_side": 0,
            "inventory_before_decision": scheduler.inventory,
            "action": "flat",
        }
        ledger.append(record)
        stale_feature = False
        if "quote_age_ms" in row:
            quote_age = _integer_timestamp(row["quote_age_ms"])
            record["quote_age_ms"] = quote_age
            stale_feature = quote_age > 1000
        if decision_time < tape[0].timestamp_ms or decision_time >= tape[-1].timestamp_ms:
            record["action"] = "outside_tape"
            continue
        if scheduler.pending:
            record["action"] = "pending_instruction"
            continue
        if abs(scheduler.inventory) > _FLAT_TOLERANCE:
            if decision_time + latency_ms < tape[-1].timestamp_ms:
                scheduler.schedule(SignalEvent(
                    decision_time, 0, 0, signal_id=f"policy-{row_number}-flatten-retry",
                ))
                record["action"] = "residual_flatten"
            else:
                record["action"] = "insufficient_exit_tape"
            continue
        if stale_feature:
            record["action"] = "stale_feature_quote"
            continue
        if decision_time % 86_400_000 < 60_000:
            record["action"] = "funding_exclusion"
            continue
        if decision_time + horizon_ms + latency_ms >= tape[-1].timestamp_ms:
            record["action"] = "insufficient_exit_tape"
            continue
        if predicted_side:
            signal_id = f"policy-{row_number}"
            scheduler.schedule(SignalEvent(
                decision_time, predicted_side, target_notional,
                signal_id=f"{signal_id}-entry", predicted_edge_bps=max(long_net, short_net),
            ))
            scheduler.schedule(SignalEvent(
                decision_time + horizon_ms, 0, 0, signal_id=f"{signal_id}-exit",
            ))
            record["scheduled_side"] = predicted_side
            record["action"] = "entry"
    if not math.isinf(threshold_bps):
        for event in tape[event_index:]:
            scheduler.observe(event)
    replay_tape = list(scheduler.retained_events)
    if not replay_tape or replay_tape[0] is not tape[0]:
        replay_tape.insert(0, tape[0])
    if replay_tape[-1] is not tape[-1]:
        replay_tape.append(tape[-1])
    simulation = simulate_stateful_execution(replay_tape, scheduler.signals, config=config)
    _check_scheduler_agreement(scheduler, simulation)
    metrics = _metrics(
        simulation, ledger, initial_cash=initial_cash, taker_fee_bps=taker_fee_bps,
        threshold_bps=threshold_bps, tape=tape,
    )
    metrics.update({"replay_quote_count": len(replay_tape), "policy_fee_bps": policy_fee_bps,
                    "taker_fee_bps": taker_fee_bps, "latency_ms": latency_ms})
    if any(isinstance(value, float) and not math.isfinite(value) for value in metrics.values()):
        raise ValueError("execution metrics contain a non-finite value")
    if not include_ledgers:
        simulation = replace(simulation, orders=[], fills=[], positions=[])
    return PolicyReplayResult(
        metrics, ledger, simulation,
    )


_FLAT_TOLERANCE = 1e-10


class _MarketScheduler:
    """Causal inventory-only scheduling; the shared simulator verifies every fill."""
    def __init__(self, config: StatefulExecutionConfig):
        self.config = config
        self.constraints = _effective_order_constraints(config)
        self.inventory = 0.0
        self.cash = config.initial_cash
        self.signals: list[SignalEvent] = []
        self.pending: list[tuple[int, int, SignalEvent]] = []
        self.fills: list[tuple[str, int, int, float, float]] = []
        self.retained_events: list[MarketEvent] = []

    def schedule(self, signal: SignalEvent) -> None:
        self.signals.append(signal)
        heapq.heappush(self.pending, (
            signal.decision_time_ms + self.config.latency_ms, len(self.signals), signal,
        ))

    def observe(self, event: MarketEvent) -> None:
        due = bool(self.pending and self.pending[0][0] <= event.timestamp_ms)
        if self.inventory != 0.0 or due:
            self.retained_events.append(event)
        if not due:
            return
        book = _EventLiquidity.from_event(event)
        while self.pending and self.pending[0][0] <= event.timestamp_ms:
            submit_time, _, signal = heapq.heappop(self.pending)
            if event.timestamp_ms > submit_time + self.config.max_order_age_ms:
                continue
            target = signal.target_side * signal.target_notional / event.mid
            delta = target - self.inventory
            side = 1 if delta > 0 else -1
            if abs(delta) <= 1e-12:
                continue
            check = apply_order_constraints(
                side=side, price=book.best_price(side), quantity=abs(delta), constraints=self.constraints,
            )
            if not check.accepted:
                continue
            projected = self.inventory + side * check.quantity
            equity = self.cash + self.inventory * event.mid
            reduces = _reduces_inventory(self.inventory, projected)
            if not reduces and (
                abs(projected) * event.mid > self.config.max_position_notional
                or equity <= 0 or abs(projected) * event.mid > self.config.max_leverage * equity
            ):
                continue
            preview = _EventLiquidity([level[:] for level in book.bid_levels], [level[:] for level in book.ask_levels])
            price, quantity, _ = preview.consume(side=side, quantity=check.quantity)
            future_inventory = self.inventory + side * quantity
            fee = quantity * price * (self.config.taker_fee_bps + self.config.slippage_bps) / 10000
            future_equity = equity - side * quantity * (price - event.mid) - fee
            if quantity > 0 and not _reduces_inventory(self.inventory, future_inventory) and (
                future_equity <= 0
                or abs(future_inventory) * event.mid > self.config.max_leverage * future_equity + 1e-9
            ):
                continue
            if quantity <= 0:
                continue
            book.consume(side=side, quantity=check.quantity)
            self.inventory = future_inventory
            self.cash -= side * quantity * price + fee
            self.fills.append((signal.signal_id, event.timestamp_ms, side, quantity, price))


def _check_scheduler_agreement(scheduler: _MarketScheduler, simulation: StatefulSimulationResult) -> None:
    order_signals = {order.order_id: order.signal_id for order in simulation.orders}
    observed = [(order_signals[fill.order_id], fill.fill_time_ms, fill.side, fill.quantity, fill.avg_price)
                for fill in simulation.fills]
    if len(observed) != len(scheduler.fills):
        raise RuntimeError("causal scheduler and execution engine disagree on fill count")
    for expected, actual in zip(scheduler.fills, observed):
        if expected[:3] != actual[:3] or any(
            not math.isclose(left, right, rel_tol=1e-11, abs_tol=1e-10)
            for left, right in zip(expected[3:], actual[3:])
        ):
            raise RuntimeError("causal scheduler and execution engine disagree on fills")
    if not math.isclose(scheduler.inventory, simulation.final_inventory, rel_tol=1e-11, abs_tol=1e-10):
        raise RuntimeError("causal scheduler and execution engine disagree on inventory")


def _metrics(
    simulation: StatefulSimulationResult, ledger: list[dict[str, float | int | str]], *,
    initial_cash: float, taker_fee_bps: float, threshold_bps: float, tape: list[MarketEvent],
) -> dict[str, float | int | str | None]:
    net_pnl = simulation.final_equity - initial_cash
    peak = initial_cash
    max_drawdown = 0.0
    max_drawdown_fraction = 0.0
    for row in simulation.positions:
        peak = max(peak, row.equity)
        drawdown = peak - row.equity
        max_drawdown = max(max_drawdown, drawdown)
        max_drawdown_fraction = max(max_drawdown_fraction, drawdown / peak)
    order_signals = {order.order_id: order.signal_id for order in simulation.orders}
    entry_fills = [fill for fill in simulation.fills if order_signals[fill.order_id].endswith("-entry")]
    entry_notional = sum(fill.quantity * fill.avg_price for fill in entry_fills)
    inventory = 0.0
    active = False
    completed = 0
    for fill in simulation.fills:
        if order_signals[fill.order_id].endswith("-entry"):
            active = True
        inventory += fill.side * fill.quantity
        if active and abs(inventory) <= _FLAT_TOLERANCE:
            completed += 1
            active = False
    return {
        "initial_cash": initial_cash, "final_equity": simulation.final_equity,
        "net_pnl": net_pnl, "net_return_bps": net_pnl / initial_cash * 10000,
        "realized_pnl": simulation.realized_pnl, "turnover": simulation.turnover,
        "fees_and_slippage_paid": sum(fill.fee for fill in simulation.fills),
        "break_even_taker_fee_bps": (taker_fee_bps + net_pnl / simulation.turnover * 10000
                                      if simulation.turnover else None),
        "max_drawdown_pnl": max_drawdown, "max_drawdown_bps": max_drawdown_fraction * 10000,
        "filled_entries": len({fill.order_id for fill in entry_fills}),
        "completed_round_trips": completed, "fill_count": len(simulation.fills),
        "partial_fill_count": sum(fill.partial for fill in simulation.fills),
        "scheduled_entries": sum(row["action"] == "entry" for row in ledger),
        "residual_flatten_retries": sum(row["action"] == "residual_flatten" for row in ledger),
        "entry_notional": entry_notional,
        "net_bps_per_filled_entry_notional": net_pnl / entry_notional * 10000 if entry_notional else None,
        "final_inventory": simulation.final_inventory,
        "final_inventory_mark_notional": abs(simulation.final_inventory) * tape[-1].mid,
        "threshold_bps": threshold_bps if math.isfinite(threshold_bps) else None,
        "policy": "always_flat" if math.isinf(threshold_bps) else "expected_payoff_threshold",
        "tape_start_ms": tape[0].timestamp_ms, "tape_end_ms": tape[-1].timestamp_ms,
        "raw_quote_count": len(tape), "execution_basis": "displayed_depth_taker_marked_equity",
        "order_constraints": "fractional_research_defaults",
    }


def _integer_timestamp(value: object) -> int:
    if value is None or isinstance(value, bool):
        raise ValueError("feature row requires an integer decision timestamp")
    try:
        numeric = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid decision timestamp") from exc
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise ValueError("feature row requires an integer decision timestamp")
    return int(numeric)
