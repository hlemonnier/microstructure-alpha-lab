from lob_forge.execution_sim import (
    FillObservation,
    LatencyAssumptions,
    MarketEvent,
    OrderConstraints,
    QueueAssumptions,
    RateLimiter,
    apply_order_constraints,
    calibrate_fill_probability,
    simulate_passive_limit_order,
    simulate_taker_latency_order,
    validate_simulated_vs_live_fills,
)


def test_order_constraints_round_and_reject_min_notional() -> None:
    constraints = OrderConstraints(tick_size=0.1, lot_size=0.01, min_quantity=0.01, min_notional=10.0)

    rejected = apply_order_constraints(side=1, price=100.04, quantity=0.05, constraints=constraints)
    accepted = apply_order_constraints(side=1, price=100.04, quantity=0.2, constraints=constraints)

    assert not rejected.accepted
    assert rejected.rejection_reason == "min_notional"
    assert accepted.price == 100.0
    assert accepted.quantity == 0.2


def test_taker_latency_uses_delayed_market_event() -> None:
    events = [
        MarketEvent(1000, bid=99.0, ask=100.0, bid_size=1, ask_size=1),
        MarketEvent(1500, bid=100.0, ask=101.0, bid_size=1, ask_size=1),
    ]
    constraints = OrderConstraints(tick_size=0.1, lot_size=0.01, min_quantity=0.01, min_notional=1.0)

    fill = simulate_taker_latency_order(
        events,
        decision_time_ms=1000,
        side=1,
        quantity=0.1,
        constraints=constraints,
        latency=LatencyAssumptions(order_latency_ms=250, websocket_delay_ms=100),
    )

    assert fill.accepted
    assert fill.fill_time_ms == 1500
    assert fill.fill_price == 101.0


def test_passive_queue_supports_partial_fill_and_maker_exit() -> None:
    events = [
        MarketEvent(1000, bid=100.0, ask=100.2, bid_size=2, ask_size=2, trade_side="sell", trade_size=0.5),
        MarketEvent(1100, bid=100.0, ask=100.2, bid_size=2, ask_size=2, trade_side="sell", trade_size=0.75),
        MarketEvent(1200, bid=100.1, ask=100.3, bid_size=2, ask_size=2),
    ]
    constraints = OrderConstraints(tick_size=0.1, lot_size=0.01, min_quantity=0.01, min_notional=1.0)

    fill = simulate_passive_limit_order(
        events,
        side=1,
        price=100.0,
        quantity=1.0,
        constraints=constraints,
        queue=QueueAssumptions(queue_ahead_size=0.5, cancellation_rate_per_second=0.0, max_wait_ms=1000),
    )

    assert fill.accepted
    assert fill.filled_size == 0.75
    assert fill.partial
    assert fill.maker_exit_price == 100.1
    assert fill.inventory_carry == 0.0


def test_passive_order_cancels_when_signal_decays() -> None:
    events = [
        MarketEvent(1000, bid=100.0, ask=100.2, bid_size=2, ask_size=2),
        MarketEvent(1100, bid=100.0, ask=100.2, bid_size=2, ask_size=2),
    ]
    constraints = OrderConstraints(tick_size=0.1, lot_size=0.01, min_quantity=0.01, min_notional=1.0)

    fill = simulate_passive_limit_order(
        events,
        side=1,
        price=100.0,
        quantity=1.0,
        constraints=constraints,
        queue=QueueAssumptions(queue_ahead_size=0.0, cancel_replace_edge_bps=0.5),
        signal_edges_bps=[1.0, 0.1],
    )

    assert fill.canceled
    assert fill.cancel_reason == "signal_decay_cancel_replace"


def test_rate_limiter_and_fill_calibration() -> None:
    limiter = RateLimiter(interval_ms=100)
    observations = [
        FillObservation(0.5, 0.5, 0.0, 5.0, True),
        FillObservation(0.8, 0.5, 0.1, 4.0, False),
        FillObservation(10.0, 7.0, 0.8, 20.0, True),
    ]

    buckets = calibrate_fill_probability(observations)

    assert limiter.allow(1000)
    assert not limiter.allow(1050)
    assert limiter.allow(1100)
    assert len(buckets) == 2


def test_validate_simulated_vs_live_fills() -> None:
    validation = validate_simulated_vs_live_fills(
        simulated=[(100.0, 1.0), (None, 0.0)],
        live=[(100.2, 0.8), (101.0, 1.0)],
    )

    assert validation.observations == 2
    assert validation.mean_abs_price_error > 0.0
    assert validation.mean_abs_size_error == 0.6
    assert validation.fill_rate_error == 0.5
