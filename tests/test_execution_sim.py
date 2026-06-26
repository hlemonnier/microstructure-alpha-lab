from lob_forge.execution_sim import (
    FillObservation,
    LatencyAssumptions,
    MarketEvent,
    OrderConstraints,
    QueueAssumptions,
    RateLimiter,
    SignalEvent,
    StatefulExecutionConfig,
    apply_order_constraints,
    calibrate_fill_probability,
    simulate_stateful_execution,
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


def test_stateful_execution_tracks_cash_inventory_and_latency() -> None:
    events = [
        MarketEvent(1000, bid=99.0, ask=101.0, bid_size=1.0, ask_size=0.5),
        MarketEvent(1200, bid=100.0, ask=102.0, bid_size=2.0, ask_size=2.0),
        MarketEvent(1400, bid=103.0, ask=105.0, bid_size=2.0, ask_size=2.0),
    ]
    signals = [
        SignalEvent(1000, target_side=1, target_notional=100.0, signal_id="long"),
        SignalEvent(1250, target_side=0, target_notional=0.0, signal_id="flat"),
    ]

    result = simulate_stateful_execution(
        events,
        signals,
        config=StatefulExecutionConfig(
            initial_cash=1000.0,
            max_position_notional=500.0,
            max_leverage=1.0,
            taker_fee_bps=0.0,
            latency_ms=150,
        ),
    )

    assert [fill.fill_time_ms for fill in result.fills] == [1200, 1400]
    assert result.orders[0].submit_time_ms == 1150
    assert result.final_inventory == 0.0
    assert result.final_equity > 1000.0
    assert result.realized_pnl > 0.0


def test_stateful_execution_never_fills_beyond_available_liquidity() -> None:
    events = [
        MarketEvent(1000, bid=99.0, ask=100.0, bid_size=1.0, ask_size=0.25),
        MarketEvent(1100, bid=99.5, ask=100.5, bid_size=1.0, ask_size=0.25),
    ]
    result = simulate_stateful_execution(
        events,
        [SignalEvent(1000, target_side=1, target_notional=100.0)],
        config=StatefulExecutionConfig(initial_cash=1000.0, max_position_notional=500.0, max_leverage=1.0),
    )

    assert result.fills[0].quantity == 0.25
    assert result.fills[0].partial
    assert result.fills[0].available_quantity == 0.25


def test_stateful_execution_walks_l2_levels_deterministically() -> None:
    events = [
        MarketEvent(
            1000,
            bid=99.0,
            ask=100.0,
            bid_size=1.0,
            ask_size=1.0,
            ask_levels=((100.0, 0.5), (101.0, 0.5)),
        ),
    ]
    signal = SignalEvent(1000, target_side=1, target_notional=100.5)
    config = StatefulExecutionConfig(initial_cash=1000.0, max_position_notional=500.0, max_leverage=1.0)

    first = simulate_stateful_execution(events, [signal], config=config)
    second = simulate_stateful_execution(events, [signal], config=config)

    assert first.fills[0].quantity == 1.0
    assert first.fills[0].avg_price == 100.5
    assert first == second


def test_stateful_execution_revisits_passive_order_until_fill() -> None:
    events = [
        MarketEvent(1000, bid=100.0, ask=100.2, bid_size=2.0, ask_size=2.0),
        MarketEvent(1100, bid=100.0, ask=100.2, bid_size=2.0, ask_size=2.0, trade_side="sell", trade_size=0.5),
        MarketEvent(1200, bid=100.0, ask=100.2, bid_size=2.0, ask_size=2.0, trade_side="sell", trade_size=2.0),
    ]
    signal = SignalEvent(
        1000,
        target_side=1,
        target_notional=100.0,
        order_type="passive",
        limit_price=100.0,
        predicted_edge_bps=1.0,
    )

    result = simulate_stateful_execution(
        events,
        [signal],
        config=StatefulExecutionConfig(
            initial_cash=1000.0,
            max_position_notional=500.0,
            max_leverage=1.0,
            max_order_age_ms=300,
            queue_ahead_size=0.5,
        ),
    )

    assert result.orders[0].status == "filled"
    assert result.fills[0].liquidity == "maker"
    assert result.fills[0].fill_time_ms == 1200
    assert result.fills[0].quantity > 0.9


def test_stateful_execution_does_not_apply_future_passive_fill_to_earlier_signal() -> None:
    events = [
        MarketEvent(1000, bid=100.0, ask=100.2, bid_size=5.0, ask_size=5.0),
        MarketEvent(1100, bid=100.1, ask=100.3, bid_size=5.0, ask_size=5.0),
        MarketEvent(1200, bid=100.0, ask=100.2, bid_size=5.0, ask_size=5.0, trade_side="sell", trade_size=2.0),
    ]
    signals = [
        SignalEvent(
            1000,
            target_side=1,
            target_notional=100.0,
            order_type="passive",
            limit_price=100.0,
            signal_id="passive-first",
            predicted_edge_bps=1.0,
        ),
        SignalEvent(1100, target_side=1, target_notional=200.0, signal_id="market-before-passive-fill"),
    ]

    result = simulate_stateful_execution(
        events,
        signals,
        config=StatefulExecutionConfig(
            initial_cash=1000.0,
            max_position_notional=500.0,
            max_leverage=1.0,
            max_order_age_ms=500,
        ),
    )

    assert [fill.fill_time_ms for fill in result.fills] == [1100, 1200]
    assert [fill.liquidity for fill in result.fills] == ["taker", "maker"]
    assert 0.9 < result.fills[0].quantity < 1.1
    assert 1.9 < sum(fill.quantity for fill in result.fills) < 2.1


def test_stateful_execution_consumes_displayed_liquidity_globally_for_same_timestamp_orders() -> None:
    events = [
        MarketEvent(1000, bid=99.0, ask=100.0, bid_size=5.0, ask_size=1.0),
    ]
    signals = [
        SignalEvent(1000, target_side=1, target_notional=100.0, signal_id="first"),
        SignalEvent(1000, target_side=1, target_notional=200.0, signal_id="second"),
    ]

    result = simulate_stateful_execution(
        events,
        signals,
        config=StatefulExecutionConfig(initial_cash=1000.0, max_position_notional=500.0, max_leverage=1.0),
    )

    assert len(result.fills) == 1
    assert result.fills[0].quantity == 1.0
    assert result.fills[0].available_quantity == 1.0
    assert result.orders[1].status == "open"
    assert result.orders[1].reason == "unfilled"


def test_stateful_execution_records_passive_position_at_fill_timestamp() -> None:
    events = [
        MarketEvent(1000, bid=100.0, ask=100.2, bid_size=5.0, ask_size=5.0),
        MarketEvent(1200, bid=100.0, ask=100.2, bid_size=5.0, ask_size=5.0, trade_side="sell", trade_size=2.0),
    ]

    result = simulate_stateful_execution(
        events,
        [
            SignalEvent(
                1000,
                target_side=1,
                target_notional=100.0,
                order_type="passive",
                limit_price=100.0,
                predicted_edge_bps=1.0,
            )
        ],
        config=StatefulExecutionConfig(
            initial_cash=1000.0,
            max_position_notional=500.0,
            max_leverage=1.0,
            max_order_age_ms=500,
        ),
    )

    first_inventory_position = next(position for position in result.positions if position.inventory > 0.0)
    assert result.fills[0].fill_time_ms == 1200
    assert first_inventory_position.timestamp_ms == 1200


def test_stateful_execution_flat_signal_cancels_pending_passive_before_same_time_trade() -> None:
    events = [
        MarketEvent(1000, bid=100.0, ask=100.2, bid_size=5.0, ask_size=5.0),
        MarketEvent(1100, bid=100.0, ask=100.2, bid_size=5.0, ask_size=5.0, trade_side="sell", trade_size=2.0),
    ]
    signals = [
        SignalEvent(
            1000,
            target_side=1,
            target_notional=100.0,
            order_type="passive",
            limit_price=100.0,
            predicted_edge_bps=1.0,
        ),
        SignalEvent(1100, target_side=0, target_notional=0.0),
    ]

    result = simulate_stateful_execution(
        events,
        signals,
        config=StatefulExecutionConfig(
            initial_cash=1000.0,
            max_position_notional=500.0,
            max_leverage=1.0,
            max_order_age_ms=500,
        ),
    )

    assert result.orders[0].status == "canceled"
    assert result.orders[0].reason == "replaced_by_later_signal"
    assert result.fills == []


def test_stateful_execution_expires_pending_passive_before_late_trade() -> None:
    events = [
        MarketEvent(1000, bid=100.0, ask=100.2, bid_size=5.0, ask_size=5.0),
        MarketEvent(1100, bid=100.0, ask=100.2, bid_size=5.0, ask_size=5.0, trade_side="sell", trade_size=2.0),
    ]

    result = simulate_stateful_execution(
        events,
        [
            SignalEvent(
                1000,
                target_side=1,
                target_notional=100.0,
                order_type="passive",
                limit_price=100.0,
                predicted_edge_bps=1.0,
            )
        ],
        config=StatefulExecutionConfig(
            initial_cash=1000.0,
            max_position_notional=500.0,
            max_leverage=1.0,
            max_order_age_ms=50,
        ),
    )

    assert result.orders[0].status == "expired"
    assert result.fills == []


def test_stateful_execution_rate_limiter_blocks_rapid_orders() -> None:
    events = [
        MarketEvent(1000, bid=99.0, ask=100.0, bid_size=5.0, ask_size=5.0),
        MarketEvent(1050, bid=99.0, ask=100.0, bid_size=5.0, ask_size=5.0),
    ]
    signals = [
        SignalEvent(1000, target_side=1, target_notional=100.0, signal_id="first"),
        SignalEvent(1050, target_side=0, target_notional=0.0, signal_id="second"),
    ]

    result = simulate_stateful_execution(
        events,
        signals,
        config=StatefulExecutionConfig(
            initial_cash=1000.0,
            max_position_notional=500.0,
            max_leverage=1.0,
            rate_limit_interval_ms=100,
        ),
    )

    assert result.orders[1].status == "rejected"
    assert result.orders[1].reason == "rate_limited"


def test_stateful_execution_kill_switch_blocks_later_orders() -> None:
    events = [
        MarketEvent(1000, bid=99.0, ask=100.0, bid_size=5.0, ask_size=5.0),
        MarketEvent(1100, bid=80.0, ask=81.0, bid_size=5.0, ask_size=5.0),
        MarketEvent(1200, bid=79.0, ask=80.0, bid_size=5.0, ask_size=5.0),
    ]
    signals = [
        SignalEvent(1000, target_side=1, target_notional=500.0),
        SignalEvent(1100, target_side=0, target_notional=0.0),
        SignalEvent(1200, target_side=-1, target_notional=500.0),
    ]

    result = simulate_stateful_execution(
        events,
        signals,
        config=StatefulExecutionConfig(
            initial_cash=1000.0,
            max_position_notional=1000.0,
            max_leverage=1.0,
            kill_switch_loss=50.0,
        ),
    )

    assert result.kill_switch_triggered
    assert result.orders[-1].status == "rejected"
    assert result.orders[-1].reason == "kill_switch"


def test_stateful_execution_cancels_pending_passive_before_later_signal() -> None:
    events = [
        MarketEvent(1000, bid=100.0, ask=100.2, bid_size=2.0, ask_size=2.0),
        MarketEvent(1100, bid=99.8, ask=100.0, bid_size=2.0, ask_size=2.0),
        MarketEvent(1200, bid=100.0, ask=100.2, bid_size=2.0, ask_size=2.0, trade_side="sell", trade_size=2.0),
    ]
    signals = [
        SignalEvent(
            1000,
            target_side=1,
            target_notional=100.0,
            order_type="passive",
            limit_price=100.0,
            signal_id="resting-bid",
        ),
        SignalEvent(1100, target_side=-1, target_notional=100.0, signal_id="new-short"),
    ]

    result = simulate_stateful_execution(
        events,
        signals,
        config=StatefulExecutionConfig(
            initial_cash=1000.0,
            max_position_notional=500.0,
            max_leverage=1.0,
            max_order_age_ms=500,
        ),
    )

    assert result.orders[0].status == "canceled"
    assert result.orders[0].reason == "replaced_by_later_signal"
    assert [fill.fill_time_ms for fill in result.fills] == sorted(fill.fill_time_ms for fill in result.fills)
    assert all(fill.order_id != result.orders[0].order_id for fill in result.fills)


def test_stateful_execution_sizes_same_side_signal_against_pending_passive_exposure() -> None:
    events = [
        MarketEvent(1000, bid=100.0, ask=100.2, bid_size=5.0, ask_size=5.0),
        MarketEvent(1100, bid=100.0, ask=100.2, bid_size=5.0, ask_size=5.0),
        MarketEvent(1200, bid=100.0, ask=100.2, bid_size=5.0, ask_size=5.0, trade_side="sell", trade_size=5.0),
    ]
    signals = [
        SignalEvent(
            1000,
            target_side=1,
            target_notional=100.0,
            order_type="passive",
            limit_price=100.0,
            signal_id="resting-bid",
            predicted_edge_bps=1.0,
        ),
        SignalEvent(1100, target_side=1, target_notional=200.0, signal_id="increase-long"),
    ]

    result = simulate_stateful_execution(
        events,
        signals,
        config=StatefulExecutionConfig(
            initial_cash=1000.0,
            max_position_notional=500.0,
            max_leverage=1.0,
            max_order_age_ms=500,
        ),
    )

    assert result.orders[0].status == "filled"
    assert result.orders[1].status == "filled"
    assert [fill.liquidity for fill in result.fills] == ["taker", "maker"]
    assert 1.9 < sum(fill.quantity for fill in result.fills) < 2.1
    assert all(fill.quantity < 1.1 for fill in result.fills)


def test_stateful_execution_cancels_same_side_pending_when_revised_target_would_overshoot() -> None:
    events = [
        MarketEvent(1000, bid=100.0, ask=100.2, bid_size=5.0, ask_size=5.0),
        MarketEvent(1100, bid=100.0, ask=100.2, bid_size=5.0, ask_size=5.0),
        MarketEvent(1200, bid=100.0, ask=100.2, bid_size=5.0, ask_size=5.0, trade_side="sell", trade_size=5.0),
    ]
    signals = [
        SignalEvent(
            1000,
            target_side=1,
            target_notional=200.0,
            order_type="passive",
            limit_price=100.0,
            signal_id="oversized-resting-bid",
            predicted_edge_bps=1.0,
        ),
        SignalEvent(1100, target_side=1, target_notional=100.0, signal_id="reduce-long"),
    ]

    result = simulate_stateful_execution(
        events,
        signals,
        config=StatefulExecutionConfig(
            initial_cash=1000.0,
            max_position_notional=500.0,
            max_leverage=1.0,
            max_order_age_ms=500,
        ),
    )

    assert result.orders[0].status == "canceled"
    assert result.orders[0].reason == "replaced_by_later_signal"
    assert len(result.fills) == 1
    assert result.fills[0].liquidity == "taker"
    assert 0.9 < result.fills[0].quantity < 1.1


def test_stateful_execution_consumes_same_timestamp_taker_liquidity_once() -> None:
    events = [
        MarketEvent(1000, bid=99.0, ask=100.0, bid_size=5.0, ask_size=1.0),
    ]
    signals = [
        SignalEvent(1000, target_side=1, target_notional=100.0, signal_id="first"),
        SignalEvent(1000, target_side=1, target_notional=200.0, signal_id="second"),
    ]

    result = simulate_stateful_execution(
        events,
        signals,
        config=StatefulExecutionConfig(initial_cash=1000.0, max_position_notional=500.0, max_leverage=1.0),
    )

    assert sum(fill.quantity for fill in result.fills if fill.side == 1) == 1.0
    assert result.fills[0].available_quantity == 1.0
    assert result.orders[1].status == "open"
    assert result.orders[1].reason == "unfilled"


def test_stateful_execution_consumes_passive_trade_flow_once_across_orders() -> None:
    events = [
        MarketEvent(1000, bid=99.9, ask=100.1, bid_size=5.0, ask_size=5.0),
        MarketEvent(1100, bid=99.9, ask=100.1, bid_size=5.0, ask_size=5.0, trade_side="sell", trade_size=1.0),
    ]
    signals = [
        SignalEvent(1000, target_side=1, target_notional=100.0, order_type="passive", limit_price=100.0),
        SignalEvent(1000, target_side=1, target_notional=200.0, order_type="passive", limit_price=100.0),
    ]

    result = simulate_stateful_execution(
        events,
        signals,
        config=StatefulExecutionConfig(
            initial_cash=1000.0,
            max_position_notional=500.0,
            max_leverage=1.0,
            max_order_age_ms=500,
        ),
    )

    assert sum(fill.quantity for fill in result.fills if fill.liquidity == "maker") == 1.0
    assert len([fill for fill in result.fills if fill.liquidity == "maker"]) == 1
