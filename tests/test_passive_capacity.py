from lob_forge.passive_capacity import estimate_passive_capacity, format_passive_capacity


def test_passive_capacity_accounts_for_queue_and_cancellations() -> None:
    estimate = estimate_passive_capacity(
        price=100.0,
        displayed_size=10.0,
        queue_ahead_size=5.0,
        cancellation_ahead_size=2.0,
        trade_through_size=6.0,
        fill_probability=0.5,
        participation_cap=0.5,
    )

    assert estimate.order_size == 3.0
    assert estimate.expected_fill_size == 1.5
    assert estimate.expected_notional == 150.0
    assert estimate.capacity_ok
    assert format_passive_capacity([estimate]).startswith("price,order_size")
