from lob_forge.passive_capacity import estimate_passive_capacity, format_passive_capacity
from lob_forge.passive_capacity import estimate_passive_capacity_from_scenarios


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


def test_joint_scenarios_average_fills_without_probability_double_counting() -> None:
    result = estimate_passive_capacity_from_scenarios(price=100, order_size=2, scenarios=[(5, 0, 0), (5, 0, 10)] * 100)
    assert result.fill_probability == 0.5 and result.expected_fill_size == 1
    assert result.observations == 200
    assert 0 < result.expected_fill_lower < 1 < result.expected_fill_upper < 2
