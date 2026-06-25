from lob_forge.protocol import (
    assert_disjoint_partitions,
    assert_valid_selection_metric,
    construct_causal_samples,
    purged_walk_forward_indices,
)


def test_causal_samples_ignore_events_after_decision_time() -> None:
    base_events = [
        {"event_time": "1000", "microprice": "1.0", "bid": "99", "ask": "101"},
        {"event_time": "1100", "microprice": "2.0", "bid": "100", "ask": "102"},
        {"event_time": "1300", "microprice": "1000.0", "bid": "101", "ask": "103"},
        {"event_time": "1600", "microprice": "3.0", "bid": "102", "ask": "104"},
    ]
    mutated = [dict(row) for row in base_events]
    mutated[2]["microprice"] = "-9999.0"

    first = construct_causal_samples(
        base_events,
        decision_times=[1100],
        feature_columns=["microprice"],
        latency_ms=200,
        horizon_ms=300,
    )[0]
    second = construct_causal_samples(
        mutated,
        decision_times=[1100],
        feature_columns=["microprice"],
        latency_ms=200,
        horizon_ms=300,
    )[0]

    assert first.features == {"microprice": 2.0}
    assert second.features == first.features
    assert first.contract.entry_time == 1300
    assert first.contract.exit_time == 1600


def test_causal_samples_can_use_receive_timestamp() -> None:
    events = [
        {"event_time": "1000", "local_time": "1200", "feature": "1"},
        {"event_time": "1100", "local_time": "1500", "feature": "2"},
        {"event_time": "1300", "local_time": "1700", "feature": "3"},
    ]

    samples = construct_causal_samples(
        events,
        decision_times=[1300],
        feature_columns=["feature"],
        receive_timestamp_column="local_time",
        latency_ms=100,
        horizon_ms=200,
    )

    assert samples[0].features == {"feature": 1.0}
    assert samples[0].contract.entry_time == 1500
    assert samples[0].contract.exit_time == 1700


def test_purged_walk_forward_indices_apply_embargo_and_are_disjoint() -> None:
    splits = purged_walk_forward_indices(
        40,
        train_size=12,
        validation_size=8,
        test_size=6,
        sequence_length=4,
        prediction_horizon=3,
        latency_buffer=2,
        embargo=1,
    )

    assert splits
    assert splits[0].train_purged == 9
    assert splits[0].validation_purged == 8
    assert splits[0].train_indices == (0, 1, 2)
    assert splits[0].validation_indices == ()
    assert splits[0].test_indices == tuple(range(20, 26))
    assert_disjoint_partitions(splits)


def test_selection_metric_rejects_test_fields() -> None:
    assert_valid_selection_metric("validation_net_pnl")
    try:
        assert_valid_selection_metric("test_net_pnl")
    except ValueError as exc:
        assert "training/validation only" in str(exc)
    else:
        raise AssertionError("expected test metric to be rejected")
