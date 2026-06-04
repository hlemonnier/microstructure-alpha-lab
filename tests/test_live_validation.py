import csv
from pathlib import Path

from lob_forge.live_validation import (
    ShadowDecision,
    SimulatedFillPrediction,
    append_shadow_decision,
    format_shadow_fill_validation_report,
    market_events_from_feature_csv,
    merge_observed_fills_into_shadow_decisions,
    read_shadow_decisions,
    validate_shadow_fill_predictions,
    write_market_events_from_feature_csv,
    write_observed_fill_template,
    write_simulated_fill_predictions,
)


def test_shadow_decision_logger_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "shadow.csv"
    decision = ShadowDecision(
        decision_id="d1",
        timestamp_ms=1700000000000,
        venue="binance",
        symbol="BTCUSDT",
        model_name="ridge_expected_edge",
        predicted_side=1,
        predicted_edge_bps=0.8,
        order_type="paper_limit",
        intended_price=100.0,
        intended_size=0.01,
        observed_fill_price=100.1,
        observed_fill_size=0.01,
        realized_pnl=0.5,
    )

    append_shadow_decision(path, decision)
    loaded = read_shadow_decisions(path)

    assert loaded == [decision]


def test_shadow_fill_validation_passes_when_errors_are_inside_thresholds(tmp_path: Path) -> None:
    shadow_path = tmp_path / "shadow.csv"
    simulated_path = tmp_path / "simulated.csv"
    append_shadow_decision(
        shadow_path,
        ShadowDecision(
            decision_id="d1",
            timestamp_ms=1700000000000,
            venue="binance",
            symbol="BTCUSDT",
            model_name="ridge_expected_edge",
            predicted_side=1,
            predicted_edge_bps=0.8,
            order_type="paper_limit",
            intended_price=100.0,
            intended_size=1.0,
            observed_fill_price=100.1,
            observed_fill_size=0.9,
        ),
    )
    append_shadow_decision(
        shadow_path,
        ShadowDecision(
            decision_id="d2",
            timestamp_ms=1700000000100,
            venue="binance",
            symbol="BTCUSDT",
            model_name="ridge_expected_edge",
            predicted_side=1,
            predicted_edge_bps=0.2,
            order_type="paper_limit",
            intended_price=100.0,
            intended_size=1.0,
        ),
    )
    write_simulated_fill_predictions(
        simulated_path,
        [
            SimulatedFillPrediction("d1", 100.0, 1.0),
            SimulatedFillPrediction("d2", None, 0.0),
        ],
    )

    report = validate_shadow_fill_predictions(
        simulated_path=simulated_path,
        shadow_path=shadow_path,
        max_price_error=0.2,
        max_size_error=0.1,
        max_fill_rate_error=0.0,
    )
    text = format_shadow_fill_validation_report(report)
    csv_text = format_shadow_fill_validation_report(report, output_format="csv")

    assert report.passed
    assert report.matched_observations == 2
    assert report.validation.fill_rate_error == 0.0
    assert "passed=1" in text
    assert csv_text.splitlines()[1].endswith(",1")


def test_shadow_fill_validation_fails_on_missing_decisions(tmp_path: Path) -> None:
    shadow_path = tmp_path / "shadow.csv"
    simulated_path = tmp_path / "simulated.csv"
    append_shadow_decision(
        shadow_path,
        ShadowDecision(
            decision_id="live_only",
            timestamp_ms=1700000000000,
            venue="binance",
            symbol="BTCUSDT",
            model_name="ridge_expected_edge",
            predicted_side=1,
            predicted_edge_bps=0.8,
            order_type="paper_limit",
            intended_price=100.0,
            intended_size=1.0,
            observed_fill_price=100.1,
            observed_fill_size=0.9,
        ),
    )
    append_shadow_decision(
        shadow_path,
        ShadowDecision(
            decision_id="shared",
            timestamp_ms=1700000000100,
            venue="binance",
            symbol="BTCUSDT",
            model_name="ridge_expected_edge",
            predicted_side=-1,
            predicted_edge_bps=0.4,
            order_type="paper_limit",
            intended_price=101.0,
            intended_size=1.0,
            observed_fill_price=101.0,
            observed_fill_size=1.0,
        ),
    )
    write_simulated_fill_predictions(
        simulated_path,
        [
            SimulatedFillPrediction("shared", 101.0, 1.0),
            SimulatedFillPrediction("sim_only", 100.0, 1.0),
        ],
    )

    report = validate_shadow_fill_predictions(
        simulated_path=simulated_path,
        shadow_path=shadow_path,
        max_price_error=0.0,
        max_size_error=0.0,
        max_fill_rate_error=0.0,
    )
    text = format_shadow_fill_validation_report(report)

    assert not report.passed
    assert report.matched_observations == 1
    assert report.missing_simulated_decisions == ("live_only",)
    assert report.missing_shadow_decisions == ("sim_only",)
    assert "missing_simulated=live_only" in text
    assert "missing_shadow=sim_only" in text


def test_shadow_fill_validation_rejects_duplicate_prediction_ids(tmp_path: Path) -> None:
    shadow_path = tmp_path / "shadow.csv"
    simulated_path = tmp_path / "simulated.csv"
    append_shadow_decision(
        shadow_path,
        ShadowDecision(
            decision_id="d1",
            timestamp_ms=1700000000000,
            venue="binance",
            symbol="BTCUSDT",
            model_name="ridge_expected_edge",
            predicted_side=1,
            predicted_edge_bps=0.8,
            order_type="paper_limit",
            intended_price=100.0,
            intended_size=1.0,
            observed_fill_price=100.0,
            observed_fill_size=1.0,
        ),
    )
    write_simulated_fill_predictions(
        simulated_path,
        [
            SimulatedFillPrediction("d1", 100.0, 1.0),
            SimulatedFillPrediction("d1", 100.1, 1.0),
        ],
    )

    try:
        validate_shadow_fill_predictions(simulated_path=simulated_path, shadow_path=shadow_path)
    except ValueError as exc:
        assert "duplicate simulated decision_id values: d1" in str(exc)
    else:
        raise AssertionError("expected duplicate decision_id validation failure")


def test_observed_fill_import_merges_partial_fills_by_decision_id(tmp_path: Path) -> None:
    shadow_path = tmp_path / "shadow.csv"
    observed_path = tmp_path / "observed.csv"
    output_path = tmp_path / "shadow_with_observed.csv"
    append_shadow_decision(
        shadow_path,
        ShadowDecision(
            decision_id="d1",
            timestamp_ms=1700000000000,
            venue="bybit",
            symbol="BTCUSDT",
            model_name="ridge_expected_edge",
            predicted_side=1,
            predicted_edge_bps=0.8,
            order_type="paper_limit",
            intended_price=100.0,
            intended_size=2.0,
        ),
    )
    append_shadow_decision(
        shadow_path,
        ShadowDecision(
            decision_id="d2",
            timestamp_ms=1700000000100,
            venue="bybit",
            symbol="BTCUSDT",
            model_name="ridge_expected_edge",
            predicted_side=-1,
            predicted_edge_bps=0.4,
            order_type="paper_limit",
            intended_price=101.0,
            intended_size=1.0,
        ),
    )
    observed_path.write_text(
        "client_order_id,avgPrice,cumExecQty,realizedPnl\n"
        "d1,100.0,0.5,0.1\n"
        "d1,101.0,1.5,0.2\n"
        "d2,,0,\n"
        "not_shadow,99.0,1.0,0.0\n"
    )

    report = merge_observed_fills_into_shadow_decisions(
        shadow_path=shadow_path,
        observed_path=observed_path,
        output_path=output_path,
    )
    decisions = read_shadow_decisions(output_path)

    assert report.observed_rows == 4
    assert report.matched_decisions == 2
    assert report.unmatched_observed_ids == ("not_shadow",)
    assert decisions[0].observed_fill_size == 2.0
    assert abs((decisions[0].observed_fill_price or 0.0) - 100.75) < 1e-9
    assert abs((decisions[0].realized_pnl or 0.0) - 0.3) < 1e-9
    assert decisions[1].observed_fill_price is None
    assert decisions[1].observed_fill_size == 0.0


def test_observed_fill_template_does_not_count_as_observation_until_populated(tmp_path: Path) -> None:
    shadow_path = tmp_path / "shadow.csv"
    template_path = tmp_path / "observed_template.csv"
    output_path = tmp_path / "shadow_with_observed.csv"
    append_shadow_decision(
        shadow_path,
        ShadowDecision(
            decision_id="d1",
            timestamp_ms=1700000000000,
            venue="bybit",
            symbol="BTCUSDT",
            model_name="ridge_expected_edge",
            predicted_side=1,
            predicted_edge_bps=0.8,
            order_type="paper_limit",
            intended_price=100.0,
            intended_size=2.0,
            notes="shadow-only",
        ),
    )

    path = write_observed_fill_template(shadow_path=shadow_path, output_path=template_path, limit=1)
    rows = list(csv.DictReader(path.open()))
    report = merge_observed_fills_into_shadow_decisions(
        shadow_path=shadow_path,
        observed_path=template_path,
        output_path=output_path,
    )
    decisions = read_shadow_decisions(output_path)

    assert rows == [
        {
            "decision_id": "d1",
            "client_order_id": "d1",
            "venue": "bybit",
            "symbol": "BTCUSDT",
            "predicted_side": "1",
            "intended_price": "100",
            "intended_size": "2",
            "avgPrice": "",
            "cumExecQty": "",
            "realizedPnl": "",
            "notes": "shadow-only",
        }
    ]
    assert report.observed_rows == 0
    assert report.matched_decisions == 0
    assert decisions[0].observed_fill_price is None
    assert decisions[0].observed_fill_size is None


def test_observed_fill_template_rejects_negative_limit(tmp_path: Path) -> None:
    shadow_path = tmp_path / "shadow.csv"
    append_shadow_decision(
        shadow_path,
        ShadowDecision(
            decision_id="d1",
            timestamp_ms=1700000000000,
            venue="okx",
            symbol="BTC-USDT-SWAP",
            model_name="ridge_expected_edge",
            predicted_side=1,
            predicted_edge_bps=0.8,
            order_type="paper_limit",
            intended_price=100.0,
            intended_size=1.0,
        ),
    )

    try:
        write_observed_fill_template(shadow_path=shadow_path, output_path=tmp_path / "template.csv", limit=-1)
    except ValueError as exc:
        assert "limit must be non-negative" in str(exc)
    else:
        raise AssertionError("expected negative limit validation failure")


def test_observed_fill_import_rejects_positive_size_without_price(tmp_path: Path) -> None:
    shadow_path = tmp_path / "shadow.csv"
    observed_path = tmp_path / "observed.csv"
    output_path = tmp_path / "shadow_with_observed.csv"
    append_shadow_decision(
        shadow_path,
        ShadowDecision(
            decision_id="d1",
            timestamp_ms=1700000000000,
            venue="okx",
            symbol="BTC-USDT-SWAP",
            model_name="ridge_expected_edge",
            predicted_side=1,
            predicted_edge_bps=0.8,
            order_type="paper_limit",
            intended_price=100.0,
            intended_size=1.0,
        ),
    )
    observed_path.write_text("decision_id,fill_size\n" "d1,1.0\n")

    try:
        merge_observed_fills_into_shadow_decisions(
            shadow_path=shadow_path,
            observed_path=observed_path,
            output_path=output_path,
        )
    except ValueError as exc:
        assert "positive fill size but no fill price" in str(exc)
    else:
        raise AssertionError("expected observed fill import validation failure")


def test_feature_csv_exports_market_events_for_fill_simulation(tmp_path: Path) -> None:
    feature_path = tmp_path / "features.csv"
    output_path = tmp_path / "market_events.csv"
    feature_path.write_text(
        "event_time,entry_event_time,bid,ask,entry_bid,entry_ask,bid_qty,ask_qty,top_imbalance,realized_volatility_5,trade_count,trade_qty,trade_imbalance\n"
        "1000,1100,99,101,100,102,3,4,0.2,0.0003,2,5,0.6\n"
        "2000,2100,98,100,99,101,2,5,-0.4,0.0001,1,4,-0.5\n"
    )

    events = market_events_from_feature_csv(feature_path)
    write_market_events_from_feature_csv(feature_path, output_path)

    assert len(events) == 2
    assert events[0].timestamp_ms == 1100
    assert events[0].bid == 100.0
    assert events[0].ask == 102.0
    assert events[0].trade_side == "buy"
    assert abs(events[0].volatility_bps - 3.0) < 1e-9
    assert events[1].trade_side == "sell"
    assert output_path.read_text().splitlines()[0].startswith("timestamp_ms,bid,ask")
