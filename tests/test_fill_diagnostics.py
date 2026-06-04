from pathlib import Path

from lob_forge.fill_diagnostics import (
    format_fill_diagnostics,
    format_fill_regime_diagnostics,
    run_fill_diagnostics,
    run_fill_regime_diagnostics,
)


def test_fill_diagnostics_reports_side_fill_rates(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    _write_feature_csv(path)

    diagnostics = run_fill_diagnostics(
        path,
        feature="microprice_deviation",
        threshold=0.1,
        maker_fee_bps=0.0,
        taker_fee_bps=0.0,
    )
    output = format_fill_diagnostics(diagnostics)
    by_side = {item.side: item for item in diagnostics}

    assert by_side["all"].signals == 4
    assert by_side["all"].fills == 2
    assert by_side["all"].fill_rate == 0.5
    assert by_side["long"].signals == 2
    assert by_side["long"].fills == 1
    assert by_side["short"].signals == 2
    assert by_side["short"].fills == 1
    assert by_side["all"].mean_fill_latency_ms == 500.0
    assert output.splitlines()[0].startswith("group,side,signals")


def test_fill_diagnostics_can_group_by_source_date(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    _write_feature_csv(path)

    diagnostics = run_fill_diagnostics(
        path,
        feature="microprice_deviation",
        threshold=0.1,
        by_source_date=True,
    )

    groups = {item.group for item in diagnostics}
    assert groups == {"2023-05-16", "2023-05-17", "all"}
    assert len(diagnostics) == 9


def test_fill_regime_diagnostics_bins_market_state(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    _write_feature_csv(path)

    diagnostics = run_fill_regime_diagnostics(
        path,
        feature="microprice_deviation",
        threshold=0.1,
        regime_features=["spread_mean_5"],
        bins=2,
    )
    output = format_fill_regime_diagnostics(diagnostics)

    assert len(diagnostics) == 6
    assert diagnostics[0].regime_feature == "spread_mean_5"
    assert diagnostics[0].bucket == 1
    assert diagnostics[3].bucket == 2
    assert sum(item.diagnostics.signals for item in diagnostics if item.side == "all") == 4
    assert output.splitlines()[0].startswith("regime_feature,bucket")


def _write_feature_csv(path: Path) -> None:
    columns = [
        "source_date",
        "event_time",
        "entry_event_time",
        "label",
        "microprice_deviation",
        "spread_mean_5",
        "bid",
        "ask",
        "entry_bid",
        "entry_ask",
        "future_bid",
        "future_ask",
        "maker_long_fillable",
        "maker_short_fillable",
        "maker_long_fill_event_time",
        "maker_short_fill_event_time",
    ]
    rows = [
        {
            "source_date": "2023-05-16",
            "event_time": "0",
            "entry_event_time": "1000",
            "label": "1",
            "microprice_deviation": "0.5",
            "spread_mean_5": "0.1",
            "bid": "100.0",
            "ask": "100.1",
            "entry_bid": "100.0",
            "entry_ask": "100.1",
            "future_bid": "101.0",
            "future_ask": "101.1",
            "maker_long_fillable": "1",
            "maker_short_fillable": "0",
            "maker_long_fill_event_time": "1500",
            "maker_short_fill_event_time": "",
        },
        {
            "source_date": "2023-05-16",
            "event_time": "1000",
            "entry_event_time": "2000",
            "label": "1",
            "microprice_deviation": "0.5",
            "spread_mean_5": "0.2",
            "bid": "100.0",
            "ask": "100.1",
            "entry_bid": "100.0",
            "entry_ask": "100.1",
            "future_bid": "101.0",
            "future_ask": "101.1",
            "maker_long_fillable": "0",
            "maker_short_fillable": "0",
            "maker_long_fill_event_time": "",
            "maker_short_fill_event_time": "",
        },
        {
            "source_date": "2023-05-17",
            "event_time": "2000",
            "entry_event_time": "3000",
            "label": "-1",
            "microprice_deviation": "-0.5",
            "spread_mean_5": "0.3",
            "bid": "100.0",
            "ask": "100.1",
            "entry_bid": "100.0",
            "entry_ask": "100.1",
            "future_bid": "99.0",
            "future_ask": "99.1",
            "maker_long_fillable": "0",
            "maker_short_fillable": "1",
            "maker_long_fill_event_time": "",
            "maker_short_fill_event_time": "3500",
        },
        {
            "source_date": "2023-05-17",
            "event_time": "3000",
            "entry_event_time": "4000",
            "label": "-1",
            "microprice_deviation": "-0.5",
            "spread_mean_5": "0.4",
            "bid": "100.0",
            "ask": "100.1",
            "entry_bid": "100.0",
            "entry_ask": "100.1",
            "future_bid": "99.0",
            "future_ask": "99.1",
            "maker_long_fillable": "0",
            "maker_short_fillable": "0",
            "maker_long_fill_event_time": "",
            "maker_short_fill_event_time": "",
        },
    ]
    with path.open("w") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join(row[column] for column in columns) + "\n")
