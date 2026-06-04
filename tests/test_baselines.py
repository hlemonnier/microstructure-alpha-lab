from lob_forge.baselines import (
    compute_metrics,
    evaluate_rule_file,
    evaluate_maker_entry_economics,
    evaluate_taker_economics,
    format_calendar_walk_forward_results,
    format_conditional_walk_forward_results,
    format_walk_forward_results,
    run_fee_sweep,
    run_calendar_walk_forward_thresholds,
    format_regime_analysis,
    run_regime_analysis,
    run_conditional_walk_forward_thresholds,
    run_threshold_baselines,
    run_walk_forward_thresholds,
)


def test_compute_metrics() -> None:
    metrics = compute_metrics([-1, 0, 1], [-1, 0, 0])
    assert metrics.n == 3
    assert metrics.accuracy == 2 / 3
    assert metrics.coverage == 1 / 3


def test_run_threshold_baselines(tmp_path) -> None:
    path = tmp_path / "features.csv"
    with path.open("w") as handle:
        handle.write(
            "label,microprice_deviation,top_imbalance,trade_imbalance,bid,ask,future_bid,future_ask\n"
        )
        for _ in range(10):
            handle.write("1,0.5,0.4,0.3,100.0,100.1,100.4,100.5\n")
            handle.write("0,0.0,0.0,0.0,100.0,100.1,100.0,100.1\n")
            handle.write("-1,-0.5,-0.4,-0.3,100.0,100.1,99.6,99.7\n")

    results = run_threshold_baselines(path)
    assert results
    assert results[0].validation.macro_f1 > 0.9
    assert results[0].test_economics.trades > 0


def test_run_threshold_baselines_can_sort_by_validation_net_pnl(tmp_path) -> None:
    path = tmp_path / "features.csv"
    with path.open("w") as handle:
        handle.write(
            "label,microprice_deviation,top_imbalance,trade_imbalance,bid,ask,future_bid,future_ask\n"
        )
        for _ in range(10):
            handle.write("1,0.5,0.4,0.3,100.0,100.1,100.4,100.5\n")
            handle.write("0,0.0,0.0,0.0,100.0,100.1,100.0,100.1\n")
            handle.write("-1,-0.5,-0.4,-0.3,100.0,100.1,99.6,99.7\n")

    results = run_threshold_baselines(path, sort_by="validation_net_pnl", taker_fee_bps=1000)

    assert results[0].name == "always_flat"
    assert results[0].validation_economics.net_pnl == 0


def test_taker_economics_reports_fee_turnover_and_break_even() -> None:
    rows = [
        {
            "label": "1",
            "bid": "100.0",
            "ask": "101.0",
            "future_bid": "103.0",
            "future_ask": "104.0",
        }
    ]

    economics = evaluate_taker_economics(
        rows,
        lambda row: 1,
        taker_fee_bps=10.0,
        slippage_bps=5.0,
    )

    assert economics.trades == 1
    assert economics.gross_pnl == 2.0
    assert economics.fee_turnover == 204.0
    assert round(economics.net_pnl, 6) == 1.694
    assert round(economics.break_even_taker_fee_bps, 6) == round(2.0 / 204.0 * 10_000.0 - 5.0, 6)
    assert economics.median_net_pnl_per_trade == economics.mean_net_pnl_per_trade
    assert economics.profit_factor == float("inf")
    assert economics.max_drawdown_pnl == 0.0
    assert economics.sharpe_per_trade == 0.0


def test_taker_economics_reports_drawdown_and_profit_factor() -> None:
    rows = [
        {
            "label": "1",
            "bid": "100.0",
            "ask": "100.0",
            "future_bid": "101.0",
            "future_ask": "101.0",
        },
        {
            "label": "1",
            "bid": "100.0",
            "ask": "100.0",
            "future_bid": "98.0",
            "future_ask": "98.0",
        },
        {
            "label": "1",
            "bid": "100.0",
            "ask": "100.0",
            "future_bid": "101.0",
            "future_ask": "101.0",
        },
    ]

    economics = evaluate_taker_economics(
        rows,
        lambda row: 1,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
    )

    assert economics.trades == 3
    assert economics.net_pnl == 0.0
    assert economics.median_net_pnl_per_trade == 1.0
    assert economics.median_net_return_bps_per_trade == 100.0
    assert economics.profit_factor == 1.0
    assert economics.max_drawdown_pnl == 2.0
    assert round(economics.sharpe_per_trade, 6) == 0.0


def test_maker_entry_economics_tracks_fills_and_exit_fee_break_even() -> None:
    rows = [
        {
            "label": "1",
            "entry_bid": "100.0",
            "entry_ask": "100.1",
            "future_bid": "101.0",
            "future_ask": "101.1",
            "maker_long_fillable": "1",
            "maker_short_fillable": "0",
        },
        {
            "label": "1",
            "entry_bid": "100.0",
            "entry_ask": "100.1",
            "future_bid": "101.0",
            "future_ask": "101.1",
            "maker_long_fillable": "0",
            "maker_short_fillable": "0",
        },
    ]

    economics = evaluate_maker_entry_economics(
        rows,
        lambda row: 1,
        maker_fee_bps=1.0,
        taker_fee_bps=2.0,
        slippage_bps=0.0,
    )

    assert economics.signals == 2
    assert economics.trades == 1
    assert economics.fill_rate == 0.5
    assert round(economics.gross_pnl, 6) == 1.0
    assert round(economics.net_pnl, 6) == round(1.0 - 0.01 - 0.0202, 6)
    assert round(economics.break_even_taker_fee_bps, 6) == round((1.0 - 0.01) / 101.0 * 10_000.0, 6)


def test_run_fee_sweep_selects_best_rule_per_fee(tmp_path) -> None:
    path = tmp_path / "features.csv"
    with path.open("w") as handle:
        handle.write(
            "label,microprice_deviation,top_imbalance,trade_imbalance,bid,ask,future_bid,future_ask\n"
        )
        for _ in range(10):
            handle.write("1,0.5,0.4,0.3,100.0,100.1,100.4,100.5\n")
            handle.write("0,0.0,0.0,0.0,100.0,100.1,100.0,100.1\n")
            handle.write("-1,-0.5,-0.4,-0.3,100.0,100.1,99.6,99.7\n")

    sweep = run_fee_sweep(path, fees_bps=[0.0, 1000.0])

    assert [item.fee_bps for item in sweep] == [0.0, 1000.0]
    assert sweep[0].result.validation_economics.net_pnl > 0
    assert sweep[1].result.name == "always_flat"


def test_run_regime_analysis_bins_rule_by_market_state(tmp_path) -> None:
    path = tmp_path / "features.csv"
    with path.open("w") as handle:
        handle.write(
            "label,microprice_deviation,spread,realized_volatility_5,depth_imbalance_1pct,"
            "notional_imbalance_1pct,bid,ask,future_bid,future_ask,maker_long_fillable,"
            "maker_short_fillable\n"
        )
        for idx in range(12):
            label = 1 if idx % 2 == 0 else -1
            feature = 0.5 if label == 1 else -0.5
            future_bid = 100.4 if label == 1 else 99.6
            future_ask = 100.5 if label == 1 else 99.7
            handle.write(
                f"{label},{feature},{0.01 + idx * 0.01},{idx * 0.001},0.1,0.2,"
                f"100.0,100.1,{future_bid},{future_ask},{int(idx % 3 == 0)},{int(idx % 4 == 0)}\n"
            )

    evaluations = run_regime_analysis(
        path,
        feature="microprice_deviation",
        threshold=0.1,
        regime_features=["spread"],
        bins=3,
        taker_fee_bps=0.0,
    )
    output = format_regime_analysis(evaluations)

    assert len(evaluations) == 3
    assert sum(evaluation.row_count for evaluation in evaluations) == 12
    assert evaluations[0].lower_bound < evaluations[-1].lower_bound
    assert evaluations[0].economics.trades == evaluations[0].economics.signals
    assert output.splitlines()[0].startswith("regime_feature,bucket")


def test_run_walk_forward_thresholds_purges_label_overlap(tmp_path) -> None:
    path = tmp_path / "features.csv"
    with path.open("w") as handle:
        handle.write(
            "event_time,future_event_time,label,microprice_deviation,top_imbalance,trade_imbalance,bid,ask,future_bid,future_ask\n"
        )
        for idx in range(18):
            event_time = idx * 1000
            future_event_time = event_time + 1500
            if idx % 3 == 0:
                label = 1
                feature = 0.5
                future_bid = 100.4
                future_ask = 100.5
            elif idx % 3 == 1:
                label = 0
                feature = 0.0
                future_bid = 100.0
                future_ask = 100.1
            else:
                label = -1
                feature = -0.5
                future_bid = 99.6
                future_ask = 99.7
            handle.write(
                f"{event_time},{future_event_time},{label},{feature},{feature},{feature},100.0,100.1,{future_bid},{future_ask}\n"
            )

    folds = run_walk_forward_thresholds(
        path,
        train_size=6,
        validation_size=4,
        test_size=4,
        step_size=4,
        taker_fee_bps=0.0,
        sort_by="validation_macro_f1",
    )

    assert len(folds) == 2
    assert folds[0].purged_train_rows == 1
    assert folds[0].purged_validation_rows == 1
    assert folds[0].validation_rows == 3
    assert folds[0].test_rows == 4
    assert folds[0].result.feature == "microprice_deviation"


def test_calendar_walk_forward_splits_by_source_date(tmp_path) -> None:
    path = tmp_path / "features.csv"
    with path.open("w") as handle:
        handle.write(
            "source_date,event_time,future_event_time,label,microprice_deviation,bid,ask,future_bid,future_ask\n"
        )
        for day_idx in range(4):
            source_date = f"2023-05-{16 + day_idx:02d}"
            for row_idx in range(4):
                idx = day_idx * 10 + row_idx
                event_time = idx * 1000
                label = 1 if row_idx % 2 == 0 else -1
                feature = 0.5 if label == 1 else -0.5
                future_bid = 100.4 if label == 1 else 99.6
                future_ask = 100.5 if label == 1 else 99.7
                handle.write(
                    f"{source_date},{event_time},{event_time + 500},{label},{feature},"
                    f"100.0,100.1,{future_bid},{future_ask}\n"
                )

    folds = run_calendar_walk_forward_thresholds(
        path,
        features=["microprice_deviation"],
        thresholds=[0.1],
        train_days=2,
        validation_days=1,
        test_days=1,
        purge_label_overlap=False,
        taker_fee_bps=0.0,
        sort_by="validation_net_pnl",
    )
    output = format_calendar_walk_forward_results(folds)

    assert len(folds) == 1
    assert folds[0].train_dates == ["2023-05-16", "2023-05-17"]
    assert folds[0].validation_dates == ["2023-05-18"]
    assert folds[0].test_dates == ["2023-05-19"]
    assert folds[0].train_rows == 8
    assert output.splitlines()[0].startswith("fold,train_dates,validation_dates,test_dates")


def test_conditional_walk_forward_can_select_regime_filter(tmp_path) -> None:
    path = tmp_path / "features.csv"
    with path.open("w") as handle:
        handle.write(
            "event_time,future_event_time,label,microprice_deviation,spread_mean_5,bid,ask,future_bid,future_ask\n"
        )
        for idx in range(30):
            event_time = idx * 1000
            in_good_regime = 10 <= idx < 15 or 20 <= idx < 25
            regime_value = 0.2 if in_good_regime else 0.1
            prediction_side = 1 if idx % 2 == 0 else -1
            label = prediction_side if in_good_regime else -prediction_side
            future_bid = 100.4 if label == 1 else 99.6
            future_ask = 100.5 if label == 1 else 99.7
            feature = 0.5 if prediction_side == 1 else -0.5
            handle.write(
                f"{event_time},{event_time + 500},{label},{feature},{regime_value},"
                f"100.0,100.1,{future_bid},{future_ask}\n"
            )

    folds = run_conditional_walk_forward_thresholds(
        path,
        features=["microprice_deviation"],
        thresholds=[0.1],
        regime_features=["spread_mean_5"],
        regime_bins=2,
        min_validation_trades=1,
        train_size=10,
        validation_size=10,
        test_size=10,
        step_size=10,
        purge_label_overlap=False,
        taker_fee_bps=0.0,
        sort_by="validation_net_pnl",
    )
    output = format_conditional_walk_forward_results(folds)

    assert len(folds) == 1
    assert folds[0].result.regime_feature == "spread_mean_5"
    assert folds[0].result.regime_bucket == 2
    assert folds[0].result.validation_economics.net_pnl > 0
    assert folds[0].result.test_economics.net_pnl > 0
    assert output.splitlines()[0].startswith("fold,train_rows")
    assert output.splitlines()[-1].startswith("summary,")


def test_format_walk_forward_results_includes_summary(tmp_path) -> None:
    path = tmp_path / "features.csv"
    with path.open("w") as handle:
        handle.write(
            "event_time,future_event_time,label,microprice_deviation,top_imbalance,trade_imbalance,bid,ask,future_bid,future_ask\n"
        )
        for idx in range(14):
            label = 1 if idx % 2 == 0 else -1
            feature = 0.5 if label == 1 else -0.5
            future_bid = 100.4 if label == 1 else 99.6
            future_ask = 100.5 if label == 1 else 99.7
            handle.write(
                f"{idx * 1000},{idx * 1000 + 500},{label},{feature},{feature},{feature},100.0,100.1,{future_bid},{future_ask}\n"
            )

    folds = run_walk_forward_thresholds(
        path,
        train_size=4,
        validation_size=4,
        test_size=4,
        step_size=4,
        purge_label_overlap=False,
        taker_fee_bps=0.0,
    )
    output = format_walk_forward_results(folds)
    lines = output.splitlines()

    assert lines[0].startswith("fold,train_rows")
    assert lines[-1].startswith("summary,")
    assert len(lines[0].split(",")) == len(lines[-1].split(","))


def test_evaluate_rule_file_source_date(tmp_path) -> None:
    path = tmp_path / "features.csv"
    with path.open("w") as handle:
        handle.write(
            "source_date,label,microprice_deviation,top_imbalance,trade_imbalance,bid,ask,future_bid,future_ask\n"
        )
        handle.write("2023-05-16,1,0.5,0.5,0.5,100.0,100.1,100.4,100.5\n")
        handle.write("2023-05-17,-1,-0.5,-0.5,-0.5,100.0,100.1,99.6,99.7\n")

    result = evaluate_rule_file(
        path,
        feature="top_imbalance",
        threshold=0.3,
        taker_fee_bps=0,
        source_date="2023-05-17",
    )

    assert result.metrics.n == 1
    assert result.metrics.accuracy == 1
    assert result.economics.trades == 1
