import csv
from pathlib import Path

from lob_forge.portfolio import (
    PortfolioConfig,
    Trade,
    capped_expected_edge_notional,
    evaluate_oos_variance_stability,
    fractional_kelly_notional,
    format_variance_stability_report,
    gated_fractional_kelly_notional,
    simulate_fixed_notional_portfolio,
)


def test_fixed_notional_portfolio_reports_risk_metrics() -> None:
    trades = [
        Trade(0, 1000, 1, 100.0, 101.0, predicted_edge_bps=1.0),
        Trade(500, 1500, -1, 100.0, 99.0, predicted_edge_bps=1.0),
        Trade(2000, 3000, 1, 100.0, 99.0, predicted_edge_bps=1.0),
    ]
    config = PortfolioConfig(
        capital=10_000.0,
        base_notional=1_000.0,
        max_notional=1_000.0,
        taker_fee_bps=0.0,
        initial_margin_rate=0.1,
        max_inventory_notional=3_000.0,
        daily_loss_limit=20.0,
    )

    result = simulate_fixed_notional_portfolio(trades, config)

    assert result.total_net_pnl == 10.0
    assert result.max_concurrency == 2
    assert result.max_leverage == 0.2
    assert result.max_margin_used == 200.0
    assert result.daily_risk[0].trade_count == 3


def test_inventory_limit_rejects_excess_trade() -> None:
    trades = [
        Trade(0, 2000, -1, 100.0, 99.0, predicted_edge_bps=1.0),
        Trade(500, 2500, -1, 100.0, 99.0, predicted_edge_bps=1.0),
    ]
    config = PortfolioConfig(
        capital=10_000.0, base_notional=1_000.0, max_notional=1_000.0, max_inventory_notional=1_500.0
    )

    result = simulate_fixed_notional_portfolio(trades, config)

    assert result.trades[1].rejected
    assert result.trades[1].rejection_reason == "max_inventory_notional"


def test_portfolio_kill_switch_rejects_entries_after_loss_is_realized() -> None:
    trades = [
        Trade(0, 1000, 1, 100.0, 99.0, predicted_edge_bps=1.0),
        Trade(2000, 3000, 1, 100.0, 101.0, predicted_edge_bps=1.0),
    ]
    config = PortfolioConfig(
        capital=10_000.0,
        base_notional=1_000.0,
        max_notional=1_000.0,
        daily_loss_limit=5.0,
    )

    result = simulate_fixed_notional_portfolio(trades, config)

    assert result.kill_switch_triggered
    assert result.trades[1].rejected
    assert result.trades[1].rejection_reason == "kill_switch"


def test_portfolio_does_not_use_future_exit_pnl_for_kill_switch() -> None:
    trades = [
        Trade(0, 100, 1, 100.0, 99.0, predicted_edge_bps=1.0),
        Trade(50, 150, 1, 100.0, 101.0, predicted_edge_bps=1.0),
        Trade(100, 200, 1, 100.0, 101.0, predicted_edge_bps=1.0),
    ]
    config = PortfolioConfig(
        capital=10_000.0,
        base_notional=1_000.0,
        max_notional=1_000.0,
        daily_loss_limit=5.0,
    )

    result = simulate_fixed_notional_portfolio(trades, config)

    assert not result.trades[1].rejected
    assert result.trades[2].rejected
    assert result.trades[2].rejection_reason == "kill_switch"
    assert result.max_concurrency == 2
    assert result.total_net_pnl == 0.0


def test_rolling_kill_switch_uses_exit_order_not_entry_order() -> None:
    trades = [
        Trade(0, 200, 1, 100.0, 102.0, predicted_edge_bps=1.0),
        Trade(10, 100, 1, 100.0, 99.0, predicted_edge_bps=1.0),
        Trade(150, 250, 1, 100.0, 101.0, predicted_edge_bps=1.0),
    ]
    config = PortfolioConfig(
        capital=10_000.0,
        base_notional=1_000.0,
        max_notional=1_000.0,
        rolling_loss_limit=5.0,
        rolling_window=2,
    )

    result = simulate_fixed_notional_portfolio(trades, config)

    assert not result.trades[1].rejected
    assert result.trades[2].rejected
    assert result.trades[2].rejection_reason == "kill_switch"
    assert result.total_net_pnl == 10.0


def test_signed_inventory_is_distinct_from_gross_exposure_and_margin() -> None:
    trades = [
        Trade(0, 1000, 1, 100.0, 101.0, predicted_edge_bps=1.0),
        Trade(100, 1500, -1, 100.0, 99.0, predicted_edge_bps=1.0),
    ]
    config = PortfolioConfig(
        capital=10_000.0,
        base_notional=1_000.0,
        max_notional=1_000.0,
        initial_margin_rate=0.1,
        max_inventory_notional=1_000.0,
    )

    result = simulate_fixed_notional_portfolio(trades, config)

    assert not result.trades[1].rejected
    assert result.trades[0].inventory_notional == 1_000.0
    assert result.trades[1].inventory_notional == 0.0
    assert result.trades[1].margin_used == 200.0
    assert result.max_inventory == 1_000.0
    assert result.max_exposure == 2_000.0
    assert result.max_leverage == 0.2
    assert result.max_margin_used == 200.0


def test_portfolio_rejects_invalid_trade_inputs() -> None:
    config = PortfolioConfig(capital=10_000.0, base_notional=1_000.0, max_notional=1_000.0)
    cases = [
        (Trade(0, 1000, 0, 100.0, 101.0), "trade side"),
        (Trade(1000, 1000, 1, 100.0, 101.0), "exit_time_ms"),
        (Trade(0, 1000, 1, 0.0, 101.0), "trade prices"),
        (Trade(0, 1000, 1, 100.0, 101.0, notional=-1.0), "trade notional"),
    ]

    for trade, message in cases:
        try:
            simulate_fixed_notional_portfolio([trade], config)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"expected invalid trade to fail: {trade}")


def test_sizing_helpers_are_capped() -> None:
    assert capped_expected_edge_notional(0.5, base_notional=1000, max_notional=5000, full_size_edge_bps=1.0) == 500
    assert (
        fractional_kelly_notional(mean_edge=0.01, variance=0.02, capital=10000, fraction=0.25, cap_fraction=0.01) == 100
    )


def test_inventory_penalty_reduces_net_pnl() -> None:
    trades = [Trade(0, 1000, 1, 100.0, 101.0, predicted_edge_bps=1.0)]
    no_penalty = simulate_fixed_notional_portfolio(
        trades,
        PortfolioConfig(capital=10_000.0, base_notional=1_000.0, max_notional=1_000.0),
    )
    penalty = simulate_fixed_notional_portfolio(
        trades,
        PortfolioConfig(
            capital=10_000.0,
            base_notional=1_000.0,
            max_notional=1_000.0,
            inventory_penalty_bps=10.0,
        ),
    )

    assert penalty.total_net_pnl < no_penalty.total_net_pnl


def test_kelly_variance_gate_passes_stable_oos_variance(tmp_path: Path) -> None:
    path = tmp_path / "folds.csv"
    _write_fold_pnl(path, [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0])

    report = evaluate_oos_variance_stability(
        path,
        column="test_net_pnl",
        min_observations=9,
        window_size=3,
        max_variance_cv=0.01,
    )
    text = format_variance_stability_report(report)
    csv_text = format_variance_stability_report(report, output_format="csv")
    notional = gated_fractional_kelly_notional(
        mean_edge=0.01,
        capital=10_000.0,
        variance_report=report,
        fraction=0.25,
        cap_fraction=0.01,
    )

    assert report.passed
    assert report.variance_mean == 1.0
    assert "passed=1" in text
    assert csv_text.splitlines()[1].endswith(",1")
    assert notional == 25.0


def test_gated_kelly_returns_zero_for_unstable_variance(tmp_path: Path) -> None:
    path = tmp_path / "folds.csv"
    _write_fold_pnl(path, [1.0, 1.1, 1.2, -10.0, 0.0, 10.0, 1.0, 1.1, 1.2])

    report = evaluate_oos_variance_stability(
        path,
        column="test_net_pnl",
        min_observations=9,
        window_size=3,
        max_variance_cv=0.1,
    )
    notional = gated_fractional_kelly_notional(
        mean_edge=0.01,
        capital=10_000.0,
        variance_report=report,
    )

    assert not report.passed
    assert notional == 0.0


def test_variance_gate_ignores_summary_rows(tmp_path: Path) -> None:
    path = tmp_path / "folds.csv"
    _write_fold_pnl(path, [1.0, 2.0, 3.0, 2.0])
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold", "test_net_pnl"])
        writer.writerow({"fold": "summary", "test_net_pnl": 1000.0})

    report = evaluate_oos_variance_stability(
        path,
        column="test_net_pnl",
        min_observations=4,
        window_size=2,
        max_variance_cv=0.5,
    )

    assert report.observations == 4
    assert report.window_variances == (0.5, 0.5)


def _write_fold_pnl(path: Path, values: list[float]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold", "test_net_pnl"])
        writer.writeheader()
        for index, value in enumerate(values, start=1):
            writer.writerow({"fold": index, "test_net_pnl": value})
