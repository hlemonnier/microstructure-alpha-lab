import csv
import io
import json
import math
from pathlib import Path
import zipfile

import pytest

from lob_forge.execution_sim import MarketEvent, SignalEvent, StatefulExecutionConfig, simulate_stateful_execution
from lob_forge.performance_replay import evaluate_policy, load_quote_tape


def _tape():
    return [
        MarketEvent(59000, 99, 101, 10, 10),
        MarketEvent(60000, 99, 101, 10, 10),
        MarketEvent(60100, 99, 101, 10, 10),
        MarketEvent(60250, 104, 106, 10, 10),
        MarketEvent(62000, 110, 112, 10, 10),
        MarketEvent(65100, 119, 121, 10, 10),
        MarketEvent(65250, 109, 111, 10, 10),
        MarketEvent(66000, 109, 111, 10, 10),
    ]


def test_positive_payoff_uses_one_account_and_exact_two_leg_fees() -> None:
    result = evaluate_policy(
        [{"decision_time": "60000"}], [[3000, -3000]], _tape(),
        threshold_bps=0, taker_fee_bps=1, latency_ms=100, include_ledgers=True,
    )
    assert result.metrics["filled_entries"] == result.metrics["completed_round_trips"] == 1
    assert result.metrics["fill_count"] == 2
    assert result.metrics["turnover"] == 220
    assert math.isclose(result.metrics["net_pnl"], 18 - 220 / 10000, abs_tol=1e-9)
    assert math.isclose(result.metrics["max_drawdown_pnl"], 1 + 101 / 10000, abs_tol=1e-9)
    assert result.metrics["final_inventory"] == 0
    assert math.isclose(result.metrics["break_even_taker_fee_bps"], 18 / 220 * 10000, abs_tol=1e-8)
    assert [fill.fill_time_ms for fill in result.simulation.fills] == [60100, 65100]
    json.dumps(result.metrics, allow_nan=False)


def test_latency_stress_keeps_policy_but_changes_both_execution_legs() -> None:
    fast = evaluate_policy(
        [{"decision_time": 60000}], [[3000, -3000]], _tape(),
        threshold_bps=0, taker_fee_bps=0, latency_ms=100, include_ledgers=True,
    )
    slow = evaluate_policy(
        [{"decision_time": 60000}], [[3000, -3000]], _tape(),
        threshold_bps=0, taker_fee_bps=0, latency_ms=250, include_ledgers=True,
    )
    assert fast.signal_ledger == slow.signal_ledger
    assert [fill.fill_time_ms for fill in slow.simulation.fills] == [60250, 65250]
    assert slow.metrics["net_pnl"] < fast.metrics["net_pnl"]
    assert math.isclose(slow.metrics["net_pnl"], (109 - 106) * (100 / 105), abs_tol=1e-8)


def test_fee_stress_can_freeze_prediction_policy_fee() -> None:
    args = dict(rows=[{"decision_time": 60000}], gross_predictions=[[1.5, -2]], tape=_tape(),
                threshold_bps=0, taker_fee_bps=1, latency_ms=100)
    repriced = evaluate_policy(**args)
    frozen = evaluate_policy(**args, policy_fee_bps=0)
    assert repriced.metrics["filled_entries"] == 0
    assert frozen.metrics["filled_entries"] == 1
    assert math.isclose(frozen.metrics["fees_and_slippage_paid"], 220 / 10000)


def test_partial_exit_blocks_new_risk_and_retries_flattening() -> None:
    tape = [
        MarketEvent(59000, 99, 101, 10, 10),
        MarketEvent(60100, 99, 101, 10, 10),
        MarketEvent(65100, 109, 111, 0.2, 10),
        MarketEvent(69000, 109, 111, 10, 10),
        MarketEvent(70100, 119, 121, 10, 10),
        MarketEvent(76000, 119, 121, 10, 10),
        MarketEvent(80100, 119, 121, 10, 10),
        MarketEvent(85100, 129, 131, 10, 10),
        MarketEvent(86000, 129, 131, 10, 10),
    ]
    result = evaluate_policy(
        [{"decision_time": time, "quote_age_ms": 1001 if time == 70000 else 0}
         for time in (60000, 70000, 80000)],
        [[3000, -3000]] * 3, tape, threshold_bps=0, taker_fee_bps=0,
        latency_ms=100, include_ledgers=True,
    )
    assert [row["action"] for row in result.signal_ledger] == ["entry", "residual_flatten", "entry"]
    assert result.metrics["residual_flatten_retries"] == 1
    assert result.metrics["filled_entries"] == result.metrics["completed_round_trips"] == 2
    assert math.isclose(result.signal_ledger[1]["inventory_before_decision"], 0.8)
    assert abs(result.metrics["final_inventory"]) < 1e-10
    assert any(fill.partial for fill in result.simulation.fills)


def test_unliquidated_inventory_is_marked_and_reported() -> None:
    tape = [MarketEvent(59000, 99, 101, 10, 10), MarketEvent(60100, 99, 101, 10, 10),
            MarketEvent(65100, 109, 111, 0, 10), MarketEvent(66000, 119, 121, 0, 10)]
    result = evaluate_policy(
        [{"decision_time": 60000}], [[3000, -3000]], tape,
        threshold_bps=0, taker_fee_bps=0, latency_ms=100,
    )
    assert result.metrics["final_inventory"] == 1
    assert result.metrics["net_pnl"] == 19
    assert result.metrics["realized_pnl"] == 0
    assert result.metrics["completed_round_trips"] == 0


def test_flat_fast_path_has_no_fabricated_trades_or_undefined_json() -> None:
    result = evaluate_policy(
        [{"decision_time": 60000}], [[3000, -3000]], _tape(),
        threshold_bps=math.inf, taker_fee_bps=1, latency_ms=100, include_ledgers=True,
    )
    assert result.metrics["net_pnl"] == result.metrics["turnover"] == result.metrics["filled_entries"] == 0
    assert result.metrics["break_even_taker_fee_bps"] is None
    assert result.metrics["replay_quote_count"] == 2
    assert result.metrics["raw_quote_count"] == len(_tape())
    assert not result.simulation.fills
    json.dumps(result.metrics, allow_nan=False)


def test_flat_interval_compression_matches_full_engine_equity_fills_and_drawdown() -> None:
    tape = [MarketEvent(time, 99, 101, 10, 10) for time in range(0, 60000, 100)] + _tape()[1:]
    result = evaluate_policy(
        [{"decision_time": 60000}], [[3000, -3000]], tape,
        threshold_bps=0, taker_fee_bps=1, latency_ms=100, include_ledgers=True,
    )
    full = simulate_stateful_execution(
        tape, [SignalEvent(60000, 1, 100, signal_id="policy-0-entry", predicted_edge_bps=2997.7),
               SignalEvent(65000, 0, 0, signal_id="policy-0-exit")],
        config=StatefulExecutionConfig(10000, 200, 1, taker_fee_bps=1, latency_ms=100),
    )
    assert result.simulation.fills == full.fills
    assert result.simulation.final_equity == full.final_equity
    assert result.simulation.final_inventory == full.final_inventory
    assert result.simulation.turnover == full.turnover
    peak = 10000
    drawdown = 0
    for row in full.positions:
        peak = max(peak, row.equity)
        drawdown = max(drawdown, peak - row.equity)
    assert result.metrics["max_drawdown_pnl"] == drawdown
    assert result.metrics["replay_quote_count"] < result.metrics["raw_quote_count"] / 10


def test_stride_funding_exclusion_and_exit_buffer_are_causal() -> None:
    times = (0, 50000, 60001, 70000)
    tape = [MarketEvent(0, 99, 101, 10, 10), MarketEvent(75000, 99, 101, 10, 10)]
    result = evaluate_policy(
        [{"decision_time": value} for value in times], [[3000, -3000]] * len(times), tape,
        threshold_bps=0, taker_fee_bps=0, latency_ms=100,
    )
    assert [row["action"] for row in result.signal_ledger] == [
        "funding_exclusion", "funding_exclusion", "insufficient_exit_tape",
    ]
    assert result.metrics["filled_entries"] == 0


def test_quote_loader_retains_tied_updates_and_observed_liquidity(tmp_path: Path) -> None:
    path = tmp_path / "quotes.zip"
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["update_id", "best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty",
                     "transaction_time", "event_time"])
    writer.writerows([[1, 99, 0, 101, 2, 1000, 1000], [2, 100, 3, 102, 4, 1000, 1000],
                      [3, 101, 5, 103, 6, 1001, 1001]])
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("quotes.csv", buffer.getvalue())
    tape = load_quote_tape(path, start_ms=1000, end_ms=1000 + 1)
    assert [event.timestamp_ms for event in tape] == [1000, 1000, 1001]
    assert [event.bid_size for event in tape] == [0, 3, 5]
    assert all(event.trade_size == 0 and event.trade_side is None for event in tape)


def test_replay_rejects_ambiguous_order_and_nonfinite_predictions() -> None:
    with pytest.raises(ValueError, match="positive latency"):
        evaluate_policy([{"decision_time": 60000}], [[1, -1]], _tape(),
                        threshold_bps=0, taker_fee_bps=0, latency_ms=0)
    with pytest.raises(ValueError, match="finite"):
        evaluate_policy([{"decision_time": 60000}], [[math.nan, -1]], _tape(),
                        threshold_bps=0, taker_fee_bps=0, latency_ms=100)
    with pytest.raises(ValueError, match="strictly increasing"):
        evaluate_policy([{"decision_time": 60000}] * 2, [[1, -1]] * 2, _tape(),
                        threshold_bps=0, taker_fee_bps=0, latency_ms=100)


def test_cost_conversion_keeps_small_long_short_asymmetry() -> None:
    result = evaluate_policy(
        [{"decision_time": 60000}], [[2, 2]], _tape(),
        threshold_bps=0, taker_fee_bps=1, latency_ms=100,
    )
    assert result.signal_ledger[0]["long_net_bps"] < 0
    assert result.signal_ledger[0]["short_net_bps"] > 0
    assert result.signal_ledger[0]["predicted_side"] == -1
    assert result.metrics["filled_entries"] == 1


def test_stale_quote_suppresses_entry_using_observable_age() -> None:
    result = evaluate_policy(
        [{"decision_time": 60000, "quote_age_ms": 1001}], [[3000, -3000]], _tape(),
        threshold_bps=0, taker_fee_bps=0, latency_ms=100,
    )
    assert result.signal_ledger[0]["action"] == "stale_feature_quote"
    assert result.metrics["filled_entries"] == 0


def test_scheduler_matches_engine_when_leverage_rejects_entry() -> None:
    result = evaluate_policy(
        [{"decision_time": 60000}], [[3000, -3000]], _tape(),
        threshold_bps=0, taker_fee_bps=1, latency_ms=100,
        initial_cash=100, target_notional=100, include_ledgers=True,
    )
    assert result.metrics["scheduled_entries"] == 1
    assert result.metrics["filled_entries"] == 0
    assert result.simulation.orders[0].reason == "max_leverage_after_costs"


def test_compression_preserves_partial_fill_retries() -> None:
    tape = [MarketEvent(time, 99, 101, 10, 10) for time in range(0, 60000, 100)] + [
        MarketEvent(60100, 99, 101, 10, 10),
        MarketEvent(65100, 109, 111, 0.2, 10),
        MarketEvent(70100, 119, 121, 0.3, 10),
        MarketEvent(79000, 119, 121, 10, 10),
        MarketEvent(80100, 129, 131, 10, 10),
        MarketEvent(86000, 129, 131, 10, 10),
    ]
    result = evaluate_policy(
        [{"decision_time": value} for value in (60000, 70000, 80000)], [[3000, -3000]] * 3,
        tape, threshold_bps=0, taker_fee_bps=1, latency_ms=100, include_ledgers=True,
    )
    signals = [
        SignalEvent(60000, 1, 100, signal_id="policy-0-entry"),
        SignalEvent(65000, 0, 0, signal_id="policy-0-exit"),
        SignalEvent(70000, 0, 0, signal_id="policy-1-flatten-retry"),
        SignalEvent(80000, 0, 0, signal_id="policy-2-flatten-retry"),
    ]
    full = simulate_stateful_execution(
        tape, signals, config=StatefulExecutionConfig(10000, 200, 1, taker_fee_bps=1, latency_ms=100),
    )
    assert result.simulation.fills == full.fills
    assert result.simulation.orders == full.orders
    assert result.simulation.final_cash == full.final_cash
    assert result.simulation.final_equity == full.final_equity
    assert result.metrics["completed_round_trips"] == 1


def test_sparse_tape_expires_entry_instead_of_filling_a_stale_instruction() -> None:
    tape = [MarketEvent(59000, 99, 101, 10, 10), MarketEvent(63000, 109, 111, 10, 10),
            MarketEvent(65100, 119, 121, 10, 10), MarketEvent(66000, 119, 121, 10, 10)]
    result = evaluate_policy(
        [{"decision_time": 60000}], [[3000, -3000]], tape,
        threshold_bps=0, taker_fee_bps=0, latency_ms=100, include_ledgers=True,
    )
    assert result.simulation.orders[0].status == "expired"
    assert result.metrics["filled_entries"] == result.metrics["net_pnl"] == 0
