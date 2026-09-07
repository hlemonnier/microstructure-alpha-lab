import csv
import json
import math
from dataclasses import replace

import pytest

from lob_forge.data_sources import NormalizedL2Row, _bybit_orderbook_payload_to_rows, normalize_l2_row
from lob_forge.holdout import build_holdout_manifest, canonical_json_sha256, write_holdout_manifest
from lob_forge.l2_replay import AtomicOrderBookReplayer, iter_l2_events, validate_l2_replay_contract
from lob_forge.ml_models import (
    L2SnapshotVector,
    SEQUENCE_CHECKPOINT_CONTRACT_VERSION,
    SEQUENCE_ECONOMICS_VERSION,
    _load_l2_top_n_vectors_from_rows,
    _restore_class_probabilities,
    _stationarize_l2_sequences,
    _sequence_stateful_economics,
    _predict_sequence_payoff_side,
    _fit_sequence_payoff_policy,
    build_torch_sequence_classifier,
    evaluate_l2_tensor_readiness,
    evaluate_l2_sequence_final_holdout,
    fit_xgboost_classifier,
    freeze_l2_sequence_candidate,
    predict_sklearn_probabilities,
    run_l2_torch_sequence_experiment,
)


def _row(kind, side, price, size, timestamp=1000, sequence=1, **kw):
    return NormalizedL2Row(
        kind, timestamp, side, price, size, sequence=sequence, update_id=sequence, venue="test", symbol="TEST", **kw
    )


def _snapshot():
    return [_row("snapshot", "bid", 99, 1), _row("snapshot", "ask", 101, 1)]


def test_atomic_message_does_not_flag_transient_cross_and_caps_whole_message():
    rows = _snapshot() + [
        _row("delta", "bid", 102, 2, 2000, 2),
        _row("delta", "ask", 101, 0, 2000, 2),
        _row("delta", "ask", 103, 3, 2000, 2),
    ]
    assert validate_l2_replay_contract(rows).passed
    assert list(iter_l2_events(rows, max_rows=4)) == [rows[:2]]
    def asdict(r):
        return {key: value for key, value in vars(r).items() if value is not None}
    vectors, count = _load_l2_top_n_vectors_from_rows([asdict(r) for r in rows], depth=1, max_rows=4, max_snapshots=10)
    assert count == 2 and len(vectors) == 1
    assert vectors[0].bids == ((99, 1),)


@pytest.mark.parametrize(
    "invalid",
    [
        [_row("delta", "bid", 99, 2, 900, 2)],
        [_row("delta", "bid", 99, 2, 2000, 1)],
        [_row("delta", "bid", 99, 2, 2000, 3)],
        [replace(_row("delta", "bid", 99, 2, 2000, 2), symbol="OTHER")],
        [_row("delta", "bid", math.nan, 2, 2000, 2)],
        [_row("delta", "bid", 99, math.inf, 2000, 2)],
    ],
)
def test_readiness_and_tensor_reader_share_invalid_event_rejection(tmp_path, invalid):
    rows = _snapshot() + invalid
    path = tmp_path / "l2.csv"
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(vars(rows[0])))
        w.writeheader()
        w.writerows(vars(row) for row in rows)
    assert not evaluate_l2_tensor_readiness(path, min_rows=1).passed
    with path.open(newline="") as h, pytest.raises(ValueError):
        _load_l2_top_n_vectors_from_rows(csv.DictReader(h), depth=1, max_rows=100, max_snapshots=100)


def test_snapshot_resets_sequence_epoch_and_delta_without_snapshot_fails():
    replayer = AtomicOrderBookReplayer()
    replayer.apply_event([replace(row, sequence=100, update_id=100) for row in _snapshot()])
    replayer.apply_event([replace(row, exchange_timestamp=2000) for row in _snapshot()])
    replayer.apply_event([_row("delta", "bid", 99, 2, 3000, 2)])
    assert replayer.bids[99] == 2
    with pytest.raises(ValueError, match="initial valid snapshot"):
        AtomicOrderBookReplayer().apply_event([_row("delta", "bid", 99, 2)])


def test_bybit_matching_engine_and_publishing_time_are_not_receipt_time():
    payload = {
        "type": "snapshot",
        "cts": 100,
        "ts": 120,
        "data": {"s": "TEST", "u": 1, "seq": 11, "b": [["99", "1"]], "a": [["101", "1"]]},
    }
    rows = list(_bybit_orderbook_payload_to_rows(payload, default_symbol=None))
    assert rows[0].exchange_timestamp == 100
    assert rows[0].publisher_timestamp == 120
    assert rows[0].local_timestamp is None
    assert normalize_l2_row(vars(rows[0]), require_sequence=True).publisher_timestamp == 120


def test_causal_return_channel_retains_price_movement_and_no_future_dependence():
    up = [[101 * c, 1, 99 * c, 1, 0] for c in [1, 2, 4]]
    down = list(reversed(up))
    transformed = _stationarize_l2_sequences([up, down])
    assert transformed[0][-1][:-1] == transformed[1][-1][:-1]
    assert transformed[0][-1][-1] > 0 > transformed[1][-1][-1]
    modified = [*up[:2], [999, 1, 997, 1, 0]]
    assert _stationarize_l2_sequences([modified])[0][:2] == transformed[0][:2]


def test_balanced_probability_correction_recovers_original_distribution():
    p = [0.2, 0.3, 0.5]
    priors = [0.1, 0.1, 0.8]
    q = [p[k] / priors[k] for k in range(3)]
    q = [v / sum(q) for v in q]
    assert _restore_class_probabilities([q], priors, "balanced")[0] == pytest.approx(p)


def test_tcn_context_covers_actual_requested_window():
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    model = build_torch_sequence_classifier(window=64, feature_count=6, model_name="sequence_tcn")
    assert model.receptive_field >= 64
    x = torch.randn(1, 64, 6, requires_grad=True)
    model(x).sum().backward()
    assert x.grad[0, 0].abs().sum().item() > 0


def test_xgboost_probability_adapter_preserves_economic_labels():
    pytest.importorskip("xgboost")
    rows = [{"x": str(i % 3), "label": str((i % 3) - 1)} for i in range(30)]
    model = fit_xgboost_classifier(rows, ["x"])
    probabilities = predict_sklearn_probabilities(model, rows, ["x"])
    assert set(probabilities[0]) == {-1, 0, 1}
    assert max(probabilities[0], key=probabilities[0].get) == -1


def _vectors(prices, *, size=1.0, step=1000):
    return [
        L2SnapshotVector(
            [ask, size, bid, size],
            timestamp_ms=1000 + i * step,
            bids=((bid, size),),
            asks=((ask, size),),
            event_index=i,
        )
        for i, (bid, ask) in enumerate(prices)
    ]


def test_execution_uses_actual_depth_and_declared_latency():
    tiny = _vectors([(99, 101), (103, 105)], size=0.1)
    result = _sequence_stateful_economics(
        tiny, [0], [2], label_horizon=1, target_notional=100, taker_fee_bps=0, slippage_bps=0
    )
    assert result["turnover"] == pytest.approx(20.4)
    assert result["net_pnl"] == pytest.approx(0.2)
    # Entry at 2000 arrives at the second snapshot, close at 3000; true clock used.
    delayed = _sequence_stateful_economics(
        _vectors([(99, 101), (103, 105), (107, 109)]),
        [0],
        [2],
        label_horizon=1,
        target_notional=100,
        taker_fee_bps=0,
        slippage_bps=0,
        latency_ms=1000,
    )
    assert delayed["turnover"] == pytest.approx((105 + 107) * (100 / 104))
    assert delayed["final_inventory"] == 0
    with pytest.raises(ValueError, match="timestamped"):
        _sequence_stateful_economics(
            [[101, 1, 99, 1], [105, 1, 103, 1]],
            [0],
            [2],
            label_horizon=1,
            target_notional=100,
            taker_fee_bps=0,
            slippage_bps=0,
        )


def test_payoff_policy_can_reject_high_directional_confidence_and_never_reads_test():
    policy = {
        "version": "conditional_payoff_v1",
        "bins": 5,
        "min_observations": 5,
        "margin_bps": 0.0,
        "cells": {"2:4": {"count": 20, "short_net_bps": -0.5, "long_net_bps": -1.1}},
    }
    assert _predict_sequence_payoff_side(policy, [0.1, 0, 0.9]) == 1
    vectors = _vectors([(99, 101)] * 8 + [(199, 201)] * 2)
    kwargs = dict(
        label_horizon=1, target_notional=100, taker_fee_bps=0, slippage_bps=0, latency_ms=1000, calibration_stop_index=8
    )
    first = _fit_sequence_payoff_policy(vectors, list(range(5)), [[0.1, 0, 0.9]] * 5, **kwargs)
    changed = vectors[:8] + _vectors([(999, 1001), (999, 1001)])
    second = _fit_sequence_payoff_policy(changed, list(range(5)), [[0.1, 0, 0.9]] * 5, **kwargs)
    assert first == second
    assert _predict_sequence_payoff_side(first, [0.1, 0, 0.9]) == 1


def _write_clean_l2(path, n=100):
    rows = []
    for i in range(n):
        # Snapshot refreshes are authoritative, contiguous; only every fifth event
        # is a delta so no stale deep levels build up.
        kind = "snapshot" if i % 5 == 0 else "delta"
        bid = 100 + (i % 5) * 0.02
        levels = []
        if kind == "delta":
            previous = 100 + ((i - 1) % 5) * 0.02
            levels += [("bid", previous, 0), ("ask", previous + 0.01, 0)]
        levels += [("bid", bid, 10), ("ask", bid + 0.01, 10)]
        for side, price, size in levels:
            rows.append(vars(_row(kind, side, price, size, 100000 + i * 10, i + 1)))
    with path.open("w", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("model_name", ["sequence_tcn", "sequence_transformer"])
def test_actual_cpu_training_freeze_and_holdout_contract(tmp_path, model_name):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    l2 = tmp_path / "l2.csv"
    _write_clean_l2(l2)
    audit = tmp_path / "audit.csv"
    audit.write_text("fold_count,acceptance_passed,rejection_reasons\n20,1,\n")
    manifest = build_holdout_manifest(
        l2,
        split_column="exchange_timestamp",
        holdout_values=[str(100000 + i * 10) for i in range(80, 100)],
        created_at_utc="2026-09-07T00:00:00Z",
        git_commit="a" * 40,
    )
    manifest_path = tmp_path / "holdout.json"
    write_holdout_manifest(manifest, manifest_path)
    artifact = tmp_path / "model.csv"
    checkpoint = tmp_path / "model.pt"
    report = run_l2_torch_sequence_experiment(
        model_name=model_name,
        l2_path=l2,
        baseline_audit_path=audit,
        output_path=artifact,
        depth=1,
        window=8,
        label_horizon=1,
        epochs=2,
        batch_size=8,
        device="cpu",
        min_l2_rows=20,
        max_rows=10000,
        max_snapshots=1000,
        checkpoint_path=checkpoint,
        prediction_output_path=tmp_path / "predictions.csv",
        holdout_manifest_path=manifest_path,
        class_weighting="balanced",
        economic_latency_ms=5,
        economic_taker_fee_bps=0,
    )
    assert report.pipeline_completed and report.best_epoch > 0
    assert report.feature_count == 6  # four book fields, elapsed time, causal return
    assert report.economic_simulation_version == SEQUENCE_ECONOMICS_VERSION
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["experiment_contract"]["contract_version"] == SEQUENCE_CHECKPOINT_CONTRACT_VERSION
    assert "torch_rng_state" in payload
    candidate = freeze_l2_sequence_candidate(artifact)
    assert candidate["candidate_type"] == "l2_sequence_torch_v3"
    assert candidate["economic_latency_ms"] == 5
    with pytest.raises(ValueError, match="economic_taker_fee_bps must match"):
        evaluate_l2_sequence_final_holdout(
            l2_path=l2, manifest=manifest, candidate=candidate, economic_taker_fee_bps=100
        )
    # A separate final manifest locks the frozen candidate; full source replay
    # hydrates its initial book without using warmup observations as final labels.
    final = replace(manifest, candidate_sha256=canonical_json_sha256(candidate))
    held = evaluate_l2_sequence_final_holdout(
        l2_path=l2,
        manifest=final,
        candidate=candidate,
        predictions_output_path=tmp_path / "held.csv",
        device="cpu",
        max_rows=10000,
        max_snapshots=1000,
    )
    assert held.sequence_count > 0 and math.isfinite(held.brier_score)
    assert held.economic_latency_ms == 5
    exported = list(csv.DictReader((tmp_path / "held.csv").open()))
    assert all("trading_label" in row for row in exported)
    assert sum(json.loads(report.class_priors_json)) == pytest.approx(1)
