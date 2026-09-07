"""Independent counterexamples from the September model/maths audit."""
import csv
import math
import zipfile
from pathlib import Path

from lob_forge.baselines import (
    DEFAULT_FEATURES, compute_metrics, pool_metrics, evaluate_taker_economics,
    run_walk_forward_thresholds,
)
from lob_forge.calibration import posterior_mean_score, multiclass_brier_score
from lob_forge.edge_model import (
    _fit_ridge_weights, fit_edge_model, predict_net_edges_bps,
    run_edge_shadow_decisions_streaming,
)
from lob_forge.features import (
    BOOK_TICKER_COLUMNS, iter_quote_buckets, resolve_raw_execution_quote_resolutions, QuoteBucket,
)
from lob_forge.logistic import fit_softmax_model, predict_probabilities, run_logistic_walk_forward
from lob_forge.protocol import construct_causal_samples, purged_walk_forward_indices


def _book(path: Path, values: list[tuple[int, float, float, float, float]]) -> None:
    rows = [','.join(BOOK_TICKER_COLUMNS)]
    for i, (time, bid, ask, bid_qty, ask_qty) in enumerate(values, 1):
        rows.append(f'{i},{bid},{bid_qty},{ask},{ask_qty},{time},{time}')
    with zipfile.ZipFile(path, 'w') as handle:
        handle.writestr('book.csv', '\n'.join(rows) + '\n')


def test_decision_schedule_uses_completed_boundaries_and_raw_ofi(tmp_path: Path) -> None:
    a = tmp_path / 'a.zip'
    b = tmp_path / 'b.zip'
    base = [(100, 100, 102, 1, 1), (900, 101, 103, 3, 2)]
    _book(a, base + [(1000, 101, 103, 3, 2), (2000, 101, 103, 3, 2)])
    _book(b, base + [(950, 100, 102, 5, 4), (1000, 101, 103, 3, 2), (2000, 101, 103, 3, 2)])
    qa = list(iter_quote_buckets(a, bucket_ms=1000))
    qb = list(iter_quote_buckets(b, bucket_ms=1000))
    assert [q.decision_time for q in qa] == [1000, 2000]
    assert [q.decision_time for q in qb] == [1000, 2000]
    assert all(q.event_time < q.decision_time for q in qa + qb)
    # First transition +4, second -7: OFI sums actual events even though
    # the endpoint returns to its original price with new queue quantities.
    assert qa[0].raw_ofi == 4
    assert qb[0].raw_ofi == -3
    # EOF at 950 cannot establish completion of [0,1000).
    _book(a, base)
    assert list(iter_quote_buckets(a, bucket_ms=1000)) == []


def test_raw_and_protocol_entry_cannot_precede_decision_update(tmp_path: Path) -> None:
    path = tmp_path / 'ties.zip'
    _book(path, [(900, 99, 101, 1, 1), (900, 109, 111, 1, 1), (1900, 109, 111, 1, 1)])
    quote = QuoteBucket(0, 900, 2, 109, 111, 1, 1, decision_time_ms=900)
    result = resolve_raw_execution_quote_resolutions(path, [quote], execution_latency_ms=0, horizon_ms=1000)[0]
    assert result is not None
    assert result.entry.update_id == 2 and result.entry.ask == 111
    events = [{'event_time': '900', 'x': '1'}, {'event_time': '900', 'x': '2'},
              {'event_time': '1900', 'x': '3'}]
    sample = construct_causal_samples(events, decision_times=[900], feature_columns=['x'])[0]
    assert sample.features['x'] == 2 and sample.entry_row['x'] == '2'


def test_all_walk_forward_schedules_reject_overlapping_test_blocks(tmp_path: Path) -> None:
    (tmp_path / 'unused.csv').write_text('label,x\n1,1\n')
    for runner in (run_walk_forward_thresholds, run_logistic_walk_forward):
        try:
            runner(tmp_path / 'unused.csv', train_size=4, validation_size=4, test_size=4, step_size=1)
        except ValueError as exc:
            assert 'step_size' in str(exc)
        else:
            raise AssertionError('overlapping OOS blocks were accepted')
    try:
        purged_walk_forward_indices(20, train_size=4, validation_size=4, test_size=4, step_size=1)
    except ValueError as exc:
        assert 'step_size' in str(exc)
    else:
        raise AssertionError('overlapping protocol blocks were accepted')


def test_balanced_softmax_returns_original_class_posterior() -> None:
    rows = []
    for feature, counts in [(1, {-1: 8, 0: 12, 1: 20}), (0, {-1: 12, 0: 8, 1: 140})]:
        for label, count in counts.items():
            rows.extend([{'x': str(feature), 'label': str(label)} for _ in range(count)])
    model = fit_softmax_model(rows, ['x'], epochs=1000, learning_rate=.2, l2=0, class_weighting='balanced')
    probabilities = predict_probabilities(model, {'x': '1'})
    assert all(abs(actual - expected) < .002 for actual, expected in zip(probabilities, [.2, .3, .5]))
    assert probabilities[2] > probabilities[0]


def test_predicted_net_edge_matches_exact_two_leg_costs() -> None:
    row = {'x': '0', 'label': '1', 'bid': '100', 'ask': '100',
           'future_bid': '100.10001', 'future_ask': '100.10001'}
    model = fit_edge_model([row], ['x'], l2=1)
    predictions = predict_net_edges_bps(model, row, taker_fee_bps=5, slippage_bps=0)
    for predicted, side in zip(predictions, [1, -1]):
        measured = evaluate_taker_economics([row], lambda _: side, taker_fee_bps=5, slippage_bps=0).mean_net_return_bps_per_trade
        assert abs(predicted - measured) < 1e-9
    assert predictions[0] < 0  # additive 2*c approximation incorrectly made it positive.


def test_flat_export_remains_flat_when_diagnostics_include_it(tmp_path: Path) -> None:
    path = tmp_path / 'features.csv'
    rows = []
    for i in range(12):
        future = 101 if i < 4 or i >= 8 else 99
        rows.append({'event_time': str(i * 1000), 'future_event_time': str(i * 1000 + 500),
                     'x': '0', 'label': str(1 if future > 100 else -1), 'bid': '100',
                     'ask': '100', 'future_bid': str(future), 'future_ask': str(future)})
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    decisions = run_edge_shadow_decisions_streaming(
        path, features=['x'], train_size=4, validation_size=4, test_size=4,
        edge_thresholds_bps=[0], taker_fee_bps=0, venue='test', symbol='TEST',
        intended_size=1, include_flat=True,
    )
    assert len(decisions) == 4
    assert all(decision.predicted_side == 0 for decision in decisions)


def test_householder_ridge_recovers_near_collinear_design_without_jitter() -> None:
    matrix = [[1., i / 5, i / 5 + (1e-7 if i % 2 else -1e-7)] for i in range(1, 20)]
    target = [2 + 3 * row[1] - 4 * row[2] for row in matrix]
    weights = _fit_ridge_weights(matrix, target, l2=0)
    assert all(abs(a - b) < 1e-7 for a, b in zip(weights, [2, 3, -4]))
    try:
        _fit_ridge_weights([[1., 1., 1.]] * 4, [1.] * 4, l2=0)
    except ValueError as exc:
        assert 'singular' in str(exc)
    else:
        raise AssertionError('unidentified coefficients must not be silently jittered')
    assert 'microprice_deviation' not in DEFAULT_FEATURES
    assert 'top_imbalance' in DEFAULT_FEATURES


def test_pooled_classification_metrics_recompute_confusion_counts() -> None:
    first = compute_metrics([-1, -1, -1, -1], [-1, -1, -1, 1])
    second = compute_metrics([1, 0], [-1, 0])
    pooled = pool_metrics([first, second])
    direct = compute_metrics([-1, -1, -1, -1, 1, 0], [-1, -1, -1, 1, -1, 0])
    assert pooled == direct
    assert not math.isclose(pooled.macro_f1, (4 * first.macro_f1 + 2 * second.macro_f1) / 6)


def test_student_posterior_retains_uncertainty_and_respects_units() -> None:
    score = posterior_mean_score([1.])
    assert score.standard_error > 0 and .5 < score.p_mean_gt_zero < .99
    scaled = posterior_mean_score([10.], prior_scale=100.)
    assert math.isclose(score.p_mean_gt_zero, scaled.p_mean_gt_zero)
    assert math.isclose(score.standard_error * 10, scaled.standard_error)
    for bad in [float('nan'), float('inf')]:
        for compute in [lambda: posterior_mean_score([bad]),
                        lambda: multiclass_brier_score([1], [{1: bad, 0: 0, -1: 0}])]:
            try:
                compute()
            except ValueError:
                pass
            else:
                raise AssertionError('non-finite calibration input accepted')
