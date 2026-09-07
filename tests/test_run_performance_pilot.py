import csv
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lob_forge.binance_vision import sha256_file
from lob_forge.features import FEATURE_SEMANTICS_VERSION


def runner():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'run_performance_pilot.py'
    spec = importlib.util.spec_from_file_location('_performance_runner_test', path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def protocol():
    path = Path(__file__).resolve().parents[1] / 'docs/research/performance_pilot_20260907_window_revision.json'
    return json.loads(path.read_text())


def test_pilot_splits_are_chronological_and_reserved_dates_excluded():
    r = runner()
    p = protocol()
    dates = [f'2023-05-{i:02}' for i in range(16, 24)]
    splits = r.session_splits(dates, p)
    assert len(splits) == 3
    assert splits[0] == {'train_dates': dates[:4], 'validation_dates': [dates[4]], 'test_date': dates[5]}
    assert all(max(s['train_dates']) < min(s['validation_dates']) < s['test_date'] for s in splits)
    with pytest.raises(ValueError, match='reserved unseen'):
        r.session_splits([*dates, '2023-05-24'], p)
    with pytest.raises(ValueError, match='sorted and unique'):
        r.session_splits([dates[0], *dates], p)


def test_pilot_noon_eligibility_is_causal_and_hash_bound(tmp_path):
    r = runner()
    p = protocol()
    day = '2023-05-16'
    start = r.session_time(day, '12:01:00')
    stop = r.session_time(day, '14:00:00') - 5250
    rows = [
        {'decision_time': start - 1000, 'quote_age_ms': 0, 'future_lag_ms': 0},
        {'decision_time': start, 'quote_age_ms': 0, 'future_lag_ms': 10**8},
        {'decision_time': start + 1000, 'quote_age_ms': 1001, 'future_lag_ms': 0},
        {'decision_time': stop - 1, 'quote_age_ms': 1000, 'future_lag_ms': 0},
        {'decision_time': stop, 'quote_age_ms': 0, 'future_lag_ms': 0},
    ]
    for row in rows:
        row['feature_semantics_version'] = FEATURE_SEMANTICS_VERSION
    path = tmp_path / 'features.csv'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    session = {'features_path': str(path), 'session_date': day, 'feature': {'sha256': sha256_file(path)}}
    eligible = r.load_feature_rows(session, p)
    assert [int(row['decision_time']) for row in eligible] == [start, stop - 1]
    assert int(eligible[0]['future_lag_ms']) == 10**8  # Never select decisions on future quote timing.
    path.write_text('tampered')
    with pytest.raises(ValueError, match='feature hash mismatch'):
        r.load_feature_rows(session, p)


def test_pilot_purges_label_endpoint_equal_to_next_partition():
    r = runner()
    rows = [{'future_event_time': str(t)} for t in [999, 1000, 1001]]
    assert r.purge_before(rows, 1000) == rows[:1]


def test_validation_ties_prefer_flat_and_residual_inventory_cannot_win():
    r = runner()
    base = {'net_pnl': 0.0, 'filled_entries': 0, 'final_inventory_mark_notional': 0.0}
    assert r.rank_validation(base, math.inf) > r.rank_validation(base, 5.0)
    assert r.rank_validation(base, math.inf) > r.rank_validation({**base, 'filled_entries': 1}, 5.0)
    assert r.rank_validation(base, math.inf) > r.rank_validation({**base, 'net_pnl': 1000, 'final_inventory_mark_notional': 1}, 0)


def test_threshold_selection_uses_validation_and_reuses_exact_side_policies():
    r = runner()
    p = protocol()
    calls = []
    attempts = []
    rows = [{'decision_time': '123'}]
    predictions = [(0.2, -0.2)]

    def replay(passed_rows, passed_predictions, tape, **kwargs):
        assert passed_rows is rows and passed_predictions is predictions
        calls.append(kwargs['threshold_bps'])
        active = kwargs['threshold_bps'] < 0.2
        return SimpleNamespace(metrics={'net_pnl': 1.0 if active else 0.0, 'filled_entries': int(active),
                                        'final_inventory_mark_notional': 0.0})

    with patch.object(r, 'evaluate_policy', replay):
        threshold, metrics = r.select_threshold(rows, predictions, [], 0.0, p, attempts.append)
    assert threshold == 0.1  # Same profitable side decisions; predeclared higher-threshold tie break.
    assert metrics['net_pnl'] == 1.0
    assert calls == [math.inf, 0.0]
    assert len(attempts) == 8
    assert sum(a['equivalent_policy_cache'] for a in attempts) == 6


def test_forecast_metrics_reject_silent_zip_truncation():
    r = runner()
    with pytest.raises(ValueError, match='aligned nonempty'):
        r.forecast_metrics([{}], [])
