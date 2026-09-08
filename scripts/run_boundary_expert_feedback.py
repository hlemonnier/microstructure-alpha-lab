"""Bounded synthetic preflight and exposed-date delayed expert-mixture study."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_adaptive_priors import forecast_priors
from lob_forge.boundary_expert_feedback import CASES, mix_experts
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_prequential import decoded_release_clock
from run_boundary_confirmation import save_predictions, write_json
from run_boundary_event_clock_screen import check_completed

ROOT = Path(__file__).resolve().parents[1]


def memory_guard(process, folder, limits, stop, readings):
    while not stop.is_set():
        rss = int(process.memory_info().rss)
        readings['maximum_sampled_rss_bytes'] = max(readings['maximum_sampled_rss_bytes'], rss)
        readings['maximum_sampled_simultaneous_sum_bytes'] = readings['maximum_sampled_rss_bytes']
        if rss > limits['worker_rss_bytes']:
            write_json(folder / 'resource_failure.json', {'reason': 'registered_small_process_rss', 'rss_bytes': rss})
            os._exit(3)
        stop.wait(limits['memory_sample_seconds'])


def checked_trajectory(args, case, folder, limits):
    folder.mkdir(parents=True)
    def deadline_failure():
        write_json(folder / 'resource_failure.json', {'reason': 'registered_trajectory_deadline'})
        os._exit(3)
    timer = threading.Timer(limits['trajectory_seconds'], deadline_failure)
    timer.daemon = True
    timer.start()
    begin = time.monotonic()
    result = mix_experts(*args, case=case)
    replay = mix_experts(*args, case=case)
    for key in result:
        np.testing.assert_array_equal(result[key], replay[key])
    limit = 128
    changed = [a.copy() for a in args]
    unavailable = args[4] > args[3][limit - 1]
    changed[2][unavailable] = (changed[2][unavailable] + 2) % 3 - 1
    changed[0][limit:] = np.roll(changed[0][limit:], 1, axis=2)
    changed[1][limit:] = np.roll(changed[1][limit:], 1, axis=2)
    altered = mix_experts(*changed, case=case)
    short = mix_experts(*(a[:limit] for a in args[:5]), args[5], case=case)
    for key in ("probabilities", "weights", "updates", "contexts"):
        np.testing.assert_array_equal(result[key][:limit], altered[key][:limit])
        np.testing.assert_array_equal(result[key][:limit], short[key])
    path = folder / 'trajectory.npz'
    np.savez_compressed(path, **result)
    with np.load(path, allow_pickle=False) as restored:
        for key in result:
            np.testing.assert_array_equal(result[key], restored[key])
    elapsed = time.monotonic() - begin
    if elapsed > limits['trajectory_seconds']:
        raise ValueError('Registered coefficient-only trajectory deadline exceeded')
    timer.cancel()
    record = {'case': case, 'seconds': elapsed, 'rows': len(args[0]),
        'released_loss_vector_updates': int(result['updates'].sum()),
        'maximum_expert_weight': float(result['weights'].max()),
        'mean_reference_weight': float(result['weights'][:, -1].mean()),
        'repeat_saved_and_prefix_replay_exact': True, 'future_forecasts_and_unreleased_labels_prefix_exact': True,
        'trajectory_sha256': sha256_file(path)}
    write_json(folder / 'operational.json', record)
    return result, record


def preflight(protocol, output):
    rng = np.random.default_rng(20260908)
    p = rng.dirichlet([1, 2, 1], size=(7070, 9))
    pi = np.tile([.2, .5, .3], (7070, 9, 1))
    y = rng.choice([-1, 0, 1], 7070, p=[.2, .5, .3])
    times = 1_600_000_000_000 + 1000 * np.arange(7070)
    releases = times + rng.integers(5100, 12001, 7070)
    args = (p, pi, y, times, releases, np.array([.2, .5, .3]))
    records = []
    for case in CASES:
        _, record = checked_trajectory(args, case, output / case, protocol['limits'])
        records.append(record)
        print(f'expert_feedback_synthetic={case} seconds={record["seconds"]:.3f}', flush=True)
    return {'evidence_status': 'synthetic_runtime_only', 'all_cases_passed': True, 'base_model_fits': 0,
        'market_assessment_metrics_computed': False, 'records': records}


def market(protocol, output):
    parent = ROOT / protocol['source_run']
    parent_identity = json.loads((parent / 'frozen_screen.json').read_text())
    source_manifest = json.loads((ROOT / protocol['manifest']).read_text())
    pre = json.loads((ROOT / protocol['preflight_summary']).read_text())
    if not pre['all_cases_passed'] or len(pre['records']) != len(CASES):
        raise ValueError('Every full-length small synthetic case must pass first')
    original_summary = json.loads((parent / 'summary.json').read_text())
    panels, operations = [], []
    for day in protocol['dates']:
        source = parent / 'dates' / day
        check_completed(source, parent_identity)
        for symbol in protocol['symbols']:
            folder = output / 'panels' / day / symbol
            ps, pis, y, times, prior = [], [], None, None, None
            for name in protocol['models']:
                with np.load(source / 'predictions' / f'{symbol}_{name}_forecast_3600.npz', allow_pickle=False) as saved:
                    if y is None:
                        y, times, prior = (saved[k].copy() for k in ('labels', 'decision_times', 'train_priors'))
                    for key, expected in [('labels', y), ('decision_times', times), ('train_priors', prior)]:
                        np.testing.assert_array_equal(saved[key], expected)
                    ps.append(saved['probabilities'].copy())
                    pis.append(np.broadcast_to(saved['decision_priors'], saved['probabilities'].shape).copy())
            if len(y) != 7070:
                raise ValueError('Every original noon assessment row must remain')
            raw = next(r for r in source_manifest['sessions'] if (r['symbol'], r['session_date']) == (symbol, day))
            outcomes = pd.read_parquet(ROOT / raw['features_path'], columns=['decision_time', 'label', 'future_event_time']).set_index('decision_time').loc[times]
            np.testing.assert_array_equal(outcomes.label.to_numpy(), y)
            releases = decoded_release_clock(outcomes.future_event_time.to_numpy())
            if (releases < times + 5100).any():
                raise ValueError('Use actual exact target releases, never premature labels')
            with np.load(ROOT / protocol['prior_run'] / 'priors' / day / f'{symbol}_original_reference.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['decision_times'], times)
                initial = saved['labels_cumulative'][0].copy()
            args = (np.stack(ps, axis=1), np.stack(pis, axis=1), y, times, releases, prior)
            for case in CASES:
                result, record = checked_trajectory(args, case, folder / case, protocol['limits'])
                operations.append({'date': day, 'symbol': symbol, **record})
                for policy in protocol['decision_policies']:
                    pi = prior if policy == 'registered' else forecast_priors(result['probabilities'], times, initial, half_life_seconds=3600)
                    save_predictions(folder / case / f'{policy}.npz', result['probabilities'], y, times, prior, pi)
                    if case == 'frozen_reference':
                        with np.load(source / 'predictions' / f'{symbol}_{protocol["reference"]}_{policy}.npz', allow_pickle=False) as saved:
                            np.testing.assert_array_equal(saved['probabilities'], result['probabilities'])
                            np.testing.assert_array_equal(saved['decision_priors'], pi)
            completion = {'date': day, 'symbol': symbol, 'source_rows_labels_clocks_and_priors_exact': True,
                'frozen_reference_both_policies_exact': True, 'minimum_release_delay_ms': int((releases - times).min()),
                'artifact_hashes': {str(p.relative_to(folder)): sha256_file(p) for p in folder.rglob('*') if p.is_file()}}
            write_json(folder / 'completed.json', completion)
            panels.append(folder)
            print(f'expert_feedback_panels_frozen={len(panels)}/6', flush=True)
    records = []
    for folder in panels:
        completed = json.loads((folder / 'completed.json').read_text())
        for path, checksum in completed['artifact_hashes'].items():
            if sha256_file(folder / path) != checksum:
                raise ValueError('Completed expert trajectory changed before score reveal')
        for case in CASES:
            for policy in protocol['decision_policies']:
                with np.load(folder / case / f'{policy}.npz', allow_pickle=False) as saved:
                    metrics = classification_metrics(SimpleNamespace(priors=saved['decision_priors']), saved['probabilities'], saved['labels'])
                row = {'date': completed['date'], 'symbol': completed['symbol'], 'case': case, 'policy': policy, 'metrics': metrics}
                if case == 'frozen_reference':
                    expected = next(r['metrics'] for r in original_summary['records'] if (r['date'], r['symbol'], r['model'], r['policy'])
                        == (row['date'], row['symbol'], protocol['reference'], policy))
                    if metrics != expected:
                        raise ValueError('Frozen reference metrics must reproduce exactly')
                records.append(row)
    if len(records) != protocol['post_fit_panels']:
        raise ValueError('Incomplete fixed expert-feedback comparison family')
    board = []
    for case in CASES:
        for policy in protocol['decision_policies']:
            rows = [r for r in records if (r['case'], r['policy']) == (case, policy)]
            board.append({'case': case, 'policy': policy,
                'means': {k: float(np.mean([r['metrics'][k] for r in rows])) for k in ('balanced_accuracy', 'natural_accuracy', 'log_loss', 'macro_f1')},
                'balanced_accuracy_by_asset': {s: float(np.mean([r['metrics']['balanced_accuracy'] for r in rows if r['symbol'] == s])) for s in protocol['symbols']}})
    return {'evidence_status': protocol['evidence_status'], 'base_model_fits': 0, 'coefficient_adaptation_trajectories': 48,
        'assessment_rows_per_source': 42420, 'substantial_gain_confirmed': False, 'all_six_panels_frozen_before_score_reveal': True,
        'operations': operations, 'records': records, 'leaderboard': sorted(board, key=lambda r: r['means']['balanced_accuracy'], reverse=True)}


def run(protocol_path, output, synthetic):
    import psutil
    from threadpoolctl import threadpool_limits

    protocol = json.loads(protocol_path.read_text())
    if protocol['cases'] != list(CASES) or output.exists():
        raise ValueError('Use the complete fixed family and preserve every earlier attempt')
    expected_status = 'synthetic_runtime_only' if synthetic else 'development_only_on_exposed_dates'
    if protocol['evidence_status'] != expected_status:
        raise ValueError('Synthetic and market protocols cannot be interchanged')
    for path, checksum in protocol['input_hashes'].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f'Frozen expert-feedback input changed: {path}')
    output.mkdir(parents=True)
    write_json(output / 'attempt.json', {'protocol_sha256': sha256_file(protocol_path), 'synthetic': synthetic})
    readings = {'maximum_sampled_rss_bytes': 0, 'maximum_sampled_gpu_driver_bytes': 0, 'maximum_sampled_simultaneous_sum_bytes': 0}
    stop = threading.Event()
    process = psutil.Process()
    process.memory_info()
    guard = threading.Thread(target=memory_guard, args=(process, output, protocol['limits'], stop, readings), daemon=True)
    guard.start()
    begin = time.monotonic()
    with threadpool_limits(limits=1):
        result = preflight(protocol, output) if synthetic else market(protocol, output)
    stop.set()
    guard.join()
    if readings['maximum_sampled_rss_bytes'] <= 0:
        raise ValueError('The registered small-process memory measurement must succeed')
    result.update(protocol_sha256=sha256_file(protocol_path), seconds=time.monotonic() - begin, memory=readings)
    result['artifact_hashes'] = {str(p.relative_to(output)): sha256_file(p) for p in output.rglob('*') if p.is_file()}
    write_json(output / 'summary.json', result)
    print(f'expert_feedback_complete {output / "summary.json"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve(), args.synthetic)
