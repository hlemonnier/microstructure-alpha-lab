"""Verify cached broad-audit inputs against original models without refitting."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_confirm_model import SYMBOLS
from lob_forge.boundary_pooled import PooledForecaster
from run_boundary_broad_accuracy_screen import prepare
from run_boundary_confirmation import write_json
from run_boundary_event_clock_screen import check_completed

ROOT = Path(__file__).resolve().parents[1]


def run(protocol_path, output):
    import torch
    from threadpoolctl import threadpool_limits

    protocol = json.loads(protocol_path.read_text())
    if output.exists() or protocol['market_model_fits'] != 0 or protocol['evidence_status'] != 'data_and_frozen_model_replay_only':
        raise ValueError('Preserve every existing replay and keep this audit free of fitting')
    for path, checksum in protocol['input_hashes'].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f'Frozen original-input replay source changed: {path}')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    identity = {'protocol_sha256': sha256_file(protocol_path), 'input_hashes': protocol['input_hashes']}
    write_json(output / 'identity.json', identity)
    begin, records = time.monotonic(), []
    with threadpool_limits(limits=2):
        for day in protocol['dates']:
            folder = output / 'dates' / day
            prepare(day, protocol, folder, identity)
            training = joblib.load(folder / 'training.joblib')
            queries = joblib.load(folder / 'queries.joblib')
            retrieval = ROOT / protocol['retrieval_run']
            check_completed(retrieval / 'dates' / day, json.loads((retrieval / 'frozen_screen.json').read_text()))
            expected = joblib.load(retrieval / 'dates' / day / 'observations/training.joblib')
            parent = ROOT / protocol['context_run'] / 'dates' / day
            model = PooledForecaster.load(parent / 'observations_neural')
            tree = joblib.load(parent / 'observations_tree.joblib')
            rows = []
            for asset, symbol in enumerate(SYMBOLS):
                selected = expected['assets'] == asset
                np.testing.assert_array_equal(training['labels'][symbol], expected['labels'][selected])
                np.testing.assert_array_equal(training['times'][symbol], expected['times'][selected])
                normalized = model.matrix(training['features'][symbol], asset)
                np.testing.assert_array_equal(normalized, expected['matrix'][selected])
                np.testing.assert_array_equal(model.matrix(training['validation_features'][symbol], asset), expected['validation'][asset])
                np.testing.assert_array_equal(training['validation_labels'][symbol], expected['validation_labels'][asset])
                np.testing.assert_array_equal(model.priors[asset], [(training['labels'][symbol] == c).mean() for c in (-1, 0, 1)])
                forecasts = {'observations_neural': model.predict_proba(queries[symbol], asset),
                    'observations_tree': tree.predict_proba(queries[symbol], symbol)}
                for name, probability in forecasts.items():
                    with np.load(retrieval / 'dates' / day / 'predictions' / f'{symbol}_{name}_registered.npz', allow_pickle=False) as saved:
                        np.testing.assert_array_equal(probability, saved['probabilities'])
                        np.testing.assert_array_equal(saved['train_priors'], model.priors[asset])
                rows.append({'symbol': symbol, 'training_rows': len(normalized), 'validation_rows': len(expected['validation'][asset]),
                    'query_rows': len(queries[symbol]), 'normalized_dimensions': normalized.shape[1],
                    'training_normalized_matrix_labels_clocks_exact': True, 'validation_matrix_and_labels_exact': True,
                    'neural_and_tree_full_query_forecasts_exact': True, 'original_training_priors_exact': True})
                del normalized, forecasts
            record = {'date': day, 'checks': rows, 'market_model_fits': 0, 'new_assessment_metrics_computed': False}
            write_json(folder / 'replay_checks.json', record)
            records.append(record)
            del training, queries, expected, model, tree
            gc.collect()
            print(f'broad_anchor_replay_verified={len(records)}/{len(protocol["dates"])}', flush=True)
    summary = {'identity': identity, 'evidence_status': protocol['evidence_status'], 'market_model_fits': 0,
        'new_assessment_metrics_computed': False, 'seconds': time.monotonic() - begin, 'records': records,
        'training_parameter_reproduction_claimed': False,
        'scope': 'Original preprocessing and inference applied to the broad cache. No neural or tree retraining; exact retraining parity is still required by the future twenty-date audit.',
        'artifact_hashes': {str(p.relative_to(output)): sha256_file(p) for p in output.rglob('*') if p.is_file()}}
    write_json(output / 'summary.json', summary)
    print(f'broad_anchor_replay_complete {output / "summary.json"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())
