"""Execute the frozen, chronological expected-payoff ablation on verified real data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from lob_forge.binance_vision import sha256_file
from lob_forge.features import FEATURE_SEMANTICS_VERSION
from lob_forge.performance_models import fit_performance_model, gross_edge_targets_bps
from lob_forge.performance_replay import evaluate_policy, load_quote_tape

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def day_start(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000)


def session_time(day: str, clock: str) -> int:
    return int(datetime.fromisoformat(f'{day}T{clock}').replace(tzinfo=timezone.utc).timestamp() * 1000)


def session_splits(dates: list[str], protocol: dict) -> list[dict]:
    split = protocol['split']
    train_count, val_count = split['training_sessions'], split['validation_sessions']
    if split['test_sessions'] != 1 or split['step_sessions'] != 1:
        raise ValueError('pilot runner requires one-session nonoverlapping test steps')
    if dates != sorted(set(dates)):
        raise ValueError('session dates must be sorted and unique')
    if set(dates) & set(protocol['data']['reserve_unseen_dates']):
        raise ValueError('reserved unseen dates cannot enter this pilot')
    result = []
    for index in range(train_count + val_count, len(dates)):
        result.append({'train_dates': dates[index-val_count-train_count:index-val_count],
                       'validation_dates': dates[index-val_count:index], 'test_date': dates[index]})
    if [item['test_date'] for item in result] != split['test_dates']:
        raise ValueError('actual OOS dates do not match the frozen protocol')
    if val_count != 1:
        raise ValueError('this bounded runner requires one validation session')
    return result


def load_feature_rows(session: dict, protocol: dict) -> list[dict[str, str]]:
    path = Path(session['features_path'])
    expected = session.get('feature_sha256') or session['feature']['sha256']
    if sha256_file(path) != expected:
        raise ValueError(f'feature hash mismatch: {path}')
    start = session_time(session['session_date'], protocol['data']['decision_start_utc'])
    stop = (session_time(session['session_date'], protocol['data']['decision_end_utc'])
            - protocol['execution']['horizon_ms'] - protocol['execution']['stress_latency_ms'])
    rows = []
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row.get('feature_semantics_version') != FEATURE_SEMANTICS_VERSION:
                raise ValueError('stale feature semantics')
            decision = int(row['decision_time'])
            # Eligibility depends only on the clock and the observed decision quote.
            # Future quote lag and realized targets cannot select test decisions.
            if not start <= decision < stop:
                continue
            if int(row['quote_age_ms']) > protocol['features']['decision_quote_max_age_ms']:
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f'no causally eligible rows in {path}')
    if any(int(a['decision_time']) >= int(b['decision_time']) for a,b in zip(rows, rows[1:])):
        raise ValueError('feature decisions must be strictly chronological')
    return rows


def purge_before(rows: list[dict[str, str]], next_start_ms: int) -> list[dict[str, str]]:
    return [row for row in rows if int(row['future_event_time']) < next_start_ms]


def forecast_metrics(rows: list[dict[str, str]], predictions: list[tuple[float, float]]) -> dict:
    if len(rows) != len(predictions) or not rows:
        raise ValueError('forecast metrics require aligned nonempty rows and predictions')
    target = gross_edge_targets_bps(rows)
    return {'rows': len(rows),
            'long_rmse_bps': math.sqrt(math.fsum((p[0]-t[0])**2 for p,t in zip(predictions,target))/len(rows)),
            'short_rmse_bps': math.sqrt(math.fsum((p[1]-t[1])**2 for p,t in zip(predictions,target))/len(rows)),
            'max_entry_lag_ms': max(int(row['entry_lag_ms']) for row in rows),
            'max_future_lag_ms': max(int(row['future_lag_ms']) for row in rows)}


def rank_validation(metrics: dict, threshold: float) -> tuple:
    if metrics['final_inventory_mark_notional'] > 1e-6:
        return (-math.inf, -metrics['filled_entries'], threshold)
    if not math.isfinite(metrics['net_pnl']):
        raise ValueError('non-finite validation PnL')
    return (round(metrics['net_pnl'], 10), -metrics['filled_entries'], threshold)


def side_key(predictions: list[tuple[float,float]], fee: float, threshold: float) -> str:
    sides = bytearray()
    for long, short in predictions:
        long_net = (1-fee/10000)*long-2*fee
        short_net = (1+fee/10000)*short-2*fee
        sides.append(2 if long_net > threshold and long_net >= short_net
                     else 0 if short_net > threshold and short_net > long_net else 1)
    return hashlib.sha256(sides).hexdigest()


def write_predictions(path: Path, rows: list[dict[str,str]], predictions: list[tuple[float,float]]) -> None:
    target = gross_edge_targets_bps(rows)
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['decision_time','predicted_long_gross_bps','predicted_short_gross_bps',
                         'observed_long_gross_bps','observed_short_gross_bps'])
        for row, prediction, actual in zip(rows,predictions,target):
            writer.writerow([row['decision_time'], *prediction, *actual])


def write_dataclasses(path: Path, objects: list) -> None:
    if not objects:
        path.write_text('')
        return
    with path.open('w', newline='') as handle:
        first=asdict(objects[0])
        writer=csv.DictWriter(handle,fieldnames=list(first))
        writer.writeheader()
        writer.writerows(asdict(item) for item in objects)


def validate_model_config(metadata: dict, protocol: dict) -> None:
    declared = protocol['models']
    actual = metadata['config']
    if metadata['name'].startswith('ridge_'):
        if actual['l2'] != declared['ridge_l2']:
            raise ValueError('ridge regularization differs from the frozen protocol')
        return
    for name, value in declared['hgb'].items():
        expected = max(100, int(0.005 * metadata['training_rows'])) if name == 'min_samples_leaf' else value
        if actual[name] != expected:
            raise ValueError(f'boosting parameter {name} differs from the frozen protocol')
    if actual['random_state'] != declared['seed']:
        raise ValueError('boosting seed differs from the frozen protocol')


def select_threshold(rows, predictions, tape, fee, protocol, append_attempt):
    cfg = protocol['execution']
    candidates=[]
    cache={}
    for threshold in [math.inf, *protocol['selection']['edge_thresholds_bps']]:
        key=side_key(predictions,fee+cfg['slippage_bps_per_side'],threshold)
        reused=key in cache
        if not reused:
            result=evaluate_policy(rows,predictions,tape,threshold_bps=threshold,taker_fee_bps=fee,
                                   latency_ms=cfg['primary_latency_ms'],target_notional=cfg['target_notional_usdt'],
                                   initial_cash=cfg['initial_cash_usdt'],horizon_ms=cfg['horizon_ms'],
                                   decision_stride_ms=cfg['decision_stride_ms'],slippage_bps=cfg['slippage_bps_per_side'])
            cache[key]=dict(result.metrics)
        metrics={**cache[key], 'threshold_bps': threshold if math.isfinite(threshold) else None,
                 'policy': 'always_flat' if math.isinf(threshold) else 'expected_payoff_threshold'}
        candidates.append((threshold,metrics))
        append_attempt({'stage':'validation_threshold','fee_bps':fee,'threshold_bps':metrics['threshold_bps'],
                        'policy_hash':key,'equivalent_policy_cache':reused,'metrics':metrics})
    return max(candidates,key=lambda item:rank_validation(item[1],item[0]))


def aggregate_results(records: list[dict], protocol: dict) -> dict:
    output=[]
    primary={}
    for model in protocol['models']['names']:
        for fee in protocol['execution']['fee_scenarios_bps_per_side']:
            for latency in [protocol['execution']['primary_latency_ms'],protocol['execution']['stress_latency_ms']]:
                subset=[r for r in records if r['model']==model and r['fee_bps']==fee and r['latency_ms']==latency]
                if not subset:
                    continue
                net=math.fsum(r['test']['net_pnl'] for r in subset)
                turnover=math.fsum(r['test']['turnover'] for r in subset)
                row={'model':model,'fee_bps':fee,'latency_ms':latency,'asset_sessions':len(subset),
                     'distinct_dates':len({r['test_date'] for r in subset}), 'net_pnl':net,'turnover':turnover,
                     'filled_entries':sum(r['test']['filled_entries'] for r in subset),
                     'break_even_fee_bps_on_this_policy':fee+net/turnover*10000 if turnover else None,
                     'positive_asset_sessions':sum(r['test']['net_pnl']>1e-10 for r in subset),
                     'worst_session_drawdown':max(r['test']['max_drawdown_pnl'] for r in subset),
                     'residual_sessions':sum(r['test']['final_inventory_mark_notional']>1e-6 for r in subset),
                     'by_asset':{symbol:math.fsum(r['test']['net_pnl'] for r in subset if r['symbol']==symbol)
                                 for symbol in protocol['data']['symbols']},
                     'by_date':{day:math.fsum(r['test']['net_pnl'] for r in subset if r['test_date']==day)
                                for day in protocol['split']['test_dates']}}
                row['valid_for_comparison'] = row['residual_sessions'] == 0
                row['failure_reason'] = None if row['valid_for_comparison'] else 'unclosed_session_inventory'
                output.append(row)
                if fee==protocol['execution']['primary_fee_bps_per_side'] and latency==protocol['execution']['primary_latency_ms']:
                    primary[model]=row
    baseline=primary.get('ridge_default')
    paired=[]
    if baseline:
        for model,row in primary.items():
            differences={day:row['by_date'][day]-baseline['by_date'][day] for day in baseline['by_date']}
            paired.append({'model':model,'total_improvement_usdt':row['net_pnl']-baseline['net_pnl'],
                           'valid_for_comparison':row['valid_for_comparison'] and baseline['valid_for_comparison'],
                           'paired_date_differences':differences,
                           'positive_difference_dates':sum(v>1e-10 for v in differences.values()),
                           'inference_status':'descriptive_only_three_distinct_OOS_dates'})
    return {'scenario_totals':output,'primary_paired_comparison':paired,
            'always_flat_net_pnl':0.0,'empirical_promotion':False,
            'limits':['three distinct test dates','two fixed UTC hours per session',
                      'assumed fees, latency and zero extra slippage','fractional research order sizes',
                      'top-of-book liquidity, unmodeled market impact and receive-time delays']}


def run(protocol_path: Path, dataset_path: Path, output: Path) -> dict:
    import joblib
    protocol=json.loads(protocol_path.read_text())
    dataset=json.loads(dataset_path.read_text())
    protocol_sha=sha256_file(protocol_path)
    if dataset['acquisition_identity']['protocol_sha256'] != protocol_sha:
        raise ValueError('dataset does not match this predeclared protocol')
    files=['scripts/run_performance_pilot.py','src/lob_forge/performance_models.py','src/lob_forge/performance_replay.py',
           'src/lob_forge/edge_model.py','src/lob_forge/logistic.py','src/lob_forge/baselines.py',
           'src/lob_forge/execution_sim.py','src/lob_forge/features.py']
    identity={'protocol_sha256':protocol_sha,'dataset_manifest_sha256':sha256_file(dataset_path),
              'code_hashes':{name:sha256_file(ROOT/name) for name in files},'python':platform.python_version(),
              'dependencies':{name:version(name) for name in ['numpy','scikit-learn','joblib']}}
    output.mkdir(parents=True,exist_ok=True)
    frozen_path=output/'frozen_run.json'
    if frozen_path.exists() and json.loads(frozen_path.read_text())!=identity:
        raise ValueError('existing run has a different frozen data/code/protocol identity; retain it and use a new output directory')
    write_json(frozen_path,identity)
    index={(s['symbol'],s['session_date']):s for s in dataset['sessions']}
    if len(index) != len(dataset['sessions']):
        raise ValueError('duplicate symbol/session data')
    dates=sorted({s['session_date'] for s in dataset['sessions']})
    splits=session_splits(dates,protocol)
    if set(index) != {(symbol,day) for symbol in protocol['data']['symbols'] for day in dates}:
        raise ValueError('incomplete or extra symbol/session data')
    all_records=[]
    for symbol in protocol['data']['symbols']:
        feature_rows={day:load_feature_rows(index[symbol,day],protocol) for day in dates}
        for split in splits:
            validation_date=split['validation_dates'][0]
            test_date=split['test_date']
            train=purge_before([row for day in split['train_dates'] for row in feature_rows[day]],day_start(validation_date))
            validation=purge_before(feature_rows[validation_date],day_start(test_date))
            test=feature_rows[test_date]
            tapes={}
            for day in [validation_date,test_date]:
                session=index[symbol,day]
                path=Path(session['book_ticker_path'])
                expected=session.get('book_ticker_sha256') or session['prefixes']['bookTicker']['sha256']
                if sha256_file(path)!=expected:
                    raise ValueError('raw tape hash mismatch')
                tapes[day]=load_quote_tape(path,start_ms=session_time(day,protocol['data']['slice_start_utc']),
                                          end_ms=session_time(day,protocol['data']['decision_end_utc']))
            for model_name in protocol['models']['names']:
                start=time.monotonic()
                job_path=output/symbol/test_date/model_name
                job_path.mkdir(parents=True,exist_ok=True)
                complete=job_path/'completed.json'
                if complete.exists():
                    saved=json.loads(complete.read_text())
                    if saved['run_identity_sha256']!=canonical_sha(identity):
                        raise ValueError('completed model job has mismatched provenance')
                    for name,digest in saved['artifact_hashes'].items():
                        if sha256_file(job_path/name)!=digest:
                            raise ValueError('completed model job artifact hash mismatch')
                    all_records.extend(saved['records'])
                    print(f'reused symbol={symbol} test_date={test_date} model={model_name}',flush=True)
                    continue
                with (job_path/'attempt_registry.jsonl').open('a') as attempts:
                    def record_attempt(value):
                        attempts.write(json.dumps({'symbol':symbol,'test_date':test_date,'model':model_name,
                                                   **value},allow_nan=False,sort_keys=True)+'\n')
                        attempts.flush()
                    record_attempt({'stage':'fit_started','train_rows':len(train),'validation_rows':len(validation),'test_rows':len(test)})
                    model=fit_performance_model(model_name,train)
                    validate_model_config(model.metadata(), protocol)
                    joblib.dump(model,job_path/'model.joblib')
                    checkpoint = joblib.load(job_path/'model.joblib')
                    parity_rows = validation[:128]
                    if checkpoint.predict_gross_edges(parity_rows) != model.predict_gross_edges(parity_rows):
                        raise ValueError('saved model does not reproduce in-memory validation predictions')
                    model = checkpoint
                    record_attempt({'stage':'checkpoint_reload_verified','validation_prefix_rows':len(parity_rows)})
                    write_json(job_path/'model_metadata.json',{**model.metadata(),**split,'symbol':symbol})
                    val_predictions=model.predict_gross_edges(validation)
                    test_predictions=model.predict_gross_edges(test)
                    write_predictions(job_path/'validation_predictions.csv',validation,val_predictions)
                    write_predictions(job_path/'test_predictions.csv',test,test_predictions)
                    diagnostics={'validation':forecast_metrics(validation,val_predictions),'test':forecast_metrics(test,test_predictions)}
                    cfg=protocol['execution']
                    records=[]
                    for fee in cfg['fee_scenarios_bps_per_side']:
                        threshold,validation_metrics=select_threshold(validation,val_predictions,tapes[validation_date],fee,protocol,record_attempt)
                        for latency in [cfg['primary_latency_ms'],cfg['stress_latency_ms']]:
                            evaluated=evaluate_policy(test,test_predictions,tapes[test_date],threshold_bps=threshold,
                                taker_fee_bps=fee,latency_ms=latency,target_notional=cfg['target_notional_usdt'],
                                initial_cash=cfg['initial_cash_usdt'],horizon_ms=cfg['horizon_ms'],
                                decision_stride_ms=cfg['decision_stride_ms'],slippage_bps=cfg['slippage_bps_per_side'],
                                policy_fee_bps=fee,include_ledgers=True)
                            tag=f'fee_{fee:g}_latency_{latency}'
                            write_json(job_path/f'{tag}_signals.json',evaluated.signal_ledger)
                            for name,values in [('orders',evaluated.simulation.orders),('fills',evaluated.simulation.fills),
                                                ('equity',evaluated.simulation.positions)]:
                                write_dataclasses(job_path/f'{tag}_{name}.csv',values)
                            result={'symbol':symbol,'model':model_name,**split,'fee_bps':fee,'latency_ms':latency,
                                    'threshold_bps':threshold if math.isfinite(threshold) else None,
                                    'validation':validation_metrics,'test':evaluated.metrics,'forecast':diagnostics}
                            record_attempt({'stage':'OOS_evaluated','result':result})
                            records.append(result)
                    write_json(complete,{'run_identity_sha256':canonical_sha(identity),'records':records,
                        'duration_seconds':time.monotonic()-start,
                        'artifact_hashes':{path.name:sha256_file(path) for path in job_path.iterdir()
                                           if path.is_file() and path.name!='completed.json'}})
                    all_records.extend(records)
                    main=[r for r in records if r['fee_bps']==cfg['primary_fee_bps_per_side'] and r['latency_ms']==cfg['primary_latency_ms']][0]
                    print(f"completed symbol={symbol} test_date={test_date} model={model_name} seconds={time.monotonic()-start:.1f} primary_test_pnl={main['test']['net_pnl']:.8f}",flush=True)
                write_json(output/'progress.json',{'completed_jobs':len(all_records)//12,'summary':aggregate_results(all_records,protocol)})
    summary={**aggregate_results(all_records,protocol),'records':all_records,'run_identity':identity,
             'completed_jobs':len(all_records)//12,'completed_at_utc':datetime.now(timezone.utc).isoformat()}
    write_json(output/'summary.json',summary)
    print(f'completed_pilot output={output/"summary.json"} jobs={summary["completed_jobs"]}',flush=True)
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/research/performance_pilot_20260907_window_revision.json')
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/research/performance_pilot_20260907_window_revision/dataset_manifest.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results/performance_pilot_20260907_window_revision')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if not args.run:
        print(json.dumps({'protocol':str(args.protocol),'dataset':str(args.dataset),'output':str(args.output),
                          'mode':'dry_run','requires':'--run'},indent=2))
        return
    run(args.protocol,args.dataset,args.output)


if __name__=='__main__':
    main()
