"""Reserve prospective confirmation archives as checksum-verified opaque bytes."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from lob_forge.binance_vision import archive_key, sha256_file, url_for_key

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def object_metadata(record):
    key = record['key']
    with urllib.request.urlopen(url_for_key(key + '.CHECKSUM'), timeout=30) as response:
        checksum_body = response.read(4096).decode('ascii').strip()
    parts = checksum_body.split()
    if not parts or not re.fullmatch(r'[0-9a-fA-F]{64}', parts[0]):
        raise ValueError('A valid official SHA256 is mandatory before acquisition')
    if len(parts) > 1 and parts[-1].lstrip('*') != Path(key).name:
        raise ValueError('Official checksum filename must match the requested archive')
    request = urllib.request.Request(url_for_key(key), method='HEAD')
    with urllib.request.urlopen(request, timeout=30) as response:
        size = int(response.headers['Content-Length'])
        modified = response.headers.get('Last-Modified')
    if size <= 0:
        raise ValueError('A positive remote archive byte length is required')
    return {**record, 'url': url_for_key(key), 'checksum_url': url_for_key(key + '.CHECKSUM'),
        'official_sha256': parts[0].lower(), 'official_checksum_body': checksum_body,
        'bytes': size, 'last_modified': modified, 'status': 'metadata_verified'}


def download_opaque(record, folder):
    destination = folder / 'raw' / record['key']
    partial = destination.with_name(destination.name + '.download')
    if destination.exists() or partial.exists():
        raise ValueError('Preserve any previous archive attempt; no implicit replacement')
    destination.parent.mkdir(parents=True, exist_ok=True)
    log = destination.with_name(destination.name + '.curl.log')
    begin = time.monotonic()
    command = ['curl', '--fail', '--location', '--silent', '--show-error', '--retry', '2',
        '--connect-timeout', '15', '--max-time', '300', '--output', str(partial), record['url']]
    with log.open('w') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=930)
    if result.returncode:
        raise ValueError(f'Archive transfer failed with code {result.returncode}; preserve partial bytes and log')
    actual_bytes, actual_sha256 = partial.stat().st_size, sha256_file(partial)
    if actual_bytes != record['bytes'] or actual_sha256 != record['official_sha256']:
        raise ValueError('Archive bytes disagree with the frozen official identity; preserve the partial file')
    partial.rename(destination)
    return {**record, 'status': 'opaque_archive_verified', 'local_path': str(destination.relative_to(ROOT)),
        'actual_sha256': actual_sha256, 'actual_bytes': actual_bytes, 'seconds': time.monotonic() - begin,
        'archive_contents_opened': False}


def run(protocol_path, output, stage):
    protocol = json.loads(protocol_path.read_text())
    if output.exists() or stage != protocol['stage']:
        raise ValueError('Use the matching frozen acquisition stage and preserve earlier attempts')
    for path, checksum in protocol['input_hashes'].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f'Frozen reserve protocol input changed: {path}')
    inventory = [{'date': day, 'symbol': symbol, 'dataset': dataset,
        'key': archive_key(market='futures/um', frequency='daily', dataset=dataset, symbol=symbol, date_value=day)}
        for day in protocol['dates'] for symbol in protocol['symbols'] for dataset in protocol['datasets']]
    if len(inventory) != 80 or len({r['key'] for r in inventory}) != 80 or protocol['workers'] != 2:
        raise ValueError('Exactly eighty registered archives with two bounded transfer workers required')
    if stage == 'download':
        metadata = json.loads((ROOT / protocol['metadata_summary']).read_text())
        if (not metadata['all_objects_verified'] or len(metadata['records']) != 80
            or [r['key'] for r in metadata['records']] != [r['key'] for r in inventory]):
            raise ValueError('Every reserved archive must have the frozen complete metadata inventory')
        inventory = metadata['records']
        if sum(r['bytes'] for r in inventory) > protocol['maximum_total_archive_bytes']:
            raise ValueError('Reserved archive collection exceeds its prospective byte limit')
    output.mkdir(parents=True)
    identity = {'protocol_sha256': sha256_file(protocol_path), 'stage': stage,
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'raw_archive_contents_opened': False,
        'features_labels_models_or_assessment_scores_created': False}
    write_json(output / 'identity.json', identity)
    records, begin = [], time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(object_metadata, record) if stage == 'metadata' else pool.submit(download_opaque, record, output): record for record in inventory}
        for future in concurrent.futures.as_completed(futures):
            original = futures[future]
            try:
                record = future.result()
            except Exception as error:
                record = {**original, 'status': 'failed', 'error': repr(error), 'archive_contents_opened': False}
            records.append(record)
            write_json(output / 'progress.json', {'stage': stage, 'records': records,
                'raw_archive_contents_opened': False, 'assessment_scores_computed': False})
            print(f'confirmation_reserve_{stage}_objects={len(records)}/80 failed={sum(r["status"] == "failed" for r in records)}', flush=True)
    indexed = {r['key']: r for r in records}
    records = [indexed[r['key']] for r in inventory]
    passed = len(records) == 80 and all(r['status'] != 'failed' for r in records)
    total_bytes = sum(r.get('bytes', 0) for r in records)
    if passed and total_bytes > protocol['maximum_total_archive_bytes']:
        passed = False
    summary = {'identity': identity, 'evidence_status': 'reserved_opaque_sources_only',
        'all_objects_verified': passed, 'records': records, 'total_archive_bytes': total_bytes,
        'seconds': time.monotonic() - begin, 'raw_archive_contents_opened': False,
        'features_labels_models_or_assessment_scores_created': False,
        'confirmation_algorithm_frozen': False, 'independent_confirmation_run': False,
        'artifact_hashes': {str(p.relative_to(output)): sha256_file(p) for p in output.rglob('*') if p.is_file()}}
    write_json(output / 'summary.json', summary)
    if not passed:
        raise ValueError('Preserve incomplete source reservation; no date replacement or content inspection is authorized by this protocol')
    print(f'confirmation_reserve_{stage}_complete objects=80 bytes={total_bytes}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['metadata', 'download'], required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve(), args.stage)
