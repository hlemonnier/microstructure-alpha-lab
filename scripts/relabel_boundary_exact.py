"""Correct threshold labels without changing any observed forecast input."""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_events import FORECAST_SEMANTICS_VERSION, read_archive
from lob_forge.label_math import exact_label_array, price_movement_label

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    source = ROOT / protocol["label_amendment"]["legacy_manifest"]["path"]
    if sha256_file(source) != protocol["label_amendment"]["legacy_manifest"]["sha256"]:
        raise ValueError("Legacy input manifest changed")
    for name, checksum in protocol["code_hashes"].items():
        if sha256_file(ROOT / name) != checksum:
            raise ValueError("Registered implementation changed before relabeling")
    manifest = json.loads(source.read_text())
    identity = {
        "protocol_sha256": sha256_file(protocol_path),
        "legacy_manifest_sha256": sha256_file(source),
        "code_hashes": protocol["code_hashes"],
    }
    frozen = output / "frozen_preparation.json"
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError("Preserve existing relabeling and use a new output directory")
    write_json(frozen, identity)
    records = []
    for number, record in enumerate(manifest["sessions"], 1):
        symbol, day = record["symbol"], record["session_date"]
        old_path = ROOT / record["features_path"]
        path = output / symbol / f"{day}.parquet"
        metadata = path.with_suffix(".json")
        if metadata.exists():
            current = json.loads(metadata.read_text())
            if current["identity"] != identity or sha256_file(path) != current["sha256"]:
                raise ValueError("Corrected session differs from its recorded identity")
        else:
            if sha256_file(old_path) != record["sha256"]:
                raise ValueError("Preserve changed legacy observations")
            raw = record["sources"]["bookTicker"]
            raw_path = Path(raw["local_path"])
            if sha256_file(raw_path) != raw["sha256"]:
                raise ValueError("Raw quote archive checksum changed")
            old = pd.read_parquet(old_path)
            frame = old.copy()
            quotes = read_archive(raw_path, ["best_bid_price", "best_ask_price", "event_time"])
            times = quotes.event_time.to_numpy(dtype=np.int64)
            if (np.diff(times) < 0).any():
                raise ValueError("Raw quotes must be chronological")
            entry = np.searchsorted(times, old.decision_time.to_numpy() + 100)
            future = np.searchsorted(times, old.decision_time.to_numpy() + 5100)
            resolved = future < len(times)
            np.testing.assert_array_equal(resolved, old.label.notna().to_numpy())
            bid, ask = quotes.best_bid_price.to_numpy(), quotes.best_ask_price.to_numpy()
            eb, ea, fb, fa = bid[entry[resolved]], ask[entry[resolved]], bid[future[resolved]], ask[future[resolved]]
            for column, expected in [
                ("entry_event_time", times[entry[resolved]]),
                ("future_event_time", times[future[resolved]]),
                ("entry_mid", (eb + ea) / 2),
                ("future_mid", (fb + fa) / 2),
            ]:
                np.testing.assert_array_equal(old.loc[resolved, column].to_numpy(), expected)
            tick = protocol["data_contract"]["min_tick"][symbol]
            labels = exact_label_array(eb, ea, fb, fa, min_tick=tick)
            # Deterministic mechanical cross-check; no performance or class counts.
            indices = np.arange(0, len(labels), max(1, len(labels) // 128))
            scalar = [price_movement_label(eb[i], ea[i], fb[i], fa[i], min_tick=tick) for i in indices]
            np.testing.assert_array_equal(labels[indices], scalar)
            frame.loc[resolved, "label"] = labels
            frame["feature_semantics_version"] = FORECAST_SEMANTICS_VERSION
            unchanged = [name for name in old if name not in {"label", "feature_semantics_version"}]
            pd.testing.assert_frame_equal(frame[unchanged], old[unchanged], check_exact=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".partial")
            frame.to_parquet(temporary, index=False)
            temporary.replace(path)
            restored = pd.read_parquet(path)
            pd.testing.assert_frame_equal(frame, restored, check_exact=True)
            current = {
                **record,
                "identity": identity,
                "legacy_features_path": record["features_path"],
                "legacy_sha256": record["sha256"],
                "features_path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "feature_semantics_version": FORECAST_SEMANTICS_VERSION,
                "unchanged_observation_and_resolution_parity": True,
                "scalar_label_cross_checks": len(indices),
            }
            write_json(metadata, current)
            del old, frame, restored, quotes, times, entry, future, resolved, bid, ask, eb, ea, fb, fa, labels
            gc.collect()
        records.append(current)
        print(f"exact_label_sessions_verified={number}/{len(manifest['sessions'])} {symbol} {day}", flush=True)
    write_json(
        output / "dataset_manifest.json",
        {
            **manifest,
            "identity": identity,
            "sessions": records,
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "feature_semantics_version": FORECAST_SEMANTICS_VERSION,
            "performance_inspected": False,
        },
    )
    print("exact_label_preparation_complete_no_performance_inspected", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/boundary_confirmation_20260907_exact_labels_revision.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/research/boundary_exact_labels_20260907")
    args = parser.parse_args()
    prepare(args.protocol, args.output)
