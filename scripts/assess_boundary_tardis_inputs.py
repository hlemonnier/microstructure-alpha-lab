"""Verify completed native coverage and every shared original model input."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_combined_inputs import load_combined_inputs
from lob_forge.boundary_tardis_inputs import load_tardis_inputs, select_tardis_variant
from run_boundary_confirmation import noon, write_json

ROOT = Path(__file__).resolve().parents[1]


def assess(output):
    native_path = ROOT / "data/research/boundary_tardis_continuous_20260908/depth_manifest.json"
    old_path = ROOT / "data/research/boundary_tardis_depth_fast_20260907/depth_manifest.json"
    native, old = (json.loads(p.read_text()) for p in (native_path, old_path))
    previous = {r["symbol"]: r for r in old["sessions"]}
    records = []
    for record in native["sessions"]:
        symbol = record["symbol"]
        for item in (record, previous[symbol]):
            for path, expected in item["artifact_hashes"].items():
                if sha256_file(ROOT / path) != expected:
                    raise ValueError("Frozen native replay artifact changed")
        with np.load(ROOT / record["observation_path"], allow_pickle=False) as a, np.load(ROOT / previous[symbol]["observation_path"], allow_pickle=False) as b:
            for key in ("decision_times", "delays_ms", "update_ids"):
                np.testing.assert_array_equal(a[key], b[key])
            overlap = a["available"] & b["available"]
            np.testing.assert_array_equal(a["depth"][overlap], b["depth"][overlap])
            if (b["frontier_covered"] & ~a["frontier_covered"]).any():
                raise ValueError("A valid continuous certificate cannot reduce known price ranges")
            for key in ("capture_times_ns", "publisher_times_ms"):
                if (a[key] < b[key]).any():
                    raise ValueError("A certificate cannot backdate dependency clocks")
            selected = noon(a["decision_times"], record["session_date"])
            records.append({"symbol": symbol, "rows": len(a["decision_times"]), "noon_rows": int(selected.sum()),
                "original_available_by_delay": b["available"].sum(axis=0).tolist(),
                "certified_available_by_delay": a["available"].sum(axis=0).tolist(),
                "original_noon_available_by_delay": b["available"][selected].sum(axis=0).tolist(),
                "certified_noon_available_by_delay": a["available"][selected].sum(axis=0).tolist(),
                "lost_availability_by_delay": (b["available"] & ~a["available"]).sum(axis=0).tolist(),
                "original_overlap_top25_exact": True, "all_decisions_delays_update_ids_exact": True,
                "dependency_clocks_never_backdated": True})
        new_quotes = ROOT / record["observation_path"]
        old_quotes = ROOT / previous[symbol]["observation_path"]
        with np.load(new_quotes.with_name("native_quotes.npz"), allow_pickle=False) as a, np.load(old_quotes.with_name("native_quotes.npz"), allow_pickle=False) as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
        records[-1]["all_native_quote_arrays_exact"] = True
    original_path = ROOT / "data/research/boundary_regime_data_20260907/dataset_manifest.json"
    manifest = json.loads(original_path.read_text())
    partition = output / "partition.json"
    write_json(partition, {"sessions": [r for r in manifest["sessions"] if r["session_date"] == "2023-06-01"]})
    bybit = ROOT / "data/research/boundary_bybit_depth_20260907/depth_manifest.json"
    spot = ROOT / "data/research/boundary_spot_features_20260907/feature_manifest.json"
    original_x, original_y, original_t = load_combined_inputs(ROOT, partition, bybit, spot)
    native_x, native_y, native_t = load_tardis_inputs(ROOT, partition, native_path, bybit, spot)
    widths = {}
    for key in original_x:
        source_times = native_t[key]
        retained = np.isin(original_t[key], source_times)
        np.testing.assert_array_equal(original_t[key][retained], source_times)
        np.testing.assert_array_equal(original_y[key][retained], native_y[key])
        pd.testing.assert_frame_equal(original_x[key].loc[retained].reset_index(drop=True), select_tardis_variant(native_x[key], "combined"), check_exact=True)
        for variant in ("observations", "combined", "top1_100", "top25_100", "top1_500", "top25_500"):
            width = len(select_tardis_variant(native_x[key], variant).columns)
            if variant in widths and widths[variant] != width:
                raise ValueError("Both assets require identical native feature schemas")
            widths[variant] = width
        for record in records:
            if record["symbol"] == key[0]:
                record["original_model_rows_retained"] = len(source_times)
                record["all_399_original_fields_labels_times_exact"] = True
    definition = ROOT / "docs/research/boundary_tardis_model_definition_20260908.json"
    if widths != json.loads(definition.read_text())["feature_counts"]:
        raise ValueError("Actual native source widths differ from the frozen learner definition")
    write_json(output / "evidence.json", {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "model_fits": 0,
        "labels_compared_but_no_scores_computed": True, "native_manifest_sha256": sha256_file(native_path),
        "initial_frontier_manifest_sha256": sha256_file(old_path), "model_definition_sha256": sha256_file(definition),
        "feature_counts": widths, "sessions": records, "source_checks": native["checks"]})
    print(f"native_input_parity {output / 'evidence.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assess(args.output.resolve())
