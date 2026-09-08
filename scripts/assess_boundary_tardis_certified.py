"""Check certified replay against the frozen limited-frontier observations."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_tardis_certified import CERTIFIED_SEMANTICS, CertifiedBinanceFuturesDepthState
from lob_forge.boundary_tardis_fast import FastBinanceFuturesDepthState, iter_tardis_messages_fast
from lob_forge.boundary_tardis_observations import sample_tardis_depth
from run_boundary_confirmation import write_json

ROOT = Path(__file__).resolve().parents[1]


def assess(output):
    baseline_path = ROOT / "data/research/boundary_tardis_depth_fast_20260907/depth_manifest.json"
    baseline = json.loads(baseline_path.read_text())
    raw = ROOT / "data/research/boundary_tardis_raw_20260907/minute_0000/attempt_1.payload"
    clocks = {}
    for record in baseline["sessions"]:
        path = ROOT / record["observation_path"]
        if sha256_file(path) != record["observation_sha256"]:
            raise ValueError("Frozen limited-frontier observations changed")
        with np.load(path, allow_pickle=False) as values:
            times = values["decision_times"]
            clocks[record["symbol"]] = times[times < 1685578200000]
    observations = []
    seconds = []
    for factory in (FastBinanceFuturesDepthState, CertifiedBinanceFuturesDepthState):
        begin = time.monotonic()
        observations.append(sample_tardis_depth([raw], clocks, state_factory=factory, message_reader=iter_tardis_messages_fast))
        seconds.append(time.monotonic() - begin)
    old, old_quotes, old_checks = observations[0]
    new, new_quotes, new_checks = observations[1]
    for symbol, values in old.items():
        # This initial interval never leaves the original frontiers. Therefore
        # every sampled value, mask, source clock and quote must remain exact.
        for key, expected in values.items():
            np.testing.assert_array_equal(new[symbol][key], expected)
        for key, expected in old_quotes[symbol].items():
            np.testing.assert_array_equal(new_quotes[symbol][key], expected)
    files = ["src/lob_forge/boundary_tardis_depth.py", "src/lob_forge/boundary_tardis_fast.py",
        "src/lob_forge/boundary_tardis_certified.py", "src/lob_forge/boundary_tardis_observations.py",
        "scripts/prepare_boundary_tardis_depth.py", "scripts/assess_boundary_tardis_certified.py",
        "tests/test_boundary_tardis_certified.py", "tests/test_boundary_tardis_certified_observations.py"]
    write_json(output, {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "semantics": CERTIFIED_SEMANTICS,
        "model_fits": 0, "target_columns_read": False, "interval": "2023-06-01 00:00-00:10 UTC",
        "raw_path": str(raw.relative_to(ROOT)), "raw_sha256": sha256_file(raw),
        "baseline_manifest_sha256": sha256_file(baseline_path), "all_initial_sample_and_quote_arrays_exact": True,
        "baseline_checks": old_checks, "certified_checks": new_checks,
        "reference_seconds": seconds[0], "certified_seconds": seconds[1],
        "timing_limit": "Single adjacent operational run, not a controlled performance benchmark.",
        "code_hashes": {f: sha256_file(ROOT / f) for f in files}})
    print(f"certified_initial_parity {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assess(args.output.resolve())
