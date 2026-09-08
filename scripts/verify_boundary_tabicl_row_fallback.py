"""Reproduce and fix query-batch dependence on an already saved failed context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_tabicl import SYMBOLS, TabiclForecaster
from lob_forge.boundary_tabicl_preprocessing import fallback_counts, install_row_local_power_fallback
from run_boundary_confirmation import write_json

ROOT = Path(__file__).resolve().parents[1]


def run(protocol_path, output):
    from tabicl import TabICLClassifier
    import torch

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads(protocol_path.read_text())
    for path, checksum in protocol["input_hashes"].items():
        if sha256_file(ROOT / path) != checksum:
            raise ValueError(f"Frozen diagnostic input changed: {path}")
    if output.exists():
        raise ValueError("Preserve the previous saved-context diagnostic")
    folder = ROOT / protocol["failed_context"]
    metadata = json.loads((folder / "model.json").read_text())
    if metadata["semantics"] != "tabicl_case_control_prior_v1":
        raise ValueError("The exact preserved pre-amendment context is required")
    estimator = TabICLClassifier.load(folder / "context.pkl", device="cpu")
    model = TabiclForecaster(estimator, metadata["columns"],
        {s: np.array(p) for s, p in metadata["population_priors"].items()},
        {s: np.array(p) for s, p in metadata["sampled_priors"].items()}, metadata["checkpoint"], metadata["checkpoint_sha256"])
    queries = joblib.load(ROOT / protocol["query_features"])
    before, changed = {}, {}
    for s in SYMBOLS:
        prefix = queries[s][model.columns].iloc[:64]
        modified = prefix.copy()
        modified.iloc[16:] = -7 * modified.iloc[16:].to_numpy() + 123
        before[s] = model.predict_proba(prefix, s)
        changed[s] = model.predict_proba(modified, s)
    records = {}
    install_row_local_power_fallback(estimator)
    for s in SYMBOLS:
        prefix = queries[s][model.columns].iloc[:64]
        modified = prefix.copy()
        modified.iloc[16:] = -7 * modified.iloc[16:].to_numpy() + 123
        fixed = model.predict_proba(prefix, s)
        later = model.predict_proba(modified, s)
        shorter = model.predict_proba(prefix.iloc[:16], s)
        np.testing.assert_allclose(later[:16], fixed[:16], rtol=0, atol=2e-6)
        np.testing.assert_allclose(shorter, fixed[:16], rtol=0, atol=2e-5)
        records[s] = {"original_later_query_max_error": float(np.max(np.abs(changed[s][:16] - before[s][:16]))),
            "fixed_later_query_max_error": float(np.max(np.abs(later[:16] - fixed[:16]))),
            "fixed_shorter_query_max_error": float(np.max(np.abs(shorter - fixed[:16]))),
            "unperturbed_prefix_change": float(np.max(np.abs(fixed - before[s])))}
    if max(r["original_later_query_max_error"] for r in records.values()) <= 2e-6:
        raise ValueError("The preserved failure must reproduce before the fix is accepted")
    write_json(output, {"protocol_sha256": sha256_file(protocol_path), "records": records,
        "purpose": "Query-causality verification on the already failed saved context; no target labels or accuracy metrics read.",
        "additional_context_fits": 0, "assessment_scores_inspected": False,
        "perturbation_diagnostic_fallback_counts": fallback_counts(estimator)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())
