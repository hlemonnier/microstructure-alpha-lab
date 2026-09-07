"""Require the extracted confirmation trainer to reproduce a completed development fit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_confirm_model import SYMBOLS, fit_confirm_neural
from lob_forge.boundary_event_inputs import load_event_inputs, utc_ms

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "results/boundary_pooled_neural_20260907/2023-05-23/pooled_mlp/BTCUSDT_ETHUSDT"
    completed = json.loads((source / "completed.json").read_text())
    record = completed["records"][0]
    features, labels, times = load_event_inputs(
        ROOT, ROOT / "data/research/boundary_event_data_20260907/dataset_manifest.json"
    )
    train = {
        symbol: pd.concat([features[symbol, day] for day in record["train_dates"]], ignore_index=True)
        for symbol in SYMBOLS
    }
    train_y = {symbol: np.concatenate([labels[symbol, day] for day in record["train_dates"]]) for symbol in SYMBOLS}
    validation, validation_y = {}, {}
    for symbol in SYMBOLS:
        day = record["validation_date"]
        keep = (times[symbol, day] >= utc_ms(day, "12:02:00")) & (times[symbol, day] < utc_ms(day, "13:59:50"))
        validation[symbol], validation_y[symbol] = features[symbol, day].loc[keep], labels[symbol, day][keep]
    model, training = fit_confirm_neural(train, train_y, validation, validation_y)
    if training["best_epoch"] != record["best_epoch"]:
        raise ValueError("Checkpoint selection differs from the frozen development trainer")
    comparisons = []
    for symbol in SYMBOLS:
        for partition, day in [("validation", record["validation_date"]), ("assessment", record["assessment_date"])]:
            path = source / f"{symbol}_{partition}_predictions.npz"
            if sha256_file(path) != completed["artifact_hashes"][path.name]:
                raise ValueError("Frozen reproduction reference changed")
            old = np.load(path)
            keep = (times[symbol, day] >= utc_ms(day, "12:02:00")) & (times[symbol, day] < utc_ms(day, "13:59:50"))
            current = model.predict_proba(features[symbol, day].loc[keep], SYMBOLS.index(symbol))
            np.testing.assert_array_equal(times[symbol, day][keep], old["decision_times"])
            np.testing.assert_array_equal(current, old["probabilities"])
            comparisons.append(
                {
                    "symbol": symbol,
                    "partition": partition,
                    "rows": len(current),
                    "bit_identical": True,
                    "reference_sha256": sha256_file(path),
                }
            )
    result = {
        "status": "passed",
        "comparisons": comparisons,
        "best_epoch": training["best_epoch"],
        "code_hashes": {
            name: sha256_file(ROOT / name)
            for name in [
                "src/lob_forge/boundary_confirm_model.py",
                "src/lob_forge/boundary_pooled.py",
                "scripts/verify_boundary_confirmation_reproduction.py",
            ]
        },
    }
    output = ROOT / "docs/research/boundary_confirmation_reproduction_20260907.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
