"""Verify a complete frozen fold before resuming or revealing its outcomes."""

from __future__ import annotations

import json
from pathlib import Path

from lob_forge.binance_vision import sha256_file


def verify_frozen_fold(folder: Path, identity: dict) -> dict:
    saved = json.loads((folder / "completed.json").read_text())
    if saved["identity"] != identity or saved["assessment_date"] != folder.name:
        raise ValueError("Completed fold identity differs from the frozen study")
    expected = saved["artifact_hashes"]
    actual = {str(path.relative_to(folder)) for path in folder.rglob("*") if path.is_file() and path.name != "completed.json"}
    required = {
        f"predictions/{symbol}_{model}.npz"
        for symbol in ["BTCUSDT", "ETHUSDT"]
        for model in ["original_reference", "matched_data_control", "candidate", "neural_component", "tree_component"]
    }
    if set(expected) != actual or not required.issubset(expected):
        raise ValueError("Frozen fold artifact inventory changed or is incomplete")
    for relative, checksum in expected.items():
        if sha256_file(folder / relative) != checksum:
            raise ValueError("Frozen fold artifact changed")
    return saved
