import json

import pytest

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_confirmation_integrity import verify_frozen_fold


def test_confirmation_unsealing_rejects_changed_or_missing_predictions(tmp_path):
    folder = tmp_path / "2023-05-24"
    (folder / "predictions").mkdir(parents=True)
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        for model in ["original_reference", "matched_data_control", "candidate", "neural_component", "tree_component"]:
            (folder / "predictions" / f"{symbol}_{model}.npz").write_bytes(b"frozen")
    identity = {"protocol_sha256": "example"}
    record = {
        "identity": identity,
        "assessment_date": folder.name,
        "artifact_hashes": {str(p.relative_to(folder)): sha256_file(p) for p in folder.rglob("*.npz")},
    }
    (folder / "completed.json").write_text(json.dumps(record))
    assert verify_frozen_fold(folder, identity) == record
    changed = folder / "predictions/BTCUSDT_candidate.npz"
    changed.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_frozen_fold(folder, identity)
    changed.unlink()
    with pytest.raises(ValueError, match="inventory"):
        verify_frozen_fold(folder, identity)
