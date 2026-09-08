"""Explicit backend adapter for the unchanged corrected pretrained classifier."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lob_forge.binance_vision import sha256_file
from lob_forge.boundary_tabicl import SEMANTICS, SYMBOLS, TABICL_SETTINGS, TabiclForecaster
from lob_forge.boundary_tabicl_preprocessing import install_row_local_power_fallback


def fit_backend_context(features, labels, population_priors, *, checkpoint, checkpoint_sha256, backend):
    import torch
    from tabicl import TabICLClassifier

    if backend not in ("cpu", "mps") or (backend == "mps" and not torch.backends.mps.is_available()):
        raise ValueError("The explicitly validated CPU or Mac GPU backend is required")
    if set(features) != set(SYMBOLS) or set(labels) != set(SYMBOLS) or sha256_file(Path(checkpoint)) != checkpoint_sha256:
        raise ValueError("Both original assets and the exact fixed checkpoint are required")
    columns = list(features[SYMBOLS[0]].columns)
    sampled = {}
    for symbol in SYMBOLS:
        y = np.asarray(labels[symbol])
        if list(features[symbol].columns) != columns or y.ndim != 1 or not len(y) or len(features[symbol]) != len(y) or not np.isin(y, [-1, 0, 1]).all():
            raise ValueError("Aligned context schemas and three-class outcomes required")
        sampled[symbol] = np.array([(y == label).mean() for label in (-1, 0, 1)])
        if not np.array_equal(sampled[symbol], np.full(3, 1 / 3)):
            raise ValueError("Exact equal context asset/class cells required")
    if len(labels[SYMBOLS[0]]) != len(labels[SYMBOLS[1]]):
        raise ValueError("Exact equal context asset/class cells required")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    settings = {**TABICL_SETTINGS, "device": backend}
    estimator = TabICLClassifier(**settings, model_path=str(Path(checkpoint).resolve()))
    model = TabiclForecaster(estimator, columns, population_priors, sampled, str(Path(checkpoint).resolve()), checkpoint_sha256)
    matrix = np.concatenate([model.matrix(features[s], s) for s in SYMBOLS])
    estimator.fit(matrix, np.concatenate([labels[s] for s in SYMBOLS]))
    install_row_local_power_fallback(estimator)
    return model


def load_backend_context(folder, *, backend):
    from tabicl import TabICLClassifier

    if backend not in ("cpu", "mps"):
        raise ValueError("The explicitly registered CPU or Mac GPU backend is required")
    folder = Path(folder)
    metadata = json.loads((folder / "model.json").read_text())
    if metadata["semantics"] != SEMANTICS or sha256_file(Path(metadata["checkpoint"])) != metadata["checkpoint_sha256"] or sha256_file(folder / "context.pkl") != metadata["context_sha256"]:
        raise ValueError("Pinned pretrained weights, context or probability semantics changed")
    estimator = TabICLClassifier.load(folder / "context.pkl", device=backend)
    install_row_local_power_fallback(estimator)
    return TabiclForecaster(estimator, metadata["columns"],
        {s: np.array(p) for s, p in metadata["population_priors"].items()},
        {s: np.array(p) for s, p in metadata["sampled_priors"].items()}, metadata["checkpoint"], metadata["checkpoint_sha256"])
