"""Learned causal sequence corrections to a frozen historical neural model."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from lob_forge.boundary_prequential import integer_clock

SEMANTICS = "past_fitted_neural_with_causal_recurrent_logit_correction_v1"
ARCHITECTURES = ("instant_residual", "gru_residual", "lstm_residual")
CONFIG = {"seed": 20260908, "width": 64, "epochs": 6, "truncation_seconds": 128,
          "optimization_seconds": 512, "learning_rate": .0003, "weight_decay": .01, "gradient_norm_limit": 5.}


def build_residual_network(dimensions, architecture, *, width=64):
    import torch
    from torch import nn

    if architecture not in ARCHITECTURES or dimensions < 1 or width < 1:
        raise ValueError("A registered residual architecture and positive dimensions are required")

    class ResidualNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.architecture, self.width, self.dimensions = architecture, width, dimensions
            self.projection = nn.Sequential(nn.Linear(dimensions, width), nn.GELU())
            if architecture == "instant_residual":
                self.body = nn.Sequential(nn.Linear(width, 3 * width), nn.GELU(), nn.Linear(3 * width, width), nn.GELU())
            else:
                constructor = nn.GRU if architecture == "gru_residual" else nn.LSTM
                self.body = constructor(width, width, num_layers=1, batch_first=True, dropout=0., bidirectional=False)
            self.head = nn.Linear(width, 3)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def initial_state(self, streams):
            if self.architecture == "instant_residual":
                return None
            h = self.head.weight.new_zeros((1, streams, self.width))
            return (h, h.clone()) if self.architecture == "lstm_residual" else h

        def forward(self, matrix, state=None):
            encoded = self.projection(matrix)
            if self.architecture == "instant_residual":
                return self.head(self.body(encoded)), None
            values, next_state = self.body(encoded, state)
            return self.head(values), next_state

    # Initialize on CPU, then explicitly move the result to a validated backend.
    torch.manual_seed(CONFIG["seed"])
    return ResidualNetwork()


def map_state(state, function):
    if state is None:
        return None
    return tuple(function(value) for value in state) if isinstance(state, tuple) else function(state)


def place_state(original, index, updated):
    if original is None:
        return None
    if isinstance(original, tuple):
        return tuple(left.index_copy(1, index, right) for left, right in zip(original, updated))
    return original.index_copy(1, index, updated)


def masked_segments(available, *, offset=0, truncation_seconds=128):
    valid = np.asarray(available)
    if valid.ndim != 2 or not all(valid.shape) or valid.dtype != bool or truncation_seconds < 1 or offset < 0:
        raise ValueError("A nonempty Boolean stream/time mask and positive truncation are required")
    changes = np.flatnonzero(np.any(valid[:, 1:] != valid[:, :-1], axis=0)) + 1
    first = (offset // truncation_seconds + 1) * truncation_seconds - offset
    boundaries = sorted({0, valid.shape[1], *changes.tolist(), *range(first, valid.shape[1], truncation_seconds)})
    return [(left, right, np.flatnonzero(valid[:, left])) for left, right in zip(boundaries[:-1], boundaries[1:])]


def accumulate_temporal_block(network, matrix, base_logits, labels, weights, available, supervised, state, *, offset=0, truncation_seconds=128):
    """Accumulate one full-block mean, detaching state at registered boundaries.

    The caller zeros gradients and steps its optimizer once for this entire
    block. Missing streams are reset without executing a synthetic observation.
    """
    import torch

    x, z, y, w, valid, keep = (np.asarray(v) for v in (matrix, base_logits, labels, weights, available, supervised))
    if x.ndim != 3 or z.shape != (*x.shape[:2], 3) or any(v.shape != x.shape[:2] for v in (y, w, valid, keep)):
        raise ValueError("Aligned stream/time matrices, logits, labels and masks are required")
    if valid.dtype != bool or keep.dtype != bool or (keep & ~valid).any() or not np.isin(y[keep], [-1, 0, 1]).all():
        raise ValueError("Only eligible original three-class rows may be supervised")
    if not np.isfinite(x[valid]).all() or not np.isfinite(z[keep]).all() or not np.isfinite(w[keep]).all() or (w[keep] < 0).any():
        raise ValueError("Observed inputs, used logits and nonnegative weights must be finite")
    device = next(network.parameters()).device
    count, total, blocks = int(keep.sum()), 0., 0
    if state is None:
        state = network.initial_state(len(x))
    for left, right, active in masked_segments(valid, offset=offset, truncation_seconds=truncation_seconds):
        live = torch.as_tensor(valid[:, left].astype(np.float32), device=device).view(1, -1, 1)
        state = map_state(state, lambda value: value * live)
        if not len(active):
            state = map_state(state, lambda value: value.detach())
            continue
        index = torch.as_tensor(active, dtype=torch.long, device=device)
        subset = map_state(state, lambda value: value.index_select(1, index))
        inputs = torch.as_tensor(np.ascontiguousarray(x[active, left:right]), dtype=torch.float32, device=device)
        delta, updated = network(inputs, subset)
        state = map_state(place_state(state, index, updated), lambda value: value.detach())
        selected = keep[active, left:right]
        if selected.any():
            selected_tensor = torch.as_tensor(selected, dtype=torch.bool, device=device)
            targets = torch.as_tensor(y[active, left:right][selected] + 1, dtype=torch.long, device=device)
            original = torch.as_tensor(z[active, left:right][selected], dtype=torch.float32, device=device)
            sample_weights = torch.as_tensor(w[active, left:right][selected], dtype=torch.float32, device=device)
            loss = (torch.nn.functional.cross_entropy(original + delta[selected_tensor], targets, reduction="none") * sample_weights).sum() / count
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite learned-memory loss")
            loss.backward()
            total += float(loss.detach().cpu())
        blocks += 1
    return state, {"weighted_mean_loss": total, "supervised_rows": count, "forward_segments": blocks}


def predict_deltas(network, matrix, times, *, selected=None, chunk_rows=128):
    """Forward-only state, reset at coverage gaps, with pointwise row selection."""
    import torch

    x, clock = np.asarray(matrix), integer_clock(times, strict=True)
    keep = np.ones(len(clock), dtype=bool) if selected is None else np.asarray(selected)
    if x.ndim != 2 or len(x) != len(clock) or not len(clock) or not np.isfinite(x).all() or keep.shape != clock.shape or keep.dtype != bool or not keep.any() or chunk_rows < 1:
        raise ValueError("Finite aligned chronological inputs and a nonempty Boolean selection are required")
    device = next(network.parameters()).device
    gaps = np.flatnonzero(np.diff(clock) != 1000) + 1
    boundaries = [0, *gaps.tolist(), len(clock)]
    result, resets = [], 0
    network.eval()
    with torch.no_grad():
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            state = network.initial_state(1)
            resets += 1
            for left in range(begin, end, chunk_rows):
                right = min(left + chunk_rows, end)
                value = torch.as_tensor(np.ascontiguousarray(x[left:right][None]), dtype=torch.float32, device=device)
                delta, state = network(value, state)
                if keep[left:right].any():
                    result.append(delta[0].cpu().numpy()[keep[left:right]].copy())
    values = np.concatenate(result)
    if not np.isfinite(values).all():
        raise ValueError("Finite sequence corrections are required for every query")
    return values, {"observed_rows": len(clock), "query_rows": int(keep.sum()), "state_resets": resets, "chunk_rows": chunk_rows}


def corrected_posterior(base_logits, delta, priors, original_probabilities):
    z, change, pi, original = (np.asarray(v, dtype=float) for v in (base_logits, delta, priors, original_probabilities))
    if z.ndim != 2 or z.shape[1] != 3 or change.shape != z.shape or original.shape != z.shape or pi.shape != (3,):
        raise ValueError("Aligned three-class base logits, corrections, priors and original forecasts required")
    if any(not np.isfinite(v).all() for v in (z, change, pi, original)) or (pi <= 0).any() or not np.isclose(pi.sum(), 1) or (original < 0).any() or not np.allclose(original.sum(axis=1), 1):
        raise ValueError("Finite logits and normalized positive priors/forecasts required")
    logits = z + change + np.log(pi)
    if not np.isfinite(logits).all():
        raise ValueError("Combined corrected logits must remain finite")
    values = np.exp(logits - logits.max(axis=1, keepdims=True))
    values /= values.sum(axis=1, keepdims=True)
    # Row-local, preserving exact original arithmetic without coupling queries.
    zero = np.all(change == 0, axis=1)
    values[zero] = original[zero]
    return values


def original_logits(original, frame, asset, *, batch_size=2048):
    return matrix_logits(original, original.matrix(frame, asset), batch_size=batch_size)


def matrix_logits(original, matrix, *, batch_size=2048):
    import torch

    original.network.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(matrix), batch_size):
            pieces.append(original.network(torch.from_numpy(matrix[start:start + batch_size])).squeeze(1).numpy())
    return np.concatenate(pieces)


@dataclass
class SequenceInputs:
    matrix: np.ndarray
    times: np.ndarray
    selected: np.ndarray
    base_logits: np.ndarray
    base_probabilities: np.ndarray

    def validate(self, dimensions):
        clock = integer_clock(self.times, strict=True)
        keep = np.asarray(self.selected)
        if self.matrix.shape != (len(clock), dimensions) or keep.shape != clock.shape or keep.dtype != bool or not keep.any():
            raise ValueError("Aligned original sequence schema and a nonempty query mask required")
        if self.base_logits.shape != (int(keep.sum()), 3) or self.base_probabilities.shape != self.base_logits.shape:
            raise ValueError("Base logits and probabilities must align to selected observations")
        if any(not np.isfinite(a).all() for a in (self.matrix, self.base_logits, self.base_probabilities)):
            raise ValueError("All actual sequence observations and base forecasts must be finite")

    def predict(self, adapter, prior):
        delta, record = predict_deltas(adapter, self.matrix, self.times, selected=self.selected)
        return corrected_posterior(self.base_logits, delta, prior, self.base_probabilities), record

    def save(self, path):
        np.savez_compressed(path, matrix=self.matrix, times=self.times, selected=self.selected,
            base_logits=self.base_logits, base_probabilities=self.base_probabilities)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            return cls(*(saved[name].copy() for name in ("matrix", "times", "selected", "base_logits", "base_probabilities")))


@dataclass
class TrainingGrid:
    matrix: np.ndarray
    base_logits: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    available: np.ndarray
    supervised: np.ndarray
    assets: np.ndarray

    def validate(self, dimensions, priors):
        if self.matrix.ndim != 3 or self.matrix.shape[0] != 8 or self.matrix.shape[1] != 86400 or self.matrix.shape[2] != dimensions:
            raise ValueError("The registered eight full-day input streams are required")
        shape = self.matrix.shape[:2]
        if self.base_logits.shape != (*shape, 3) or any(a.shape != shape for a in (self.labels, self.weights, self.available, self.supervised)):
            raise ValueError("Every training grid field must align")
        if self.available.dtype != bool or self.supervised.dtype != bool or self.assets.shape != (8,) or not np.isin(self.assets, [0, 1]).all():
            raise ValueError("Boolean availability/supervision and original asset identities required")
        expected = self.available & (np.arange(86400)[None] % 4 == 0)
        np.testing.assert_array_equal(self.supervised, expected)
        total = int(expected.sum())
        if not total or not np.isin(self.labels[expected], [-1, 0, 1]).all():
            raise ValueError("Only original nonempty three-class supervision is allowed")
        for asset in (0, 1):
            keep = expected & (self.assets[:, None] == asset)
            counts = np.array([np.count_nonzero(keep & (self.labels == y)) for y in (-1, 0, 1)])
            if (counts == 0).any() or np.count_nonzero(self.assets == asset) != 4:
                raise ValueError("Each asset needs its four training dates and all original classes")
            np.testing.assert_array_equal(counts / counts.sum(), priors[asset])
            for position, label in enumerate((-1, 0, 1)):
                weights = self.weights[keep & (self.labels == label)]
                np.testing.assert_allclose(weights, total / (6 * counts[position]), rtol=1e-7, atol=0)


def fit_learned_memory(original, training, validation_inputs, validation_labels, *, architecture, device):
    """Fit on historical streams; assessment inputs and outcomes are not accepted."""
    import torch

    from lob_forge.boundary_confirm_model import SYMBOLS
    from lob_forge.boundary_forecasts import classification_metrics

    if device not in ("cpu", "mps") or (device == "mps" and not torch.backends.mps.is_available()):
        raise ValueError("An available validated CPU or Mac GPU backend is required")
    if set(validation_inputs) != set(SYMBOLS) or set(validation_labels) != set(SYMBOLS):
        raise ValueError("Both original validation assets are required")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    dimensions = int(original.active.sum()) + 1
    training.validate(dimensions, original.priors)
    for symbol in SYMBOLS:
        validation_inputs[symbol].validate(dimensions)
        if len(validation_labels[symbol]) != int(validation_inputs[symbol].selected.sum()):
            raise ValueError("Every selected validation observation requires its original label")
    adapter = build_residual_network(dimensions, architecture).to(device)
    model = LearnedMemoryForecaster(original, adapter, architecture)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    best_key, best_state, best_epoch, history, total_steps = (-np.inf, -np.inf), None, None, [], 0
    for epoch in range(1, CONFIG["epochs"] + 1):
        begin = time.monotonic()
        adapter.train()
        state, loss_sum, supervised_rows, steps = None, 0., 0, 0
        for left in range(0, 86400, CONFIG["optimization_seconds"]):
            right = min(left + CONFIG["optimization_seconds"], 86400)
            optimizer.zero_grad()
            state, result = accumulate_temporal_block(adapter, training.matrix[:, left:right], training.base_logits[:, left:right],
                training.labels[:, left:right], training.weights[:, left:right], training.available[:, left:right],
                training.supervised[:, left:right], state, offset=left, truncation_seconds=CONFIG["truncation_seconds"])
            if result["supervised_rows"]:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), CONFIG["gradient_norm_limit"], error_if_nonfinite=True)
                optimizer.step()
                steps += 1
                loss_sum += result["weighted_mean_loss"] * result["supervised_rows"]
                supervised_rows += result["supervised_rows"]
        if supervised_rows != int(training.supervised.sum()):
            raise ValueError("Every original training row must contribute exactly once per epoch")
        metrics = {}
        for asset, symbol in enumerate(SYMBOLS):
            p, _ = validation_inputs[symbol].predict(adapter, original.priors[asset])
            metrics[symbol] = classification_metrics(SimpleNamespace(priors=original.priors[asset]), p, validation_labels[symbol])
        key = (float(np.mean([m["balanced_accuracy"] for m in metrics.values()])), -float(np.mean([m["log_loss"] for m in metrics.values()])))
        if key > best_key:
            best_key, best_epoch = key, epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in adapter.state_dict().items()})
        total_steps += steps
        history.append({"epoch": epoch, "mean_training_step_loss": loss_sum / supervised_rows,
            "optimizer_steps": steps, "supervised_rows": supervised_rows, "validation": metrics, "seconds": time.monotonic() - begin})
    adapter.load_state_dict(best_state)
    adapter.eval()
    return model, {"history": history, "best_epoch": best_epoch, "optimizer_steps": total_steps,
        "config": dict(CONFIG), "backend": device, "trainable_parameters": sum(p.numel() for p in adapter.parameters()),
        "original_parameters_frozen": all(not p.requires_grad for p in original.network.parameters()), "semantics": SEMANTICS}


@dataclass
class LearnedMemoryForecaster:
    original: Any
    adapter: Any
    architecture: str

    def __post_init__(self):
        if self.original.members != 1 or self.original.hidden_size != 64 or set(self.original.priors) != {0, 1}:
            raise ValueError("The original single-member 64-wide two-asset base is required")
        if self.adapter.architecture != self.architecture or self.adapter.dimensions != int(self.original.active.sum()) + 1:
            raise ValueError("Base and residual schemas must agree")
        self.original.network.eval()
        for parameter in self.original.network.parameters():
            parameter.requires_grad_(False)

    def predict_proba(self, frame, times, asset, selected):
        matrix = self.original.matrix(frame, asset)
        delta, record = predict_deltas(self.adapter, matrix, times, selected=selected)
        query = frame.loc[np.asarray(selected)].copy()
        original = self.original.predict_proba(query, asset)
        logits = original_logits(self.original, query, asset)
        return corrected_posterior(logits, delta, self.original.priors[asset], original), record

    def save(self, folder):
        import torch

        folder = Path(folder)
        self.original.save(folder / "base")
        torch.save({"semantics": SEMANTICS, "architecture": self.architecture, "dimensions": self.adapter.dimensions,
            "width": self.adapter.width, "state": {k: v.detach().cpu() for k, v in self.adapter.state_dict().items()}}, folder / "adapter.pt")

    @classmethod
    def load(cls, folder, *, device="cpu"):
        import torch

        from lob_forge.boundary_pooled import PooledForecaster

        folder = Path(folder)
        saved = torch.load(folder / "adapter.pt", map_location="cpu", weights_only=True)
        if saved["semantics"] != SEMANTICS:
            raise ValueError("Learned-memory checkpoint semantics changed")
        original = PooledForecaster.load(folder / "base")
        if saved["dimensions"] != int(original.active.sum()) + 1:
            raise ValueError("Original input and residual dimensions changed")
        adapter = build_residual_network(saved["dimensions"], saved["architecture"], width=saved["width"])
        adapter.load_state_dict(saved["state"])
        adapter.to(device).eval()
        return cls(original, adapter, saved["architecture"])
