"""Finite, causal temporal aggregation as a correction to a frozen forecaster."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from lob_forge.boundary_learned_memory import corrected_posterior, original_logits
from lob_forge.boundary_prequential import integer_clock

SEMANTICS = "past_normalized_finite_temporal_attention_residual_v1"
ARCHITECTURES = ("instant", "uniform", "age", "content")
CONFIG = {"seed": 20260908, "epochs": 6, "width": 64, "heads": 4, "history_observations": 128,
          "optimization_seconds": 512, "learning_rate": .0003, "weight_decay": .01, "gradient_norm_limit": 5.}


def contiguous_segments(available):
    """Give each contiguous observed interval its own id; missing cells are -1."""
    valid = np.asarray(available)
    if valid.ndim != 2 or not all(valid.shape) or valid.dtype != bool:
        raise ValueError("A nonempty Boolean stream/time availability matrix is required")
    starts = valid & ~np.concatenate([np.zeros((len(valid), 1), dtype=bool), valid[:, :-1]], axis=1)
    return np.where(valid, np.cumsum(starts, axis=1, dtype=np.int64) - 1, -1)


def build_attention_network(dimensions, architecture, *, width=64, heads=4, history_observations=128):
    import torch
    from torch import nn

    if (architecture not in ARCHITECTURES or any(not isinstance(v, int) or isinstance(v, bool) or v < 1
        for v in (dimensions, width, heads, history_observations)) or width % heads):
        raise ValueError("Registered aggregation and positive head-divisible dimensions are required")

    class TemporalReadout(nn.Module):
        def __init__(self):
            super().__init__()
            self.architecture, self.dimensions, self.width = architecture, dimensions, width
            self.heads, self.history_observations = heads, history_observations
            # Initialize common layers before architecture-specific parameters.
            self.stem = nn.Sequential(nn.Linear(dimensions, width), nn.GELU())
            self.value = nn.Linear(width, width)
            self.body = nn.Sequential(nn.Linear(2 * width, 2 * width), nn.GELU(), nn.Linear(2 * width, width), nn.GELU())
            self.head = nn.Linear(width, 3)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            self.age_bias = nn.Parameter(torch.zeros(heads, history_observations)) if architecture in ("age", "content") else None
            self.query = nn.Linear(width, width) if architecture == "content" else None
            self.key = nn.Linear(width, width) if architecture == "content" else None

        def forward(self, matrix, segments, positions, *, return_attention=False):
            if (matrix.ndim != 3 or not all(matrix.shape) or matrix.shape[2] != self.dimensions
                or matrix.dtype != self.head.weight.dtype or matrix.device != self.head.weight.device
                or segments.shape != matrix.shape[:2] or segments.dtype != torch.long or segments.device != matrix.device
                or positions.ndim != 1 or not len(positions) or positions.dtype != torch.long or positions.device != matrix.device
                or (positions < 0).any() or (positions >= matrix.shape[1]).any() or (positions[1:] <= positions[:-1]).any()
                or (segments < -1).any() or not torch.isfinite(matrix).all()):
                raise ValueError("Finite aligned observations, segment ids and ordered in-range query positions are required")
            ids = segments.detach().cpu().numpy()
            changed = (ids[:, 1:] >= 0) & (ids[:, 1:] != ids[:, :-1])
            previous = np.maximum.accumulate(ids, axis=1)[:, :-1]
            if np.any(changed & (ids[:, 1:] <= previous)):
                raise ValueError("Each observed segment needs a new increasing id after a boundary or missing interval")
            # Masking scores alone is insufficient: overflow at an excluded
            # query can still produce NaN gradients through an unused softmax.
            observed_matrix = torch.where(segments[:, :, None] >= 0, matrix, torch.zeros_like(matrix))
            encoded = self.stem(observed_matrix)
            current = encoded.index_select(1, positions)
            batch, queries, _ = current.shape
            dimension = self.width // self.heads
            values = self.value(encoded).reshape(batch, matrix.shape[1], self.heads, dimension)
            current_segments = segments.index_select(1, positions)
            observed = current_segments >= 0
            attention = None
            if self.architecture == "instant":
                aggregate = values.index_select(1, positions).reshape(batch, queries, self.width)
            else:
                # Columns run oldest -> current; the trainable age index is lag.
                lags = torch.arange(self.history_observations - 1, -1, -1, device=matrix.device)
                indices = positions[:, None] - lags[None]
                safe_indices = indices.clamp_min(0)
                same_segment = segments[:, safe_indices] == current_segments[:, :, None]
                allowed = (indices[None] >= 0) & same_segment & observed[:, :, None]
                # An unobserved query has no output or gradient. One temporary
                # key prevents undefined all-masked softmax inside that row.
                allowed = allowed | ((~observed)[:, :, None] & (lags == 0)[None, None])
                historical_values = values[:, safe_indices]
                if self.architecture == "uniform":
                    weights = allowed.to(matrix.dtype)
                    weights = weights / weights.sum(dim=-1, keepdim=True)
                    attention = weights[:, :, None].expand(-1, -1, self.heads, -1)
                else:
                    scores = self.age_bias.flip(-1)[None, None].expand(batch, queries, -1, -1)
                    if self.architecture == "content":
                        query = self.query(current).reshape(batch, queries, self.heads, dimension)
                        keys = self.key(encoded).reshape(batch, matrix.shape[1], self.heads, dimension)
                        scores = scores + torch.einsum("bqhd,bqthd->bqht", query, keys[:, safe_indices]) / dimension ** .5
                    attention = torch.softmax(scores.masked_fill(~allowed[:, :, None], -torch.inf), dim=-1)
                aggregate = torch.einsum("bqht,bqthd->bqhd", attention, historical_values).reshape(batch, queries, self.width)
            delta = self.head(self.body(torch.cat([current, aggregate], dim=-1)))
            delta = torch.where(observed[:, :, None], delta, torch.zeros_like(delta))
            if not torch.isfinite(delta).all():
                raise ValueError("Nonfinite temporal-attention correction")
            return (delta, attention) if return_attention else delta

    torch.manual_seed(CONFIG["seed"])
    return TemporalReadout()


def accumulate_attention_block(network, matrix, segments, positions, base_logits, labels, weights, supervised):
    """Accumulate the entire block's weighted mean; caller owns zeroing/stepping."""
    import torch

    x, ids, query, z, y, w, keep = (np.asarray(v) for v in (matrix, segments, positions, base_logits, labels, weights, supervised))
    if (x.ndim != 3 or ids.shape != x.shape[:2] or not np.issubdtype(ids.dtype, np.integer)
        or query.ndim != 1 or not len(query) or not np.issubdtype(query.dtype, np.integer)
        or (query < 0).any() or (query >= x.shape[1]).any() or (np.diff(query) <= 0).any()):
        raise ValueError("Aligned observations, segment ids and ordered query positions are required")
    shape = (len(x), len(query))
    if (z.shape != (*shape, 3) or any(v.shape != shape for v in (y, w, keep)) or keep.dtype != bool
        or not np.issubdtype(y.dtype, np.integer) or not np.isin(y[keep], [-1, 0, 1]).all()
        or (keep & (ids[:, query] < 0)).any() or not np.isfinite(x).all()
        or not np.isfinite(z[keep]).all() or not np.isfinite(w[keep]).all() or (w[keep] < 0).any()):
        raise ValueError("Original observed supervision, finite logits and nonnegative weights are required")
    count = int(keep.sum())
    if not count:
        return {"weighted_mean_loss": 0., "supervised_rows": 0}
    parameter = next(network.parameters())
    device, dtype = parameter.device, parameter.dtype
    delta = network(torch.as_tensor(np.ascontiguousarray(x), device=device, dtype=dtype),
        torch.as_tensor(ids, device=device, dtype=torch.long), torch.as_tensor(query, device=device, dtype=torch.long))
    mask = torch.as_tensor(keep, device=device, dtype=torch.bool)
    original = torch.as_tensor(z[keep], device=device, dtype=dtype)
    target = torch.as_tensor(y[keep] + 1, device=device, dtype=torch.long)
    sample_weights = torch.as_tensor(w[keep], device=device, dtype=dtype)
    loss = (torch.nn.functional.cross_entropy(original + delta[mask], target, reduction="none") * sample_weights).sum() / count
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite temporal-attention training loss")
    loss.backward()
    return {"weighted_mean_loss": float(loss.detach().cpu()), "supervised_rows": count}


def predict_deltas(network, matrix, times, *, selected=None, chunk_rows=512):
    """Query finite windows, retaining original elapsed-time gap boundaries."""
    import torch

    x, clock = np.asarray(matrix), integer_clock(times, strict=True)
    keep = np.ones(len(clock), dtype=bool) if selected is None else np.asarray(selected)
    if (x.ndim != 2 or len(x) != len(clock) or not len(clock) or x.shape[1] != network.dimensions
        or not np.isfinite(x).all() or keep.shape != clock.shape or keep.dtype != bool or not keep.any()
        or not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows < 1):
        raise ValueError("Finite original observations, a query mask and positive chunk size are required")
    segments = np.cumsum(np.r_[True, np.diff(clock) != 1000], dtype=np.int64)
    parameter, result, calls = next(network.parameters()), [], 0
    network.eval()
    with torch.no_grad():
        for start in range(0, len(clock), chunk_rows):
            end = min(start + chunk_rows, len(clock))
            queries = np.flatnonzero(keep[start:end]) + start
            if not len(queries):
                continue
            begin = max(0, start - network.history_observations + 1)
            values = network(torch.as_tensor(np.ascontiguousarray(x[None, begin:end]), device=parameter.device, dtype=parameter.dtype),
                torch.as_tensor(segments[None, begin:end], device=parameter.device, dtype=torch.long),
                torch.as_tensor(queries - begin, device=parameter.device, dtype=torch.long))
            result.append(values[0].cpu().numpy())
            calls += 1
    return np.concatenate(result), {"observed_rows": len(clock), "query_rows": int(keep.sum()),
        "coverage_segments": int(segments[-1]), "query_calls": calls, "chunk_rows": chunk_rows}


def sequence_probabilities(network, sequence, prior, *, chunk_rows=512):
    sequence.validate(network.dimensions)
    delta, record = predict_deltas(network, sequence.matrix, sequence.times, selected=sequence.selected, chunk_rows=chunk_rows)
    return corrected_posterior(sequence.base_logits, delta, prior, sequence.base_probabilities), record


@dataclass
class TemporalAttentionForecaster:
    original: Any
    adapter: Any
    architecture: str

    def __post_init__(self):
        import torch

        if (self.original.members != 1 or self.original.hidden_size != 64 or set(self.original.priors) != {0, 1}
            or self.adapter.architecture != self.architecture or self.adapter.dimensions != int(self.original.active.sum()) + 1):
            raise ValueError("The original two-asset M1 base and aligned temporal readout are required")
        if any(p.dtype != torch.float32 for model in (self.original.network, self.adapter) for p in model.parameters()):
            raise ValueError("Persisted original and temporal models must use the registered float32 parameters")
        self.original.network.eval()
        for parameter in self.original.network.parameters():
            parameter.requires_grad_(False)

    def predict_proba(self, frame, times, asset, selected, *, chunk_rows=512):
        matrix = self.original.matrix(frame, asset)
        delta, record = predict_deltas(self.adapter, matrix, times, selected=selected, chunk_rows=chunk_rows)
        query = frame.loc[np.asarray(selected)].copy()
        original = self.original.predict_proba(query, asset)
        logits = original_logits(self.original, query, asset)
        return corrected_posterior(logits, delta, self.original.priors[asset], original), record

    def save(self, folder):
        import torch

        folder = Path(folder)
        self.original.save(folder / "base")
        torch.save({"semantics": SEMANTICS, "architecture": self.architecture, "dimensions": self.adapter.dimensions,
            "width": self.adapter.width, "heads": self.adapter.heads, "history_observations": self.adapter.history_observations,
            "state": {k: v.detach().cpu() for k, v in self.adapter.state_dict().items()}}, folder / "adapter.pt")

    @classmethod
    def load(cls, folder, *, device="cpu"):
        import torch
        from lob_forge.boundary_pooled import PooledForecaster

        if device not in ("cpu", "mps") or (device == "mps" and not torch.backends.mps.is_available()):
            raise ValueError("An available supported device is required")
        folder = Path(folder)
        saved = torch.load(folder / "adapter.pt", map_location="cpu", weights_only=True)
        if saved["semantics"] != SEMANTICS:
            raise ValueError("Temporal-attention checkpoint semantics changed")
        adapter = build_attention_network(saved["dimensions"], saved["architecture"], width=saved["width"],
            heads=saved["heads"], history_observations=saved["history_observations"])
        adapter.load_state_dict(saved["state"])
        return cls(PooledForecaster.load(folder / "base"), adapter.to(device).eval(), saved["architecture"])


def fit_temporal_attention(original, training, validation_inputs, validation_labels, *, architecture, device):
    """Fit historical corrections; current assessment data is not an input."""
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
        target = np.asarray(validation_labels[symbol])
        if (target.shape != (int(validation_inputs[symbol].selected.sum()),) or not np.issubdtype(target.dtype, np.integer)
            or not np.isin(target, [-1, 0, 1]).all()):
            raise ValueError("Every original validation query requires its integer label")
    adapter = build_attention_network(dimensions, architecture).to(device)
    model = TemporalAttentionForecaster(original, adapter, architecture)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    segments = contiguous_segments(training.available)
    best_key, best_state, best_epoch, history, total_steps = (-np.inf, -np.inf), None, None, [], 0
    for epoch in range(1, CONFIG["epochs"] + 1):
        started = time.monotonic()
        adapter.train()
        loss_sum, supervised_rows, steps = 0., 0, 0
        for left in range(0, 86400, CONFIG["optimization_seconds"]):
            right = min(left + CONFIG["optimization_seconds"], 86400)
            begin = max(0, left - CONFIG["history_observations"] + 1)
            queries = np.arange(left, right, 4)
            optimizer.zero_grad()
            record = accumulate_attention_block(adapter, training.matrix[:, begin:right], segments[:, begin:right], queries - begin,
                training.base_logits[:, queries], training.labels[:, queries], training.weights[:, queries], training.supervised[:, queries])
            if record["supervised_rows"]:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), CONFIG["gradient_norm_limit"], error_if_nonfinite=True)
                optimizer.step()
                loss_sum += record["weighted_mean_loss"] * record["supervised_rows"]
                supervised_rows += record["supervised_rows"]
                steps += 1
        if supervised_rows != int(training.supervised.sum()):
            raise ValueError("Every original training row must contribute exactly once per epoch")
        metrics = {}
        for asset, symbol in enumerate(SYMBOLS):
            p, _ = sequence_probabilities(adapter, validation_inputs[symbol], original.priors[asset])
            metrics[symbol] = classification_metrics(SimpleNamespace(priors=original.priors[asset]), p, validation_labels[symbol])
        key = (float(np.mean([m["balanced_accuracy"] for m in metrics.values()])), -float(np.mean([m["log_loss"] for m in metrics.values()])))
        if key > best_key:
            best_key, best_epoch = key, epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in adapter.state_dict().items()})
        total_steps += steps
        history.append({"epoch": epoch, "mean_training_step_loss": loss_sum / supervised_rows, "optimizer_steps": steps,
            "supervised_rows": supervised_rows, "validation": metrics, "seconds": time.monotonic() - started})
    adapter.load_state_dict(best_state)
    adapter.eval()
    return model, {"history": history, "best_epoch": best_epoch, "optimizer_steps": total_steps,
        "config": dict(CONFIG), "backend": device, "trainable_parameters": sum(p.numel() for p in adapter.parameters()),
        "original_parameters_frozen": all(not p.requires_grad for p in original.network.parameters()), "semantics": SEMANTICS}
