"""Delayed supervised updates with explicit model publication times.

An assessment outcome can train a later forecast only after its actual release.
The frozen normalizer and natural-posterior convention never use future rows.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np

from lob_forge.boundary_morning_adaptation import prior_rebasing_offsets
from lob_forge.boundary_neural import natural_posteriors
from lob_forge.boundary_pooled import PooledForecaster, balanced_asset_weights

SYMBOLS = ("BTCUSDT", "ETHUSDT")
SEMANTICS = "delayed_prequential_publication_v1"
VARIANTS = {
    "frozen": {"scope": "none", "learning_rate": 0., "replay_weight": 0.},
    "bias_recent": {"scope": "asset_bias", "learning_rate": .001, "replay_weight": 0.},
    "full_recent": {"scope": "full_network", "learning_rate": .0001, "replay_weight": 0.},
    "full_replay": {"scope": "full_network", "learning_rate": .0001, "replay_weight": .5},
    "full_replay_fast": {"scope": "full_network", "learning_rate": .0003, "replay_weight": .5},
}


def integer_clock(values, *, strict):
    raw = np.asarray(values)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer) or (raw < np.iinfo(np.int64).min).any() or (raw > np.iinfo(np.int64).max).any():
        raise ValueError("Nonoverflowing integer-millisecond vectors required")
    clock = raw.astype(np.int64)
    if (clock[1:] <= clock[:-1]).any() if strict else (clock[1:] < clock[:-1]).any():
        raise ValueError("Chronological clocks required")
    return clock


def decoded_release_clock(values):
    """Decode the parquet release column without rounding fractional timestamps.

    Canonical feature files store some event-time columns as binary64 because
    unresolved rows were originally nullable. Eligible resolved rows must be
    finite exact integers strictly inside binary64's consecutive-integer range.
    """
    raw = np.asarray(values)
    if raw.dtype == np.dtype("float64"):
        if raw.ndim != 1 or not np.isfinite(raw).all() or (np.abs(raw) >= 2**53).any() or (raw != np.trunc(raw)).any():
            raise ValueError("Exactly integral finite binary64 release milliseconds required")
        raw = raw.astype(np.int64)
    return integer_clock(raw, strict=False)


def released_window(origins, releases, anchor_ms, *, window_seconds=1800, observation_delay_ms=100):
    origins, releases = integer_clock(origins, strict=True), integer_clock(releases, strict=False)
    if origins.shape != releases.shape or (releases <= origins).any():
        raise ValueError("Aligned strictly delayed label releases required")
    if isinstance(anchor_ms, (bool, np.bool_)) or not isinstance(anchor_ms, (int, np.integer)) or window_seconds <= 0 or observation_delay_ms < 0:
        raise ValueError("Integer update time, positive window and nonnegative observation delay required")
    # Subtract from the Python-integer anchor, avoiding overflowing release + delay.
    return ((origins >= int(anchor_ms) - int(window_seconds * 1000)) & (origins < anchor_ms)
            & (releases <= int(anchor_ms) - int(observation_delay_ms)))


class PrequentialState:
    def __init__(self, original, variant):
        import torch

        if variant not in VARIANTS or original.members != 1 or set(original.priors) != {0, 1}:
            raise ValueError("A registered variant and original two-asset single-member checkpoint required")
        prior_rebasing_offsets(np.array([original.priors[a] for a in (0, 1)]), np.array([original.priors[a] for a in (0, 1)]))
        self.base = copy.deepcopy(original)
        self.variant = variant
        self.scope = VARIANTS[variant]["scope"]
        self.replay_weight = VARIANTS[variant]["replay_weight"]
        self.bias = torch.nn.Parameter(torch.zeros(2, 3), requires_grad=self.scope == "asset_bias")
        for parameter in self.base.network.parameters():
            parameter.requires_grad_(self.scope == "full_network")
        self.parameters = list(self.base.network.parameters()) if self.scope == "full_network" else [self.bias]
        self.optimizer = None if self.scope == "none" else torch.optim.AdamW(
            self.parameters, lr=VARIANTS[variant]["learning_rate"], weight_decay=.01 if self.scope == "full_network" else 0.)
        self.updates = 0

    def logits(self, matrix, assets):
        return self.base.network(matrix).squeeze(1) + self.bias[assets]

    def predict_matrix(self, matrix, asset, *, batch_size=2048):
        import torch

        if asset not in (0, 1) or matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError("Finite matrix and registered asset required")
        if not len(matrix):
            return np.empty((0, 3))
        self.base.network.eval()
        result = []
        with torch.no_grad():
            for start in range(0, len(matrix), batch_size):
                x = torch.from_numpy(matrix[start:start + batch_size])
                a = torch.full((len(x),), asset, dtype=torch.long)
                result.append(torch.softmax(self.logits(x, a), dim=-1).numpy())
        return natural_posteriors(np.concatenate(result), self.base.priors[asset])

    def accumulate_loss_gradient(self, matrix, labels, assets, weights, offsets, *, coefficient, batch_size=1024):
        """Accumulate a full-cohort mean, including a short final batch correctly."""
        import torch

        matrix, labels, assets, weights = (np.asarray(v) for v in (matrix, labels, assets, weights))
        if matrix.ndim != 2 or not len(matrix) or any(v.shape != (len(matrix),) for v in (labels, assets, weights)) or not np.isfinite(matrix).all() or not np.isin(labels, [-1, 0, 1]).all() or not np.isin(assets, [0, 1]).all() or not np.isfinite(weights).all() or (weights < 0).any() or coefficient < 0:
            raise ValueError("Aligned finite weighted supervised rows required")
        if np.asarray(offsets).shape != (2, 3) or not np.isfinite(offsets).all():
            raise ValueError("Finite per-asset logit offsets required")
        shift = torch.tensor(offsets, dtype=torch.float32)
        total = 0.
        for start in range(0, len(matrix), batch_size):
            stop = start + batch_size
            x = torch.from_numpy(matrix[start:stop].astype(np.float32, copy=False))
            a = torch.tensor(assets[start:stop], dtype=torch.long)
            y = torch.tensor(labels[start:stop] + 1, dtype=torch.long)
            w = torch.tensor(weights[start:stop], dtype=torch.float32)
            loss = coefficient * (torch.nn.functional.cross_entropy(self.logits(x, a) + shift[a], y, reduction="none") * w).sum() / len(matrix)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite prequential supervised loss")
            loss.backward()
            total += float(loss.detach())
        return total

    def update(self, recent_matrix, recent_labels, recent_assets, replay):
        import torch

        if self.optimizer is None:
            return {"optimizer_step": False, "reason": "frozen_control"}
        for asset in (0, 1):
            if not all(np.any((recent_assets == asset) & (recent_labels == label)) for label in (-1, 0, 1)):
                return {"optimizer_step": False, "reason": "missing_recent_asset_class"}
        priors, weights = balanced_asset_weights(recent_labels, recent_assets)
        offset = prior_rebasing_offsets(np.array([self.base.priors[a] for a in (0, 1)]), np.array([priors[a] for a in (0, 1)]))
        self.base.network.train()
        self.optimizer.zero_grad()
        loss = self.accumulate_loss_gradient(recent_matrix, recent_labels, recent_assets, weights, offset, coefficient=1 - self.replay_weight)
        if self.replay_weight:
            rx, ry, ra = replay
            counts = [np.count_nonzero((ra == a) & (ry == y)) for a in (0, 1) for y in (-1, 0, 1)]
            if min(counts) <= 0 or len(set(counts)) != 1:
                raise ValueError("Historical replay must have equal nonempty asset/class cells")
            loss += self.accumulate_loss_gradient(rx, ry, ra, np.ones(len(ry)), np.zeros((2, 3)), coefficient=self.replay_weight)
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, 5)
        if not torch.isfinite(norm):
            raise ValueError("Nonfinite prequential gradient")
        self.optimizer.step()
        self.updates += 1
        self.base.network.eval()
        return {"optimizer_step": True, "loss": loss, "gradient_norm_before_clip": float(norm),
            "recent_priors": {str(a): priors[a].tolist() for a in (0, 1)}}

    def save(self, folder):
        import torch

        folder = Path(folder)
        self.base.save(folder)
        torch.save({"semantics": SEMANTICS, "variant": self.variant, "bias": self.bias.detach(), "updates": self.updates,
            "optimizer_state": None if self.optimizer is None else self.optimizer.state_dict()}, folder / "online.pt")

    @classmethod
    def load(cls, folder):
        import torch

        folder = Path(folder)
        saved = torch.load(folder / "online.pt", weights_only=True, map_location="cpu")
        if saved["semantics"] != SEMANTICS:
            raise ValueError("Prequential checkpoint semantics changed")
        state = cls(PooledForecaster.load(folder), saved["variant"])
        with torch.no_grad():
            state.bias.copy_(saved["bias"])
        if state.optimizer is not None:
            state.optimizer.load_state_dict(saved["optimizer_state"])
        state.updates = int(saved["updates"])
        return state


def run_prequential(original, variant, stream_features, stream_labels, stream_times, release_times,
                    query_features, query_times, replay, anchors, *, publication_delay_ms=10000,
                    window_seconds=1800, observation_delay_ms=100, enforce_timing=True):
    """Forecast each interval with the last published state, then update it.

    Future rows may exist in this offline container but no fitting, weighting,
    normalization, or state-selection operation reads their values prematurely.
    """
    import torch

    if any(set(values) != set(SYMBOLS) for values in (stream_features, stream_labels, stream_times, release_times, query_features, query_times)):
        raise ValueError("Both assets and aligned stream/query mappings required")
    anchors = integer_clock(anchors, strict=True)
    if not len(anchors) or publication_delay_ms < 1 or (len(anchors) > 1 and np.min(np.diff(anchors)) <= publication_delay_ms):
        raise ValueError("Separated update/publication times required")
    if int(anchors[-1]) > np.iinfo(np.int64).max - publication_delay_ms:
        raise ValueError("Nonoverflowing publication clocks required")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260908)
    state = PrequentialState(original, variant)
    if variant == "frozen":
        forecasts = {s: state.base.predict_proba(query_features[s], a) for a, s in enumerate(SYMBOLS)}
        return forecasts, state, {"variant": variant, "optimizer_steps": 0, "updates": [], "frozen_prediction_path": True}
    sx, sy, st, sr, qx, qt = {}, {}, {}, {}, {}, {}
    for asset, symbol in enumerate(SYMBOLS):
        st[symbol], sr[symbol] = integer_clock(stream_times[symbol], strict=True), integer_clock(release_times[symbol], strict=False)
        qt[symbol] = integer_clock(query_times[symbol], strict=True)
        sy[symbol] = np.asarray(stream_labels[symbol])
        if sy[symbol].shape != st[symbol].shape or len(stream_features[symbol]) != len(st[symbol]) or len(query_features[symbol]) != len(qt[symbol]) or not np.isin(sy[symbol], [-1, 0, 1]).all():
            raise ValueError("Every stream and query row requires aligned labels and clocks")
        # Validate release ordering and shape before any state mutation.
        released_window(st[symbol], sr[symbol], int(anchors[0]), window_seconds=window_seconds, observation_delay_ms=observation_delay_ms)
        sx[symbol] = original.matrix(stream_features[symbol], asset)
        qx[symbol] = original.matrix(query_features[symbol], asset)
    forecasts = {s: np.full((len(qt[s]), 3), np.nan) for s in SYMBOLS}
    cursor = {s: 0 for s in SYMBOLS}
    history = []
    for anchor in anchors:
        published = int(anchor) + publication_delay_ms
        # Supply forecasts made before this update could be published, using
        # exactly the prior state, including the publication-delay interval.
        for asset, symbol in enumerate(SYMBOLS):
            stop = int(np.searchsorted(qt[symbol], published, side="left"))
            if stop > cursor[symbol]:
                forecasts[symbol][cursor[symbol]:stop] = state.predict_matrix(qx[symbol][cursor[symbol]:stop], asset)
                cursor[symbol] = stop
        begin = time.monotonic()
        masks = {s: released_window(st[s], sr[s], int(anchor), window_seconds=window_seconds, observation_delay_ms=observation_delay_ms) for s in SYMBOLS}
        matrix = np.concatenate([sx[s][masks[s]] for s in SYMBOLS])
        labels = np.concatenate([sy[s][masks[s]] for s in SYMBOLS])
        assets = np.concatenate([np.full(int(masks[s].sum()), a, dtype=int) for a, s in enumerate(SYMBOLS)])
        update = state.update(matrix, labels, assets, replay)
        elapsed = time.monotonic() - begin
        record = {"anchor_ms": int(anchor), "published_ms": published, "update_seconds": elapsed,
            "recent_rows": {s: int(masks[s].sum()) for s in SYMBOLS},
            "last_used_release_ms": {s: int(sr[s][masks[s]].max()) if masks[s].any() else None for s in SYMBOLS}, **update}
        history.append(record)
        if enforce_timing and elapsed * 1000 > publication_delay_ms:
            raise ValueError("Measured update exceeded its registered publication allowance")
    for asset, symbol in enumerate(SYMBOLS):
        forecasts[symbol][cursor[symbol]:] = state.predict_matrix(qx[symbol][cursor[symbol]:], asset)
        if not np.isfinite(forecasts[symbol]).all() or (forecasts[symbol] < 0).any() or not np.allclose(forecasts[symbol].sum(axis=1), 1):
            raise ValueError("Every unchanged query requires a finite normalized forecast")
    return forecasts, state, {"semantics": SEMANTICS, "variant": variant, "optimizer_steps": state.updates,
        "updates": history, "max_update_seconds": max(r["update_seconds"] for r in history),
        "publication_delay_ms": publication_delay_ms, "observation_delay_ms": observation_delay_ms,
        "frozen_normalizer_and_original_priors": True}


def save_history(path, history):
    Path(path).write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
