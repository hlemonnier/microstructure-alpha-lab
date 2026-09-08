"""Continuous one-second fading memory from fixed, contractive random maps.

The input normalizer is inherited from an already past-fitted checkpoint. No
labels or validation-dependent fitting are accepted by the state generator.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lob_forge.boundary_prequential import integer_clock

SEMANTICS = "past_normalized_continuous_contracting_reservoir_v1"
BLOCK_WIDTH = 128
LEAKS = np.array([.1, .01], dtype=float)[:, None]
MECHANISMS = ("instant", "ema", "reservoir")


@dataclass
class MemoryState:
    ema: np.ndarray
    reservoir: np.ndarray
    previous_time: int | None = None

    @classmethod
    def zero(cls):
        return cls(np.zeros((2, BLOCK_WIDTH)), np.zeros((2, BLOCK_WIDTH)))


@dataclass
class FadingMemoryMap:
    normalizer: Any
    active: np.ndarray
    columns: list[str]
    projection: np.ndarray
    bias: np.ndarray
    recurrent: np.ndarray

    @classmethod
    def from_checkpoint(cls, original):
        if original.members != 1 or original.hidden_size != 64:
            raise ValueError("The original single-member 64-wide checkpoint is required")
        dimensions = int(original.active.sum()) + 1
        rng = np.random.default_rng(20260908)
        projection = rng.normal(size=(dimensions, 2 * BLOCK_WIDTH)) / np.sqrt(dimensions)
        bias = .1 * rng.normal(size=2 * BLOCK_WIDTH)
        q, r = np.linalg.qr(rng.normal(size=(BLOCK_WIDTH, BLOCK_WIDTH)))
        q *= np.where(np.diag(r) < 0, -1., 1.)[None, :]
        result = cls(copy.deepcopy(original.normalizer), original.active.copy(), original.columns.copy(), projection, bias, .9 * q)
        result.validate()
        return result

    def validate(self):
        if not self.columns or len(set(self.columns)) != len(self.columns) or self.active.shape != (len(self.columns),) or self.active.dtype != bool:
            raise ValueError("The original unique feature schema and active mask are required")
        if self.projection.shape != (int(self.active.sum()) + 1, 2 * BLOCK_WIDTH) or self.bias.shape != (2 * BLOCK_WIDTH,) or self.recurrent.shape != (BLOCK_WIDTH, BLOCK_WIDTH):
            raise ValueError("Registered reservoir dimensions required")
        if any(not np.isfinite(a).all() for a in (self.projection, self.bias, self.recurrent)) or np.linalg.norm(self.recurrent, 2) > .9 + 1e-12:
            raise ValueError("Finite input maps and recurrent operator norm at most .9 required")

    def input_matrix(self, frame, asset):
        if list(frame.columns) != self.columns or asset not in (0, 1):
            raise ValueError("Original feature schema and asset identity required")
        raw = frame.to_numpy(dtype=float)
        if not np.isfinite(raw).all():
            raise ValueError("Finite observed input features required")
        values = self.normalizer.transform(raw[:, self.active])
        return np.column_stack([values, np.full(len(raw), 2 * asset - 1)]).astype(np.float32)

    def observe(self, frame, times, asset, *, state=None):
        clock = integer_clock(times, strict=True)
        if not len(clock) or len(frame) != len(clock):
            raise ValueError("Nonempty aligned one-second observations required")
        carry = MemoryState.zero() if state is None else copy.deepcopy(state)
        if any(a.shape != (2, BLOCK_WIDTH) or not np.isfinite(a).all() or (np.abs(a) > 1 + 1e-12).any() for a in (carry.ema, carry.reservoir)):
            raise ValueError("Finite bounded memory state required")
        if carry.previous_time is not None:
            integer_clock(np.array([carry.previous_time]), strict=True)
        if carry.previous_time is not None and int(clock[0]) <= carry.previous_time:
            raise ValueError("A continued memory stream must advance its clock")
        drive = (self.input_matrix(frame, asset) @ self.projection + self.bias).reshape(-1, 2, BLOCK_WIDTH)
        instant = np.tanh(drive)
        values = {"instant": instant.reshape(len(clock), -1).astype(np.float32),
                  "ema": np.empty((len(clock), 2 * BLOCK_WIDTH), dtype=np.float32),
                  "reservoir": np.empty((len(clock), 2 * BLOCK_WIDTH), dtype=np.float32)}
        resets = 0
        for row, now in enumerate(clock):
            if carry.previous_time is None or int(now) - carry.previous_time != 1000:
                carry.ema.fill(0)
                carry.reservoir.fill(0)
                resets += 1
            carry.ema = (1 - LEAKS) * carry.ema + LEAKS * instant[row]
            carry.reservoir = (1 - LEAKS) * carry.reservoir + LEAKS * np.tanh(drive[row] + carry.reservoir @ self.recurrent.T)
            values["ema"][row] = carry.ema.reshape(-1)
            values["reservoir"][row] = carry.reservoir.reshape(-1)
            carry.previous_time = int(now)
        for value in values.values():
            if not np.isfinite(value).all() or (np.abs(value) > 1).any():
                raise ValueError("Every generated coordinate must remain finite and bounded")
        return values, carry, {"rows": len(clock), "state_resets": resets, "last_decision_time": int(clock[-1])}

    def save(self, path):
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"semantics": SEMANTICS, "generator": self}, path)

    @classmethod
    def load(cls, path):
        import joblib

        stored = joblib.load(path)
        if stored["semantics"] != SEMANTICS or not isinstance(stored["generator"], cls):
            raise ValueError("Fading-memory artifact semantics changed")
        stored["generator"].validate()
        return stored["generator"]


def append_memory(frame, values):
    if np.asarray(values).shape != (len(frame), 2 * BLOCK_WIDTH) or not np.isfinite(values).all():
        raise ValueError("Exactly 256 aligned finite memory coordinates required")
    columns = [f"fading_memory_{index:03}" for index in range(2 * BLOCK_WIDTH)]
    if set(columns).intersection(frame.columns):
        raise ValueError("Memory feature names must not collide with original observations")
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(values, columns=columns)], axis=1)


def observe_selected(generator, frame, times, asset, selected, *, chunk_rows=2048):
    """Advance on every observed second, retaining only chosen supervised rows."""
    clock, keep = integer_clock(times, strict=True), np.asarray(selected)
    if len(frame) != len(clock) or keep.shape != clock.shape or keep.dtype != bool or not keep.any():
        raise ValueError("Aligned observations and a nonempty Boolean selection required")
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 1:
        raise ValueError("A positive integer chunk size is required")
    chunks = {name: [] for name in MECHANISMS}
    carry, resets = None, 0
    for start in range(0, len(clock), chunk_rows):
        end = min(start + chunk_rows, len(clock))
        values, carry, record = generator.observe(frame.iloc[start:end], clock[start:end], asset, state=carry)
        resets += record["state_resets"]
        if keep[start:end].any():
            for name in MECHANISMS:
                chunks[name].append(values[name][keep[start:end]].copy())
    return {name: np.concatenate(chunks[name]) for name in MECHANISMS}, {
        "observed_rows": len(clock), "retained_rows": int(keep.sum()), "state_resets": resets,
        "chunk_rows": chunk_rows, "first_observation_time": int(clock[0]), "last_observation_time": int(clock[-1]),
    }
