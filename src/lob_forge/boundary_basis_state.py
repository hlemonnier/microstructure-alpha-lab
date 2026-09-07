"""Causal price-basis innovations relative to their slowly varying level."""

from __future__ import annotations

import numpy as np
import pandas as pd

HALF_LIVES = (10, 60, 300)
BASIS_STATE_SEMANTICS = "strict_observed_basis_prior_moments_v1"


def prior_basis_moments(values, decision_times, available, *, half_life_seconds):
    """Compute moments before the current observed basis enters the filter.

    Each uninterrupted one-second segment resets its state. Missing source
    observations have zero mass. Moment coordinates are centered on the first
    available basis to avoid subtracting squares of a large constant premium.
    The first available observation has no historical innovation and is flagged.
    """
    from scipy.signal import lfilter

    z, clock, valid = np.asarray(values, float), np.asarray(decision_times), np.asarray(available)
    if (z.ndim != 1 or not len(z) or clock.shape != z.shape or valid.shape != z.shape
        or not np.issubdtype(clock.dtype, np.integer) or (np.diff(clock) <= 0).any()
        or valid.dtype != bool or not np.isfinite(z).all()
        or not np.isfinite(half_life_seconds) or half_life_seconds <= 0):
        raise ValueError("Finite aligned bases, integer clocks, boolean availability and positive half life required")
    decay = np.exp(-np.log(2) / half_life_seconds)
    result = np.zeros((len(z), 5))
    starts = np.r_[0, np.flatnonzero(np.diff(clock) != 1000) + 1, len(clock)]
    for left, right in zip(starts[:-1], starts[1:], strict=True):
        present = valid[left:right]
        first = np.flatnonzero(present)
        if not len(first):
            continue
        anchor = z[left + first[0]]
        centered = np.where(present, z[left:right] - anchor, 0)
        mass = lfilter([1.0], [1.0, -decay], present.astype(float))
        first_moment = lfilter([1.0], [1.0, -decay], centered)
        second_moment = lfilter([1.0], [1.0, -decay], centered * centered)
        mass, first_moment, second_moment = [np.r_[0.0, a[:-1]] for a in (mass, first_moment, second_moment)]
        history = present & (mass > 0)
        mean = np.divide(first_moment, mass, out=np.zeros_like(mass), where=mass > 0)
        second = np.divide(second_moment, mass, out=np.zeros_like(mass), where=mass > 0)
        std = np.sqrt(np.maximum(0, second - mean * mean))
        innovation = centered - mean
        block = np.column_stack([anchor + mean, innovation, innovation / np.maximum(std, 0.001),
                                 np.log1p(std), history.astype(float)])
        result[left:right] = np.where(history[:, None], block, 0)
    return result


def basis_state_features(frame, decision_times):
    """Read only delayed foreign/spot observations already present in the input."""
    required = ("foreign_venue_mid_basis_bps", "foreign_venue_available", "aux_last_basis_bps", "aux_available")
    if any(c not in frame for c in required):
        raise ValueError("Frozen delayed foreign quote and spot trade fields required")
    if not np.isfinite(frame[list(required)].to_numpy()).all():
        raise ValueError("Finite observed source fields required")
    available = frame.foreign_venue_available.to_numpy() == 1
    spot_available = frame.aux_available.to_numpy() == 1
    foreign = frame.foreign_venue_mid_basis_bps.to_numpy()
    if (foreign[available] <= -10000).any():
        raise ValueError("Available foreign/local price ratios must be positive")
    foreign = 10000 * np.log1p(np.where(available, foreign, 0) / 10000)
    spot = frame.aux_last_basis_bps.to_numpy()
    streams = {"bybit": (foreign, available), "bybit_spot": (foreign - spot, available & spot_available)}
    columns = {}
    for name, (values, present) in streams.items():
        columns[f"basis_state_{name}_available"] = present.astype(float)
        for half_life in HALF_LIVES:
            moments = prior_basis_moments(values, decision_times, present, half_life_seconds=half_life)
            for i, field in enumerate(("mean_bps", "innovation_bps", "innovation_z", "log_std_bps", "history_available")):
                columns[f"basis_state_{name}_{field}_{half_life}s"] = moments[:, i]
    return pd.DataFrame(columns, index=frame.index)
