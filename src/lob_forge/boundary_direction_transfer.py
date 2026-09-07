"""Transfer only direction information between simultaneous asset forecasts."""

from __future__ import annotations

import numpy as np

STRENGTHS = (-0.5, -0.25, -0.125, 0.0, 0.125, 0.25, 0.5, 0.75, 1.0)


def transfer_direction(own_probabilities, peer_probabilities, *, strength):
    """Add beta times the peer's directional log odds, retaining own move mass.

The caller must align predictions available at the same decision timestamp. No
realized peer outcome is used. Beta zero preserves source probabilities exactly.
    """
    own, peer = np.asarray(own_probabilities, dtype=float), np.asarray(peer_probabilities, dtype=float)
    if own.ndim != 2 or own.shape[1] != 3 or peer.shape != own.shape or any(
        not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1) for p in (own, peer)
    ):
        raise ValueError("Aligned finite normalized three-class probabilities are required")
    if strength not in STRENGTHS:
        raise ValueError("Use a registered direction-transfer strength")
    if strength == 0:
        return own.copy()
    eps = np.finfo(float).eps
    log_odds = np.log(np.maximum(own[:, 2], eps)) - np.log(np.maximum(own[:, 0], eps))
    log_odds += strength * (np.log(np.maximum(peer[:, 2], eps)) - np.log(np.maximum(peer[:, 0], eps)))
    direction = np.exp(-np.logaddexp(0, -log_odds))
    movement = own[:, 0] + own[:, 2]
    return np.column_stack([movement * (1 - direction), own[:, 1], movement * direction])
