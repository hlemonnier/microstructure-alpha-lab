"""Historical feature-budget and nested-context controls for fixed transformers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lob_forge.boundary_tabicl import SYMBOLS, select_balanced_context

SEMANTICS = "historical_balanced_bin_information_with_correlation_filter_v1"


def balanced_bin_information(codes, labels, assets, *, bins):
    """Average I(bin; class | asset) after giving each asset/class equal mass."""
    code, y, a = np.asarray(codes), np.asarray(labels), np.asarray(assets)
    if code.ndim != 1 or not len(code) or y.shape != code.shape or a.shape != code.shape or not np.issubdtype(code.dtype, np.integer):
        raise ValueError("Aligned nonempty integer bin codes, labels and asset identities required")
    if bins < 1 or (code < 0).any() or (code >= bins).any() or not np.isin(y, [-1, 0, 1]).all() or not np.isin(a, [0, 1]).all():
        raise ValueError("Valid finite bin, class and two-asset identities required")
    scores = []
    for asset in (0, 1):
        counts = np.column_stack([np.bincount(code[(a == asset) & (y == label)], minlength=bins) for label in (-1, 0, 1)]).astype(float)
        total = counts.sum(axis=0)
        if (total == 0).any():
            raise ValueError("Every historical asset/class cell must be observed")
        joint = counts / (3 * total)
        expected = joint.sum(axis=1, keepdims=True) / 3
        positive = joint > 0
        denominator = np.broadcast_to(expected, joint.shape)
        scores.append(float(np.sum(joint[positive] * np.log(joint[positive] / denominator[positive]))))
    return max(0., float(np.mean(scores)))


def nested_balanced_context(labels, *, small_rows=3072, large_rows=12288):
    """Preserve the original sample, then uniformly augment within each cell."""
    if isinstance(large_rows, (bool, np.bool_)) or not isinstance(large_rows, (int, np.integer)) or large_rows <= small_rows or large_rows % 6:
        raise ValueError("A larger exact context budget divisible by six is required")
    small, population, sampled = select_balanced_context(labels, requested_rows=small_rows, seed=20260908)
    if any(len(small[s]) != small_rows // 2 for s in SYMBOLS):
        raise ValueError("Every cell must meet the complete smaller context budget")
    rng = np.random.default_rng(20260909)
    large = {}
    for symbol in SYMBOLS:
        y = np.asarray(labels[symbol])
        selected = np.zeros(len(y), dtype=bool)
        selected[small[symbol]] = True
        added = []
        for label in (-1, 0, 1):
            pool = np.flatnonzero((y == label) & ~selected)
            requested = (large_rows - small_rows) // 6
            if len(pool) < requested:
                raise ValueError("Every cell must meet the complete larger context budget")
            added.append(rng.choice(pool, requested, replace=False))
        large[symbol] = np.sort(np.concatenate([small[symbol], *added]))
        if len(np.unique(large[symbol])) != large_rows // 2 or not np.isin(small[symbol], large[symbol]).all():
            raise ValueError("Expanded contexts must contain every original row without duplicates")
    return {small_rows: small, large_rows: large}, population, sampled


@dataclass
class HistoricalFeatureBudget:
    columns: list[str]
    selected_columns: list[str]
    random_columns: list[str]
    ranking: list[dict]
    bin_edges: list[list[float]]
    correlation_limit: float
    diversity_selected: int

    @classmethod
    def fit(cls, features, labels, *, budget=96, quantile_bins=32, correlation_limit=.95):
        if set(features) != set(SYMBOLS) or set(labels) != set(SYMBOLS) or isinstance(budget, (bool, np.bool_)) or not isinstance(budget, (int, np.integer)) or budget < 1 or isinstance(quantile_bins, (bool, np.bool_)) or not isinstance(quantile_bins, (int, np.integer)) or not 2 <= quantile_bins <= 254 or not 0 < correlation_limit <= 1:
            raise ValueError("Both assets and valid historical feature-selection controls required")
        columns = list(features[SYMBOLS[0]].columns)
        if not columns or len(set(columns)) != len(columns) or any(list(features[s].columns) != columns or len(features[s]) != len(labels[s]) for s in SYMBOLS):
            raise ValueError("Identical feature schemas and aligned historical labels required")
        raw = pd.concat([features[s] for s in SYMBOLS], ignore_index=True).to_numpy(dtype=float)
        y = np.concatenate([np.asarray(labels[s]) for s in SYMBOLS])
        assets = np.concatenate([np.full(len(labels[s]), i) for i, s in enumerate(SYMBOLS)])
        if raw.ndim != 2 or not len(raw) or not np.isfinite(raw).all():
            raise ValueError("Finite nonempty historical features required")
        active = np.flatnonzero(raw.max(axis=0) != raw.min(axis=0))
        if len(active) < budget:
            raise ValueError("Not enough nonconstant historical columns for the exact feature budget")
        codes = np.zeros((len(raw), len(columns)), dtype=np.uint8)
        scores = np.zeros(len(columns), dtype=float)
        boundaries = [[] for _ in columns]
        for index in active:
            edges = np.unique(np.quantile(raw[:, index], np.linspace(0, 1, quantile_bins + 1)[1:-1]))
            values = np.searchsorted(edges, raw[:, index], side="right")
            codes[:, index] = values
            boundaries[index] = edges.tolist()
            scores[index] = balanced_bin_information(values, y, assets, bins=len(edges) + 1)
        # Sufficient statistics keep temporary storage bounded. These are
        # correlations of pooled quantile-bin codes, not exact Spearman ranks.
        sums = np.zeros(len(columns), dtype=float)
        cross = np.zeros((len(columns), len(columns)), dtype=float)
        for left in range(0, len(codes), 4096):
            block = codes[left:left + 4096].astype(float)
            sums += block.sum(axis=0)
            cross += block.T @ block
        covariance = cross / len(codes) - np.outer(sums / len(codes), sums / len(codes))
        scale = np.sqrt(np.maximum(np.diag(covariance), 0))
        denominator = np.outer(scale, scale)
        correlation = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0)
        ranked = sorted(active.tolist(), key=lambda i: (-scores[i], i))
        accepted = []
        for index in ranked:
            if not accepted or np.max(np.abs(correlation[index, accepted])) < correlation_limit:
                accepted.append(index)
                if len(accepted) == budget:
                    break
        diversity_selected = len(accepted)
        accepted.extend(i for i in ranked if i not in accepted)
        chosen = sorted(accepted[:budget])
        random = sorted(np.random.default_rng(20260908).choice(active, size=budget, replace=False).tolist())
        ranking = [{"column": columns[i], "original_index": i, "balanced_conditional_information_nats": float(scores[i]),
            "selected": i in chosen} for i in ranked]
        return cls(columns, [columns[i] for i in chosen], [columns[i] for i in random], ranking,
            boundaries, correlation_limit, diversity_selected)

    def transform(self, frame, variant):
        if list(frame.columns) != self.columns or variant not in ("random", "selected"):
            raise ValueError("The original ordered query schema and registered selector variant are required")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("Finite query observations required")
        columns = self.random_columns if variant == "random" else self.selected_columns
        return frame[columns].copy()

    def save(self, path):
        value = {"semantics": SEMANTICS, **vars(self)}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text())
        if value.pop("semantics") != SEMANTICS:
            raise ValueError("Historical feature-selection semantics changed")
        result = cls(**value)
        if len(set(result.columns)) != len(result.columns) or any(len(set(c)) != len(c) or not set(c).issubset(result.columns) for c in (result.selected_columns, result.random_columns)):
            raise ValueError("Stored feature schemas must remain unique subsets of the original")
        return result
