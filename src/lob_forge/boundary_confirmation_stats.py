"""Paired date-cluster uncertainty for a fixed independent forecast comparison."""

from __future__ import annotations

import numpy as np

MODELS = ("original_reference", "matched_data_control", "candidate")
METRICS = ("balanced_accuracy", "natural_accuracy", "log_loss", "macro_f1")


def cluster_interval(values, *, block_length=1, resamples=10000, seed=20260907):
    differences = np.asarray(values, dtype=float)
    n = len(differences)
    if (
        differences.ndim != 1
        or n < 2
        or not np.isfinite(differences).all()
        or not 1 <= block_length <= n
        or resamples < 100
    ):
        raise ValueError("Finite paired date deltas and valid bootstrap dimensions required")
    rng = np.random.default_rng(seed)
    blocks = (n + block_length - 1) // block_length
    starts = rng.integers(0, n, size=(resamples, blocks))
    indices = ((starts[:, :, None] + np.arange(block_length)) % n).reshape(resamples, -1)[:, :n]
    estimates = differences[indices].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975], method="linear")
    return {
        "mean": float(differences.mean()),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "distinct_dates": n,
        "block_length": block_length,
        "resamples": resamples,
        "seed": seed,
    }


def confirmation_summary(records, expected_dates):
    dates = sorted(expected_dates)
    symbols = ["BTCUSDT", "ETHUSDT"]
    expected = {(symbol, day) for symbol in symbols for day in dates}
    observed = {(r["symbol"], r["assessment_date"]) for r in records}
    if len(dates) < 20 or len(set(dates)) != len(dates) or observed != expected or len(records) != len(expected):
        raise ValueError("Complete paired assessments for at least twenty unique dates and both assets are required")
    for record in records:
        for model in MODELS:
            if any(not np.isfinite(record[model][metric]) for metric in METRICS):
                raise ValueError("Every registered metric must be finite")
    means = {
        model: {metric: float(np.mean([r[model][metric] for r in records])) for metric in METRICS} for model in MODELS
    }
    by_asset = {
        symbol: {
            model: {
                metric: float(np.mean([r[model][metric] for r in records if r["symbol"] == symbol]))
                for metric in METRICS
            }
            for model in MODELS
        }
        for symbol in symbols
    }
    paired = []
    for day in dates:
        rows = [r for r in records if r["assessment_date"] == day]
        paired.append(
            {
                "date": day,
                **{
                    reference: float(
                        np.mean([r["candidate"]["balanced_accuracy"] - r[reference]["balanced_accuracy"] for r in rows])
                    )
                    for reference in ["original_reference", "matched_data_control"]
                },
            }
        )
    uncertainty = {
        reference: {
            "date_cluster": cluster_interval([r[reference] for r in paired]),
            "three_day_blocks": cluster_interval([r[reference] for r in paired], block_length=3),
        }
        for reference in ["original_reference", "matched_data_control"]
    }
    primary = uncertainty["original_reference"]["date_cluster"]
    asset_gains = {
        symbol: by_asset[symbol]["candidate"]["balanced_accuracy"]
        - by_asset[symbol]["original_reference"]["balanced_accuracy"]
        for symbol in symbols
    }
    loss_ratio = means["candidate"]["log_loss"] / means["original_reference"]["log_loss"]
    gates = {
        "at_least_twenty_dates": len(dates) >= 20,
        "mean_balanced_accuracy_gain_at_least_five_points": primary["mean"] >= 0.05,
        "date_cluster_lower_bound_positive": primary["lower_95"] > 0,
        "both_assets_improve": all(value > 0 for value in asset_gains.values()),
        "log_loss_ratio_at_most_1_01": loss_ratio <= 1.01,
    }
    return {
        "asset_date_assessments": len(records),
        "distinct_dates": len(dates),
        "mean_metrics": means,
        "by_asset": by_asset,
        "paired_date_balanced_accuracy_deltas": paired,
        "uncertainty": uncertainty,
        "asset_balanced_accuracy_gains": asset_gains,
        "log_loss_ratio": loss_ratio,
        "gates": gates,
        "substantial_predictive_gain_confirmed": all(gates.values()),
        "economic_performance_confirmed": False,
        "uncertainty_scope": "Dates are the resampling units; assets stay paired within each date. The three-day block interval checks sensitivity to short serial dependence. These intervals do not eliminate longer regime dependence.",
    }
