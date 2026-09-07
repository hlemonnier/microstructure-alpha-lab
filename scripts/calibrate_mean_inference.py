#!/usr/bin/env python3
"""Reproducible null stress test; PYTHONPATH=src python scripts/calibrate_mean_inference.py.

These experiments describe a calibration domain, not a guarantee for arbitrary
market dependence, selection bias, nonstationarity, or heavy-tailed processes.
"""
from __future__ import annotations

import argparse
import json
import math
import random

from lob_forge.alpha_factory import (
    effective_folds_ar1_diagnostic,
    one_sided_hac_p_value_mean_le_zero,
    one_sided_separated_batch_t_p_value_mean_le_zero,
)
from lob_forge.statistics import newey_west_standard_error


def run_calibration(samples: int = 10000, seed: int = 20260907) -> dict[str, object]:
    if samples < 100:
        raise ValueError("use at least 100 null replications")
    rng = random.Random(seed)
    scenarios = []
    for n, phi in [(20, 0.0), (20, 0.8), (60, 0.0), (60, 0.8), (20, 0.95), (60, 0.95)]:
        original_rejections = new_rejections = persistence_rejections = 0
        for _ in range(samples):
            state = rng.gauss(0.0, 1.0)
            values = []
            for _ in range(n):
                state = phi * state + math.sqrt(1.0 - phi * phi) * rng.gauss(0.0, 1.0)
                values.append(state)
            se = newey_west_standard_error(values)
            original_p = 0.5 * math.erfc((sum(values) / n / se) / math.sqrt(2.0)) if se > 0 else 1.0
            guarded_p = max(one_sided_hac_p_value_mean_le_zero(values),
                            one_sided_separated_batch_t_p_value_mean_le_zero(values))
            original_rejections += original_p < 0.05
            new_rejections += guarded_p < 0.05
            persistence_rejections += effective_folds_ar1_diagnostic(values) < 8
        rate = new_rejections / samples
        scenarios.append({
            "folds": n, "ar1_phi": phi, "replications": samples,
            "original_hac_z_rejection_rate": original_rejections / samples,
            "guarded_rejection_rate": rate,
            "monte_carlo_standard_error": math.sqrt(rate * (1.0 - rate) / samples),
            "persistence_screen_fraction": persistence_rejections / samples,
        })
    return {
        "seed": seed, "nominal_one_sided_alpha": 0.05,
        "null_model": "stationary Gaussian AR(1), mean=0, unconditional variance=1",
        "method": "separated_batch_t_with_hac_guard_v1",
        "limitations": "This is a declared synthetic calibration domain, not evidence of real strategy alpha or a universal error-control guarantee. The effective-fold screen is an AR(1) approximation estimated noisily from the same sample.",
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    print(json.dumps(run_calibration(args.samples, args.seed), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
