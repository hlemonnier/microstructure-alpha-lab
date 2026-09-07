"""Predetermined history breadth and clock sampling for forecast development."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np


def history_cohorts(assessment_day):
    day = date.fromisoformat(assessment_day)
    return {
        name: {"train_dates": [(day + timedelta(days=i)).isoformat() for i in range(-length - 1, -1)], "stride_seconds": stride}
        for name, length, stride in (("recent4_stride4", 4, 4), ("wide14_stride4", 14, 4), ("wide14_stride14", 14, 14))
    }


def calendar_stride_mask(decision_times, day_start_ms, *, stride_seconds):
    clock = np.asarray(decision_times)
    if stride_seconds not in (4, 14) or clock.ndim != 1 or not np.isfinite(clock).all() or not np.equal(clock, np.floor(clock)).all():
        raise ValueError("Use a registered stride and finite integer-millisecond clocks")
    if (np.diff(clock) <= 0).any() or (clock < day_start_ms).any() or (clock >= day_start_ms + 86400000).any():
        raise ValueError("Clocks must be increasing within the registered UTC day")
    # Calendar phase is fixed before fitting and never depends on a label,
    # volatility, available sample count, or an assessment outcome.
    return (clock.astype(np.int64) - day_start_ms) % (1000 * stride_seconds) == 0
