from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Iterable, Sequence


VALID_SELECTION_METRICS = {
    "validation_macro_f1",
    "validation_balanced_accuracy",
    "validation_net_pnl",
    "validation_gross_pnl",
}


@dataclass(frozen=True)
class EventTimeContract:
    decision_time: int
    entry_time: int
    exit_time: int
    latency_ms: int
    horizon_ms: int
    timestamp_column: str
    receive_timestamp_column: str | None
    tie_break: str
    stale_after_ms: int | None


@dataclass(frozen=True)
class CausalSample:
    decision_time: int
    features: dict[str, float]
    entry_row: dict[str, str]
    exit_row: dict[str, str]
    contract: EventTimeContract


@dataclass(frozen=True)
class PurgedSplit:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_purged: int
    validation_purged: int
    embargo: int

    @property
    def raw_indices(self) -> tuple[int, ...]:
        return self.train_indices + self.validation_indices + self.test_indices


def assert_valid_selection_metric(metric: str) -> None:
    if metric.startswith("test_"):
        raise ValueError(f"{metric!r} is not a valid model-selection metric; choose on training/validation only")
    if metric not in VALID_SELECTION_METRICS:
        allowed = ", ".join(sorted(VALID_SELECTION_METRICS))
        raise ValueError(f"selection metric must be one of: {allowed}")


def construct_causal_samples(
    events: Sequence[dict[str, str]],
    *,
    decision_times: Sequence[int],
    feature_columns: Sequence[str],
    timestamp_column: str = "event_time",
    receive_timestamp_column: str | None = None,
    latency_ms: int = 0,
    horizon_ms: int = 1000,
    stale_after_ms: int | None = None,
) -> list[CausalSample]:
    """Build event-time samples with explicit <= decision feature visibility.

    Features are copied from the last event whose selected timestamp is less
    than or equal to the decision time. Entry and exit rows are resolved from
    the first executable event at or after decision+latency and
    decision+latency+horizon. Ties use stable input order after timestamp sort.
    """
    if latency_ms < 0:
        raise ValueError("latency_ms must be non-negative")
    if horizon_ms <= 0:
        raise ValueError("horizon_ms must be positive")
    if stale_after_ms is not None and stale_after_ms < 0:
        raise ValueError("stale_after_ms must be non-negative")
    if not events:
        raise ValueError("events cannot be empty")
    timestamp_key = receive_timestamp_column or timestamp_column
    indexed = sorted(enumerate(events), key=lambda item: (_event_time(item[1], timestamp_key), item[0]))
    ordered_events = [row for _, row in indexed]
    times = [_event_time(row, timestamp_key) for row in ordered_events]

    samples: list[CausalSample] = []
    for decision_time in decision_times:
        feature_pos = bisect_right_int(times, decision_time) - 1
        if feature_pos < 0:
            continue
        feature_row = ordered_events[feature_pos]
        feature_time = times[feature_pos]
        if stale_after_ms is not None and decision_time - feature_time > stale_after_ms:
            continue

        entry_target = decision_time + latency_ms
        exit_target = entry_target + horizon_ms
        entry_pos = bisect_left(times, entry_target)
        exit_pos = bisect_left(times, exit_target)
        if entry_pos >= len(ordered_events) or exit_pos >= len(ordered_events):
            continue

        features = {column: _float(feature_row.get(column, "0")) for column in feature_columns}
        samples.append(
            CausalSample(
                decision_time=decision_time,
                features=features,
                entry_row=ordered_events[entry_pos],
                exit_row=ordered_events[exit_pos],
                contract=EventTimeContract(
                    decision_time=decision_time,
                    entry_time=times[entry_pos],
                    exit_time=times[exit_pos],
                    latency_ms=latency_ms,
                    horizon_ms=horizon_ms,
                    timestamp_column=timestamp_column,
                    receive_timestamp_column=receive_timestamp_column,
                    tie_break="stable_input_order_after_timestamp_sort",
                    stale_after_ms=stale_after_ms,
                ),
            )
        )
    return samples


def purged_walk_forward_indices(
    row_count: int,
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    step_size: int | None = None,
    sequence_length: int = 1,
    prediction_horizon: int = 1,
    latency_buffer: int = 0,
    embargo: int = 0,
) -> list[PurgedSplit]:
    if min(row_count, train_size, validation_size, test_size, sequence_length, prediction_horizon) <= 0:
        raise ValueError("row_count and split sizes must be positive")
    if latency_buffer < 0 or embargo < 0:
        raise ValueError("latency_buffer and embargo must be non-negative")
    purge = max(0, sequence_length - 1 + prediction_horizon + latency_buffer)
    step = step_size or test_size
    if step <= 0:
        raise ValueError("step_size must be positive")
    total = train_size + validation_size + test_size
    splits: list[PurgedSplit] = []
    start = 0
    while start + total <= row_count:
        train_end = start + train_size
        validation_start = train_end
        validation_end = validation_start + validation_size
        test_start = validation_end
        test_end = test_start + test_size

        train_keep_end = max(start, validation_start - purge - embargo)
        validation_keep_end = max(validation_start, test_start - purge - embargo)
        train = tuple(range(start, train_keep_end))
        validation = tuple(range(validation_start, validation_keep_end))
        test = tuple(range(test_start, test_end))
        splits.append(
            PurgedSplit(
                train_indices=train,
                validation_indices=validation,
                test_indices=test,
                train_purged=train_size - len(train),
                validation_purged=validation_size - len(validation),
                embargo=embargo,
            )
        )
        start += step
    return splits


def assert_disjoint_partitions(splits: Iterable[PurgedSplit]) -> None:
    for split in splits:
        seen: set[int] = set()
        for name, indices in (
            ("train", split.train_indices),
            ("validation", split.validation_indices),
            ("test", split.test_indices),
        ):
            overlap = seen & set(indices)
            if overlap:
                raise ValueError(f"{name} indices overlap prior partition: {sorted(overlap)[:5]}")
            seen.update(indices)


def bisect_right_int(values: Sequence[int], value: int) -> int:
    lo = 0
    hi = len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if value < values[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _event_time(row: dict[str, str], column: str) -> int:
    try:
        return int(float(row[column]))
    except KeyError as exc:
        raise ValueError(f"timestamp column {column!r} missing") from exc


def _float(value: str | None) -> float:
    if value is None:
        return 0.0
    if value == "":
        return 0.0
    return float(value)
