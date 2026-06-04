from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


CLASSES = [-1, 0, 1]
DEFAULT_FEATURES = [
    "microprice_deviation",
    "top_imbalance",
    "top_imbalance_mean_5",
    "quote_ofi_normalized",
    "quote_ofi_5_normalized",
    "mid_return_1",
    "mid_return_5",
    "realized_volatility_5",
    "trade_imbalance",
    "depth_imbalance_1pct",
    "notional_imbalance_1pct",
    "depth_imbalance_5pct",
    "notional_imbalance_5pct",
]
DEFAULT_THRESHOLDS = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75]
DEFAULT_REGIME_FEATURES = [
    "spread",
    "realized_volatility_5",
    "depth_imbalance_1pct",
    "notional_imbalance_1pct",
]


@dataclass(frozen=True)
class Metrics:
    n: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    coverage: float


@dataclass(frozen=True)
class EconomicMetrics:
    signals: int
    trades: int
    fill_rate: float
    gross_pnl: float
    net_pnl: float
    fee_turnover: float
    break_even_taker_fee_bps: float
    mean_net_pnl_per_trade: float
    mean_net_return_bps_per_trade: float
    median_net_pnl_per_trade: float
    median_net_return_bps_per_trade: float
    profit_factor: float
    max_drawdown_pnl: float
    sharpe_per_trade: float
    win_rate: float


@dataclass(frozen=True)
class BaselineResult:
    name: str
    feature: str
    threshold: float
    train: Metrics
    validation: Metrics
    test: Metrics
    train_economics: EconomicMetrics
    validation_economics: EconomicMetrics
    test_economics: EconomicMetrics


@dataclass(frozen=True)
class RuleEvaluation:
    feature: str
    threshold: float
    metrics: Metrics
    economics: EconomicMetrics


@dataclass(frozen=True)
class FeeSweepResult:
    fee_bps: float
    result: BaselineResult


@dataclass(frozen=True)
class WalkForwardFoldResult:
    fold: int
    train_rows: int
    validation_rows: int
    test_rows: int
    purged_train_rows: int
    purged_validation_rows: int
    result: BaselineResult


@dataclass(frozen=True)
class CalendarWalkForwardFoldResult:
    fold: int
    train_dates: list[str]
    validation_dates: list[str]
    test_dates: list[str]
    train_rows: int
    validation_rows: int
    test_rows: int
    purged_train_rows: int
    purged_validation_rows: int
    result: BaselineResult


@dataclass(frozen=True)
class RegimeEvaluation:
    regime_feature: str
    bucket: int
    lower_bound: float
    upper_bound: float
    row_count: int
    label_down: int
    label_flat: int
    label_up: int
    maker_long_fillable_rate: float
    maker_short_fillable_rate: float
    metrics: Metrics
    economics: EconomicMetrics


@dataclass(frozen=True)
class RegimeBucketSpec:
    feature: str
    bucket: int
    lower_bound: float
    upper_bound: float
    is_last: bool


@dataclass(frozen=True)
class ConditionalResult:
    name: str
    feature: str
    threshold: float
    regime_feature: str
    regime_bucket: int
    regime_lower_bound: float
    regime_upper_bound: float
    train: Metrics
    validation: Metrics
    test: Metrics
    train_economics: EconomicMetrics
    validation_economics: EconomicMetrics
    test_economics: EconomicMetrics


@dataclass(frozen=True)
class ConditionalWalkForwardFoldResult:
    fold: int
    train_rows: int
    validation_rows: int
    test_rows: int
    purged_train_rows: int
    purged_validation_rows: int
    result: ConditionalResult


def run_threshold_baselines(
    feature_csv: Path | str,
    *,
    features: list[str] | None = None,
    thresholds: list[float] | None = None,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    execution_model: str = "taker",
    maker_fee_bps: float = 0.0,
    taker_fee_bps: float = 5.0,
    slippage_bps: float = 0.0,
    sort_by: str = "validation_net_pnl",
) -> list[BaselineResult]:
    rows = _read_rows(Path(feature_csv))
    if len(rows) < 10:
        raise ValueError("need at least 10 rows for baseline evaluation")

    train_rows, validation_rows, test_rows = split_time_ordered(
        rows,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    feature_names = _available_features(rows, features)
    threshold_values = thresholds or DEFAULT_THRESHOLDS

    results = [
        _make_result(
            name="always_flat",
            feature="constant",
            threshold=0.0,
            train_rows=train_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
            taker_fee_bps=taker_fee_bps,
            maker_fee_bps=maker_fee_bps,
            slippage_bps=slippage_bps,
            execution_model=execution_model,
            predictor=lambda row: 0,
        )
    ]

    for feature in feature_names:
        for threshold in threshold_values:
            results.append(
                _make_result(
                    name=f"{feature}_threshold",
                    feature=feature,
                    threshold=threshold,
                    train_rows=train_rows,
                    validation_rows=validation_rows,
                    test_rows=test_rows,
                    taker_fee_bps=taker_fee_bps,
                    maker_fee_bps=maker_fee_bps,
                    slippage_bps=slippage_bps,
                    execution_model=execution_model,
                    predictor=lambda row, feature=feature, threshold=threshold: predict_feature_threshold(
                        row,
                        feature,
                        threshold,
                    ),
                )
            )

    return sorted(results, key=lambda result: _sort_key(result, sort_by), reverse=True)


def run_walk_forward_thresholds(
    feature_csv: Path | str,
    *,
    features: list[str] | None = None,
    thresholds: list[float] | None = None,
    train_size: int = 2400,
    validation_size: int = 1200,
    test_size: int = 1200,
    step_size: int | None = None,
    purge_label_overlap: bool = True,
    execution_model: str = "taker",
    maker_fee_bps: float = 0.0,
    taker_fee_bps: float = 5.0,
    slippage_bps: float = 0.0,
    sort_by: str = "validation_net_pnl",
) -> list[WalkForwardFoldResult]:
    rows = _read_rows(Path(feature_csv))
    if not rows:
        raise ValueError("no rows available for walk-forward evaluation")
    if train_size <= 0 or validation_size <= 0 or test_size <= 0:
        raise ValueError("train_size, validation_size, and test_size must be positive")
    effective_step = step_size or test_size
    if effective_step <= 0:
        raise ValueError("step_size must be positive")

    feature_names = _available_features(rows, features)
    threshold_values = thresholds or DEFAULT_THRESHOLDS
    total_window = train_size + validation_size + test_size
    if len(rows) < total_window:
        raise ValueError("not enough rows for one walk-forward fold")

    folds: list[WalkForwardFoldResult] = []
    start = 0
    fold_idx = 1
    while start + total_window <= len(rows):
        train_raw = rows[start : start + train_size]
        validation_raw = rows[start + train_size : start + train_size + validation_size]
        test_rows = rows[start + train_size + validation_size : start + total_window]

        train_rows = train_raw
        validation_rows = validation_raw
        if purge_label_overlap:
            validation_start_time = _event_time_ms(validation_raw[0])
            test_start_time = _event_time_ms(test_rows[0])
            train_rows = _purge_rows_crossing_boundary(train_raw, validation_start_time)
            validation_rows = _purge_rows_crossing_boundary(validation_raw, test_start_time)

        if not train_rows or not validation_rows or not test_rows:
            start += effective_step
            fold_idx += 1
            continue

        result = _select_threshold_result(
            train_rows=train_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
            feature_names=feature_names,
            threshold_values=threshold_values,
            taker_fee_bps=taker_fee_bps,
            maker_fee_bps=maker_fee_bps,
            slippage_bps=slippage_bps,
            execution_model=execution_model,
            sort_by=sort_by,
        )
        folds.append(
            WalkForwardFoldResult(
                fold=fold_idx,
                train_rows=len(train_rows),
                validation_rows=len(validation_rows),
                test_rows=len(test_rows),
                purged_train_rows=len(train_raw) - len(train_rows),
                purged_validation_rows=len(validation_raw) - len(validation_rows),
                result=result,
            )
        )
        start += effective_step
        fold_idx += 1

    if not folds:
        raise ValueError("no valid walk-forward folds after purging")

    return folds


def run_calendar_walk_forward_thresholds(
    feature_csv: Path | str,
    *,
    features: list[str] | None = None,
    thresholds: list[float] | None = None,
    train_days: int = 20,
    validation_days: int = 5,
    test_days: int = 5,
    step_days: int | None = None,
    purge_label_overlap: bool = True,
    execution_model: str = "taker",
    maker_fee_bps: float = 0.0,
    taker_fee_bps: float = 5.0,
    slippage_bps: float = 0.0,
    sort_by: str = "validation_net_pnl",
) -> list[CalendarWalkForwardFoldResult]:
    rows = _read_rows(Path(feature_csv))
    if not rows:
        raise ValueError("no rows available for calendar walk-forward evaluation")
    if "source_date" not in rows[0]:
        raise ValueError("calendar walk-forward requires a source_date column")
    if train_days <= 0 or validation_days <= 0 or test_days <= 0:
        raise ValueError("train_days, validation_days, and test_days must be positive")
    effective_step = step_days or test_days
    if effective_step <= 0:
        raise ValueError("step_days must be positive")

    feature_names = _available_features(rows, features)
    threshold_values = thresholds or DEFAULT_THRESHOLDS
    dates = sorted({row["source_date"] for row in rows if row.get("source_date")})
    total_days = train_days + validation_days + test_days
    if len(dates) < total_days:
        raise ValueError("not enough source_date groups for one calendar walk-forward fold")

    rows_by_date = {date: [row for row in rows if row.get("source_date") == date] for date in dates}
    folds: list[CalendarWalkForwardFoldResult] = []
    start = 0
    fold_idx = 1
    while start + total_days <= len(dates):
        train_dates = dates[start : start + train_days]
        validation_dates = dates[start + train_days : start + train_days + validation_days]
        test_dates = dates[start + train_days + validation_days : start + total_days]
        train_raw = [row for date in train_dates for row in rows_by_date[date]]
        validation_raw = [row for date in validation_dates for row in rows_by_date[date]]
        test_rows = [row for date in test_dates for row in rows_by_date[date]]

        train_rows = train_raw
        validation_rows = validation_raw
        if purge_label_overlap:
            validation_start_time = _event_time_ms(validation_raw[0])
            test_start_time = _event_time_ms(test_rows[0])
            train_rows = _purge_rows_crossing_boundary(train_raw, validation_start_time)
            validation_rows = _purge_rows_crossing_boundary(validation_raw, test_start_time)

        if not train_rows or not validation_rows or not test_rows:
            start += effective_step
            fold_idx += 1
            continue

        result = _select_threshold_result(
            train_rows=train_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
            feature_names=feature_names,
            threshold_values=threshold_values,
            taker_fee_bps=taker_fee_bps,
            maker_fee_bps=maker_fee_bps,
            slippage_bps=slippage_bps,
            execution_model=execution_model,
            sort_by=sort_by,
        )
        folds.append(
            CalendarWalkForwardFoldResult(
                fold=fold_idx,
                train_dates=train_dates,
                validation_dates=validation_dates,
                test_dates=test_dates,
                train_rows=len(train_rows),
                validation_rows=len(validation_rows),
                test_rows=len(test_rows),
                purged_train_rows=len(train_raw) - len(train_rows),
                purged_validation_rows=len(validation_raw) - len(validation_rows),
                result=result,
            )
        )
        start += effective_step
        fold_idx += 1

    if not folds:
        raise ValueError("no valid calendar walk-forward folds after purging")
    return folds


def run_conditional_walk_forward_thresholds(
    feature_csv: Path | str,
    *,
    features: list[str] | None = None,
    thresholds: list[float] | None = None,
    regime_features: list[str] | None = None,
    regime_bins: int = 3,
    min_validation_trades: int = 1,
    train_size: int = 2400,
    validation_size: int = 1200,
    test_size: int = 1200,
    step_size: int | None = None,
    purge_label_overlap: bool = True,
    execution_model: str = "taker",
    maker_fee_bps: float = 0.0,
    taker_fee_bps: float = 5.0,
    slippage_bps: float = 0.0,
    sort_by: str = "validation_net_pnl",
) -> list[ConditionalWalkForwardFoldResult]:
    rows = _read_rows(Path(feature_csv))
    if not rows:
        raise ValueError("no rows available for conditional walk-forward evaluation")
    if train_size <= 0 or validation_size <= 0 or test_size <= 0:
        raise ValueError("train_size, validation_size, and test_size must be positive")
    if regime_bins <= 0:
        raise ValueError("regime_bins must be positive")
    if min_validation_trades < 0:
        raise ValueError("min_validation_trades cannot be negative")
    effective_step = step_size or test_size
    if effective_step <= 0:
        raise ValueError("step_size must be positive")

    feature_names = _available_features(rows, features)
    selected_regime_features = _available_regime_features(rows, regime_features)
    threshold_values = thresholds or DEFAULT_THRESHOLDS
    total_window = train_size + validation_size + test_size
    if len(rows) < total_window:
        raise ValueError("not enough rows for one conditional walk-forward fold")

    folds: list[ConditionalWalkForwardFoldResult] = []
    start = 0
    fold_idx = 1
    while start + total_window <= len(rows):
        train_raw = rows[start : start + train_size]
        validation_raw = rows[start + train_size : start + train_size + validation_size]
        test_rows = rows[start + train_size + validation_size : start + total_window]

        train_rows = train_raw
        validation_rows = validation_raw
        if purge_label_overlap:
            validation_start_time = _event_time_ms(validation_raw[0])
            test_start_time = _event_time_ms(test_rows[0])
            train_rows = _purge_rows_crossing_boundary(train_raw, validation_start_time)
            validation_rows = _purge_rows_crossing_boundary(validation_raw, test_start_time)

        if not train_rows or not validation_rows or not test_rows:
            start += effective_step
            fold_idx += 1
            continue

        result = _select_conditional_result(
            train_rows=train_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
            feature_names=feature_names,
            threshold_values=threshold_values,
            regime_features=selected_regime_features,
            regime_bins=regime_bins,
            min_validation_trades=min_validation_trades,
            taker_fee_bps=taker_fee_bps,
            maker_fee_bps=maker_fee_bps,
            slippage_bps=slippage_bps,
            execution_model=execution_model,
            sort_by=sort_by,
        )
        folds.append(
            ConditionalWalkForwardFoldResult(
                fold=fold_idx,
                train_rows=len(train_rows),
                validation_rows=len(validation_rows),
                test_rows=len(test_rows),
                purged_train_rows=len(train_raw) - len(train_rows),
                purged_validation_rows=len(validation_raw) - len(validation_rows),
                result=result,
            )
        )
        start += effective_step
        fold_idx += 1

    if not folds:
        raise ValueError("no valid conditional walk-forward folds after purging")

    return folds


def evaluate_rule_file(
    feature_csv: Path | str,
    *,
    feature: str,
    threshold: float,
    execution_model: str = "taker",
    maker_fee_bps: float = 0.0,
    taker_fee_bps: float = 5.0,
    slippage_bps: float = 0.0,
    source_date: str | None = None,
) -> RuleEvaluation:
    rows = _read_rows(Path(feature_csv))
    if source_date is not None:
        rows = [row for row in rows if row.get("source_date") == source_date]
    if not rows:
        raise ValueError("no rows available for rule evaluation")
    predictor = lambda row: predict_feature_threshold(row, feature, threshold)
    return RuleEvaluation(
        feature=feature,
        threshold=threshold,
        metrics=evaluate_predictor(rows, predictor),
        economics=evaluate_economics(
            rows,
            predictor,
            execution_model=execution_model,
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
        ),
    )


def run_regime_analysis(
    feature_csv: Path | str,
    *,
    feature: str,
    threshold: float,
    regime_features: list[str] | None = None,
    bins: int = 3,
    execution_model: str = "taker",
    maker_fee_bps: float = 0.0,
    taker_fee_bps: float = 5.0,
    slippage_bps: float = 0.0,
    source_date: str | None = None,
) -> list[RegimeEvaluation]:
    rows = _read_rows(Path(feature_csv))
    if source_date is not None:
        rows = [row for row in rows if row.get("source_date") == source_date]
    if not rows:
        raise ValueError("no rows available for regime analysis")
    if feature not in rows[0]:
        raise ValueError(f"feature column not found: {feature}")
    if bins <= 0:
        raise ValueError("bins must be positive")

    selected_regime_features = regime_features or [
        name for name in DEFAULT_REGIME_FEATURES if name in rows[0]
    ]
    if not selected_regime_features:
        raise ValueError("no default regime feature columns found")
    missing = [name for name in selected_regime_features if name not in rows[0]]
    if missing:
        raise ValueError(f"regime feature columns not found: {', '.join(missing)}")

    predictor = lambda row: predict_feature_threshold(row, feature, threshold)
    evaluations: list[RegimeEvaluation] = []
    for regime_feature in selected_regime_features:
        bucketed = _quantile_buckets(rows, regime_feature, bins)
        for bucket_idx, bucket_rows in enumerate(bucketed, start=1):
            if not bucket_rows:
                continue
            values = [_safe_float(row[regime_feature]) for row in bucket_rows]
            finite_values = [value for value in values if value is not None]
            if not finite_values:
                continue
            labels = [int(row["label"]) for row in bucket_rows]
            evaluations.append(
                RegimeEvaluation(
                    regime_feature=regime_feature,
                    bucket=bucket_idx,
                    lower_bound=min(finite_values),
                    upper_bound=max(finite_values),
                    row_count=len(bucket_rows),
                    label_down=sum(1 for label in labels if label == -1),
                    label_flat=sum(1 for label in labels if label == 0),
                    label_up=sum(1 for label in labels if label == 1),
                    maker_long_fillable_rate=_truthy_rate(bucket_rows, "maker_long_fillable"),
                    maker_short_fillable_rate=_truthy_rate(bucket_rows, "maker_short_fillable"),
                    metrics=evaluate_predictor(bucket_rows, predictor),
                    economics=evaluate_economics(
                        bucket_rows,
                        predictor,
                        execution_model=execution_model,
                        maker_fee_bps=maker_fee_bps,
                        taker_fee_bps=taker_fee_bps,
                        slippage_bps=slippage_bps,
                    ),
                )
            )
    return evaluations


def run_fee_sweep(
    feature_csv: Path | str,
    *,
    fees_bps: list[float],
    features: list[str] | None = None,
    thresholds: list[float] | None = None,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    execution_model: str = "taker",
    maker_fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    sort_by: str = "validation_net_pnl",
) -> list[FeeSweepResult]:
    sweep: list[FeeSweepResult] = []
    for fee_bps in fees_bps:
        results = run_threshold_baselines(
            feature_csv,
            features=features,
            thresholds=thresholds,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            execution_model=execution_model,
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            sort_by=sort_by,
        )
        sweep.append(FeeSweepResult(fee_bps=fee_bps, result=results[0]))
    return sweep


def _make_result(
    *,
    name: str,
    feature: str,
    threshold: float,
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    taker_fee_bps: float,
    maker_fee_bps: float,
    slippage_bps: float,
    execution_model: str,
    predictor: Callable[[dict[str, str]], int],
) -> BaselineResult:
    return BaselineResult(
        name=name,
        feature=feature,
        threshold=threshold,
        train=evaluate_predictor(train_rows, predictor),
        validation=evaluate_predictor(validation_rows, predictor),
        test=evaluate_predictor(test_rows, predictor),
        train_economics=evaluate_economics(
            train_rows,
            predictor,
            execution_model=execution_model,
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
        ),
        validation_economics=evaluate_economics(
            validation_rows,
            predictor,
            execution_model=execution_model,
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
        ),
        test_economics=evaluate_economics(
            test_rows,
            predictor,
            execution_model=execution_model,
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
        ),
    )


def _select_threshold_result(
    *,
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    feature_names: list[str],
    threshold_values: list[float],
    taker_fee_bps: float,
    maker_fee_bps: float,
    slippage_bps: float,
    execution_model: str,
    sort_by: str,
) -> BaselineResult:
    results = [
        _make_result(
            name="always_flat",
            feature="constant",
            threshold=0.0,
            train_rows=train_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
            taker_fee_bps=taker_fee_bps,
            maker_fee_bps=maker_fee_bps,
            slippage_bps=slippage_bps,
            execution_model=execution_model,
            predictor=lambda row: 0,
        )
    ]
    for feature in feature_names:
        for threshold in threshold_values:
            results.append(
                _make_result(
                    name=f"{feature}_threshold",
                    feature=feature,
                    threshold=threshold,
                    train_rows=train_rows,
                    validation_rows=validation_rows,
                    test_rows=test_rows,
                    taker_fee_bps=taker_fee_bps,
                    maker_fee_bps=maker_fee_bps,
                    slippage_bps=slippage_bps,
                    execution_model=execution_model,
                    predictor=lambda row, feature=feature, threshold=threshold: predict_feature_threshold(
                        row,
                        feature,
                        threshold,
                    ),
                )
            )
    return sorted(results, key=lambda result: _sort_key(result, sort_by), reverse=True)[0]


def _select_conditional_result(
    *,
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    feature_names: list[str],
    threshold_values: list[float],
    regime_features: list[str],
    regime_bins: int,
    min_validation_trades: int,
    taker_fee_bps: float,
    maker_fee_bps: float,
    slippage_bps: float,
    execution_model: str,
    sort_by: str,
) -> ConditionalResult:
    results = [
        _make_conditional_result(
            name="always_flat",
            feature="constant",
            threshold=0.0,
            regime_spec=None,
            train_rows=train_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
            taker_fee_bps=taker_fee_bps,
            maker_fee_bps=maker_fee_bps,
            slippage_bps=slippage_bps,
            execution_model=execution_model,
            predictor=lambda row: 0,
        )
    ]
    regime_specs = [
        spec
        for regime_feature in regime_features
        for spec in _quantile_bucket_specs(validation_rows, regime_feature, regime_bins)
    ]

    for feature in feature_names:
        for threshold in threshold_values:
            results.append(
                _make_conditional_result(
                    name=f"{feature}_threshold",
                    feature=feature,
                    threshold=threshold,
                    regime_spec=None,
                    train_rows=train_rows,
                    validation_rows=validation_rows,
                    test_rows=test_rows,
                    taker_fee_bps=taker_fee_bps,
                    maker_fee_bps=maker_fee_bps,
                    slippage_bps=slippage_bps,
                    execution_model=execution_model,
                    predictor=lambda row, feature=feature, threshold=threshold: predict_feature_threshold(
                        row,
                        feature,
                        threshold,
                    ),
                )
            )
            for spec in regime_specs:
                results.append(
                    _make_conditional_result(
                        name=f"{feature}_threshold_if_{spec.feature}_bucket_{spec.bucket}",
                        feature=feature,
                        threshold=threshold,
                        regime_spec=spec,
                        train_rows=train_rows,
                        validation_rows=validation_rows,
                        test_rows=test_rows,
                        taker_fee_bps=taker_fee_bps,
                        maker_fee_bps=maker_fee_bps,
                        slippage_bps=slippage_bps,
                        execution_model=execution_model,
                        predictor=lambda row, feature=feature, threshold=threshold, spec=spec: (
                            predict_feature_threshold(row, feature, threshold)
                            if _row_in_regime_bucket(row, spec)
                            else 0
                        ),
                    )
                )

    eligible = [
        result
        for result in results
        if result.name == "always_flat" or result.validation_economics.trades >= min_validation_trades
    ]
    return sorted(eligible, key=lambda result: _sort_key(result, sort_by), reverse=True)[0]


def _make_conditional_result(
    *,
    name: str,
    feature: str,
    threshold: float,
    regime_spec: RegimeBucketSpec | None,
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    taker_fee_bps: float,
    maker_fee_bps: float,
    slippage_bps: float,
    execution_model: str,
    predictor: Callable[[dict[str, str]], int],
) -> ConditionalResult:
    return ConditionalResult(
        name=name,
        feature=feature,
        threshold=threshold,
        regime_feature=regime_spec.feature if regime_spec else "",
        regime_bucket=regime_spec.bucket if regime_spec else 0,
        regime_lower_bound=regime_spec.lower_bound if regime_spec else 0.0,
        regime_upper_bound=regime_spec.upper_bound if regime_spec else 0.0,
        train=evaluate_predictor(train_rows, predictor),
        validation=evaluate_predictor(validation_rows, predictor),
        test=evaluate_predictor(test_rows, predictor),
        train_economics=evaluate_economics(
            train_rows,
            predictor,
            execution_model=execution_model,
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
        ),
        validation_economics=evaluate_economics(
            validation_rows,
            predictor,
            execution_model=execution_model,
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
        ),
        test_economics=evaluate_economics(
            test_rows,
            predictor,
            execution_model=execution_model,
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
        ),
    )


def _available_features(rows: list[dict[str, str]], features: list[str] | None) -> list[str]:
    row_features = set(rows[0])
    if features is not None:
        missing = [feature for feature in features if feature not in row_features]
        if missing:
            raise ValueError(f"feature columns not found: {', '.join(missing)}")
        return features

    available = [feature for feature in DEFAULT_FEATURES if feature in row_features]
    if not available:
        raise ValueError("no default feature columns found")
    return available


def _available_regime_features(rows: list[dict[str, str]], regime_features: list[str] | None) -> list[str]:
    row_features = set(rows[0])
    if regime_features is not None:
        missing = [feature for feature in regime_features if feature not in row_features]
        if missing:
            raise ValueError(f"regime feature columns not found: {', '.join(missing)}")
        return regime_features

    available = [feature for feature in DEFAULT_REGIME_FEATURES if feature in row_features]
    if not available:
        raise ValueError("no default regime feature columns found")
    return available


def _sort_key(result: BaselineResult, sort_by: str) -> float:
    if sort_by == "validation_macro_f1":
        return result.validation.macro_f1
    if sort_by == "validation_balanced_accuracy":
        return result.validation.balanced_accuracy
    if sort_by == "validation_net_pnl":
        return result.validation_economics.net_pnl
    if sort_by == "validation_gross_pnl":
        return result.validation_economics.gross_pnl
    if sort_by == "test_net_pnl":
        return result.test_economics.net_pnl
    if sort_by == "test_gross_pnl":
        return result.test_economics.gross_pnl
    raise ValueError(
        "sort_by must be one of: validation_macro_f1, validation_balanced_accuracy, "
        "validation_net_pnl, validation_gross_pnl, test_net_pnl, test_gross_pnl"
    )


def split_time_ordered(
    rows: list[dict[str, str]],
    *,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be < 1")

    train_end = int(len(rows) * train_fraction)
    validation_end = int(len(rows) * (train_fraction + validation_fraction))
    return rows[:train_end], rows[train_end:validation_end], rows[validation_end:]


def _purge_rows_crossing_boundary(
    rows: list[dict[str, str]],
    next_window_start_time_ms: int,
) -> list[dict[str, str]]:
    return [row for row in rows if _future_time_ms(row) < next_window_start_time_ms]


def _event_time_ms(row: dict[str, str]) -> int:
    return int(float(row["event_time"]))


def _future_time_ms(row: dict[str, str]) -> int:
    return int(float(row.get("future_event_time") or row["event_time"]))


def evaluate_feature_threshold(
    rows: list[dict[str, str]],
    feature: str,
    threshold: float,
) -> Metrics:
    return evaluate_predictor(
        rows,
        lambda row: predict_feature_threshold(row, feature, threshold),
    )


def predict_feature_threshold(row: dict[str, str], feature: str, threshold: float) -> int:
    value = float(row[feature])
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def evaluate_predictor(rows: list[dict[str, str]], predictor: Callable[[dict[str, str]], int]) -> Metrics:
    y_true = [int(row["label"]) for row in rows]
    y_pred = [predictor(row) for row in rows]
    return compute_metrics(y_true, y_pred)


def evaluate_constant(rows: list[dict[str, str]], label: int) -> Metrics:
    return evaluate_predictor(rows, lambda row: label)


def evaluate_economics(
    rows: list[dict[str, str]],
    predictor: Callable[[dict[str, str]], int],
    *,
    execution_model: str,
    maker_fee_bps: float,
    taker_fee_bps: float,
    slippage_bps: float,
) -> EconomicMetrics:
    if execution_model == "taker":
        return evaluate_taker_economics(
            rows,
            predictor,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
        )
    if execution_model == "maker_entry":
        return evaluate_maker_entry_economics(
            rows,
            predictor,
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
        )
    raise ValueError("execution_model must be one of: taker, maker_entry")


def evaluate_taker_economics(
    rows: list[dict[str, str]],
    predictor: Callable[[dict[str, str]], int],
    *,
    taker_fee_bps: float,
    slippage_bps: float,
) -> EconomicMetrics:
    fee_rate = taker_fee_bps / 10_000.0
    slippage_rate = slippage_bps / 10_000.0
    signals = 0
    trades = 0
    gross_pnl = 0.0
    net_pnl = 0.0
    net_return_bps = 0.0
    trade_net_pnls: list[float] = []
    trade_net_returns_bps: list[float] = []
    fee_turnover = 0.0
    wins = 0

    for row in rows:
        side = predictor(row)
        if side == 0:
            continue
        signals += 1

        bid = float(row.get("entry_bid") or row["bid"])
        ask = float(row.get("entry_ask") or row["ask"])
        future_bid = float(row["future_bid"])
        future_ask = float(row["future_ask"])

        if side > 0:
            entry = ask
            exit_price = future_bid
            trade_gross = exit_price - entry
        else:
            entry = bid
            exit_price = future_ask
            trade_gross = entry - exit_price

        fee_cost = fee_rate * (entry + exit_price)
        slippage_cost = slippage_rate * (entry + exit_price)
        trade_net = trade_gross - fee_cost - slippage_cost

        trades += 1
        gross_pnl += trade_gross
        net_pnl += trade_net
        trade_net_pnls.append(trade_net)
        trade_return_bps = 10_000.0 * trade_net / entry if entry else 0.0
        trade_net_returns_bps.append(trade_return_bps)
        fee_turnover += entry + exit_price
        net_return_bps += trade_return_bps
        if trade_net > 0:
            wins += 1

    break_even_taker_fee_bps = (
        (gross_pnl / fee_turnover * 10_000.0) - slippage_bps if fee_turnover else 0.0
    )
    return EconomicMetrics(
        signals=signals,
        trades=trades,
        fill_rate=trades / signals if signals else 0.0,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        fee_turnover=fee_turnover,
        break_even_taker_fee_bps=break_even_taker_fee_bps,
        mean_net_pnl_per_trade=net_pnl / trades if trades else 0.0,
        mean_net_return_bps_per_trade=net_return_bps / trades if trades else 0.0,
        median_net_pnl_per_trade=_median(trade_net_pnls),
        median_net_return_bps_per_trade=_median(trade_net_returns_bps),
        profit_factor=_profit_factor(trade_net_pnls),
        max_drawdown_pnl=_max_drawdown(trade_net_pnls),
        sharpe_per_trade=_sharpe(trade_net_returns_bps),
        win_rate=wins / trades if trades else 0.0,
    )


def evaluate_maker_entry_economics(
    rows: list[dict[str, str]],
    predictor: Callable[[dict[str, str]], int],
    *,
    maker_fee_bps: float,
    taker_fee_bps: float,
    slippage_bps: float,
) -> EconomicMetrics:
    maker_fee_rate = maker_fee_bps / 10_000.0
    taker_fee_rate = taker_fee_bps / 10_000.0
    slippage_rate = slippage_bps / 10_000.0
    signals = 0
    trades = 0
    gross_pnl = 0.0
    net_pnl = 0.0
    net_return_bps = 0.0
    trade_net_pnls: list[float] = []
    trade_net_returns_bps: list[float] = []
    exit_fee_turnover = 0.0
    maker_fee_turnover = 0.0
    wins = 0

    for row in rows:
        side = predictor(row)
        if side == 0:
            continue
        signals += 1

        if side > 0:
            if not _row_truthy(row.get("maker_long_fillable")):
                continue
            entry = float(row.get("entry_bid") or row["bid"])
            exit_price = float(row["future_bid"])
            trade_gross = exit_price - entry
        else:
            if not _row_truthy(row.get("maker_short_fillable")):
                continue
            entry = float(row.get("entry_ask") or row["ask"])
            exit_price = float(row["future_ask"])
            trade_gross = entry - exit_price

        maker_fee_cost = maker_fee_rate * entry
        exit_fee_cost = taker_fee_rate * exit_price
        slippage_cost = slippage_rate * exit_price
        trade_net = trade_gross - maker_fee_cost - exit_fee_cost - slippage_cost

        trades += 1
        gross_pnl += trade_gross
        net_pnl += trade_net
        trade_net_pnls.append(trade_net)
        trade_return_bps = 10_000.0 * trade_net / entry if entry else 0.0
        trade_net_returns_bps.append(trade_return_bps)
        maker_fee_turnover += entry
        exit_fee_turnover += exit_price
        net_return_bps += trade_return_bps
        if trade_net > 0:
            wins += 1

    break_even_taker_fee_bps = (
        ((gross_pnl - maker_fee_rate * maker_fee_turnover - slippage_rate * exit_fee_turnover)
         / exit_fee_turnover
         * 10_000.0)
        if exit_fee_turnover
        else 0.0
    )
    return EconomicMetrics(
        signals=signals,
        trades=trades,
        fill_rate=trades / signals if signals else 0.0,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        fee_turnover=exit_fee_turnover,
        break_even_taker_fee_bps=break_even_taker_fee_bps,
        mean_net_pnl_per_trade=net_pnl / trades if trades else 0.0,
        mean_net_return_bps_per_trade=net_return_bps / trades if trades else 0.0,
        median_net_pnl_per_trade=_median(trade_net_pnls),
        median_net_return_bps_per_trade=_median(trade_net_returns_bps),
        profit_factor=_profit_factor(trade_net_pnls),
        max_drawdown_pnl=_max_drawdown(trade_net_pnls),
        sharpe_per_trade=_sharpe(trade_net_returns_bps),
        win_rate=wins / trades if trades else 0.0,
    )


def compute_metrics(y_true: list[int], y_pred: list[int]) -> Metrics:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred length mismatch")
    if not y_true:
        raise ValueError("cannot compute metrics on empty input")

    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    recalls: list[float] = []
    f1s: list[float] = []
    for klass in CLASSES:
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == klass and pred == klass)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != klass and pred == klass)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == klass and pred != klass)
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)

    non_flat = sum(1 for pred in y_pred if pred != 0)
    return Metrics(
        n=len(y_true),
        accuracy=correct / len(y_true),
        balanced_accuracy=sum(recalls) / len(recalls),
        macro_f1=sum(f1s) / len(f1s),
        coverage=non_flat / len(y_pred),
    )


def format_baseline_results(results: list[BaselineResult], *, top: int = 10) -> str:
    lines = [
        "rank,name,feature,threshold,val_macro_f1,val_bal_acc,val_accuracy,val_signals,val_trades,val_fill_rate,val_gross_pnl,val_net_pnl,val_break_even_fee_bps,test_macro_f1,test_bal_acc,test_accuracy,test_coverage,test_signals,test_trades,test_fill_rate,test_gross_pnl,test_net_pnl,test_break_even_fee_bps,test_mean_net_pnl,test_mean_net_bps,test_median_net_pnl,test_median_net_bps,test_profit_factor,test_max_drawdown_pnl,test_sharpe_per_trade,test_win_rate"
    ]
    for rank, result in enumerate(results[:top], start=1):
        validation_economics = result.validation_economics
        test_economics = result.test_economics
        lines.append(
            ",".join(
                [
                    str(rank),
                    result.name,
                    result.feature,
                    f"{result.threshold:.6g}",
                    _fmt(result.validation.macro_f1),
                    _fmt(result.validation.balanced_accuracy),
                    _fmt(result.validation.accuracy),
                    str(validation_economics.signals),
                    str(validation_economics.trades),
                    _fmt(validation_economics.fill_rate),
                    _fmt(validation_economics.gross_pnl),
                    _fmt(validation_economics.net_pnl),
                    _fmt(validation_economics.break_even_taker_fee_bps),
                    _fmt(result.test.macro_f1),
                    _fmt(result.test.balanced_accuracy),
                    _fmt(result.test.accuracy),
                    _fmt(result.test.coverage),
                    str(test_economics.signals),
                    str(test_economics.trades),
                    _fmt(test_economics.fill_rate),
                    _fmt(test_economics.gross_pnl),
                    _fmt(test_economics.net_pnl),
                    _fmt(test_economics.break_even_taker_fee_bps),
                    _fmt(test_economics.mean_net_pnl_per_trade),
                    _fmt(test_economics.mean_net_return_bps_per_trade),
                    _fmt(test_economics.median_net_pnl_per_trade),
                    _fmt(test_economics.median_net_return_bps_per_trade),
                    _fmt(test_economics.profit_factor),
                    _fmt(test_economics.max_drawdown_pnl),
                    _fmt(test_economics.sharpe_per_trade),
                    _fmt(test_economics.win_rate),
                ]
            )
        )
    return "\n".join(lines)


def format_rule_evaluation(evaluation: RuleEvaluation) -> str:
    metrics = evaluation.metrics
    economics = evaluation.economics
    lines = [
        "feature,threshold,n,macro_f1,balanced_accuracy,accuracy,coverage,signals,trades,fill_rate,gross_pnl,net_pnl,break_even_fee_bps,mean_net_pnl,mean_net_bps,median_net_pnl,median_net_bps,profit_factor,max_drawdown_pnl,sharpe_per_trade,win_rate",
        ",".join(
            [
                evaluation.feature,
                f"{evaluation.threshold:.6g}",
                str(metrics.n),
                _fmt(metrics.macro_f1),
                _fmt(metrics.balanced_accuracy),
                _fmt(metrics.accuracy),
                _fmt(metrics.coverage),
                str(economics.signals),
                str(economics.trades),
                _fmt(economics.fill_rate),
                _fmt(economics.gross_pnl),
                _fmt(economics.net_pnl),
                _fmt(economics.break_even_taker_fee_bps),
                _fmt(economics.mean_net_pnl_per_trade),
                _fmt(economics.mean_net_return_bps_per_trade),
                _fmt(economics.median_net_pnl_per_trade),
                _fmt(economics.median_net_return_bps_per_trade),
                _fmt(economics.profit_factor),
                _fmt(economics.max_drawdown_pnl),
                _fmt(economics.sharpe_per_trade),
                _fmt(economics.win_rate),
            ]
        ),
    ]
    return "\n".join(lines)


def format_fee_sweep(sweep: list[FeeSweepResult]) -> str:
    lines = [
        "fee_bps,name,feature,threshold,val_signals,val_trades,val_fill_rate,val_gross_pnl,val_net_pnl,val_break_even_fee_bps,test_signals,test_trades,test_fill_rate,test_gross_pnl,test_net_pnl,test_break_even_fee_bps,test_mean_net_bps,test_median_net_bps,test_profit_factor,test_max_drawdown_pnl,test_sharpe_per_trade,test_win_rate"
    ]
    for item in sweep:
        result = item.result
        validation_economics = result.validation_economics
        test_economics = result.test_economics
        lines.append(
            ",".join(
                [
                    _fmt(item.fee_bps),
                    result.name,
                    result.feature,
                    f"{result.threshold:.6g}",
                    str(validation_economics.signals),
                    str(validation_economics.trades),
                    _fmt(validation_economics.fill_rate),
                    _fmt(validation_economics.gross_pnl),
                    _fmt(validation_economics.net_pnl),
                    _fmt(validation_economics.break_even_taker_fee_bps),
                    str(test_economics.signals),
                    str(test_economics.trades),
                    _fmt(test_economics.fill_rate),
                    _fmt(test_economics.gross_pnl),
                    _fmt(test_economics.net_pnl),
                    _fmt(test_economics.break_even_taker_fee_bps),
                    _fmt(test_economics.mean_net_return_bps_per_trade),
                    _fmt(test_economics.median_net_return_bps_per_trade),
                    _fmt(test_economics.profit_factor),
                    _fmt(test_economics.max_drawdown_pnl),
                    _fmt(test_economics.sharpe_per_trade),
                    _fmt(test_economics.win_rate),
                ]
            )
        )
    return "\n".join(lines)


def format_walk_forward_results(folds: list[WalkForwardFoldResult]) -> str:
    lines = [
        "fold,train_rows,validation_rows,test_rows,purged_train_rows,purged_validation_rows,name,feature,threshold,val_macro_f1,val_signals,val_trades,val_fill_rate,val_net_pnl,val_break_even_fee_bps,test_macro_f1,test_bal_acc,test_accuracy,test_coverage,test_signals,test_trades,test_fill_rate,test_gross_pnl,test_net_pnl,test_break_even_fee_bps,test_mean_net_bps,test_median_net_bps,test_profit_factor,test_max_drawdown_pnl,test_sharpe_per_trade,test_win_rate"
    ]
    total_test_signals = 0
    total_test_trades = 0
    total_test_gross_pnl = 0.0
    total_test_net_pnl = 0.0
    total_test_fee_turnover = 0.0
    total_test_rows = 0
    weighted_test_macro_f1 = 0.0
    weighted_test_bal_acc = 0.0
    weighted_test_accuracy = 0.0
    weighted_test_coverage = 0.0

    for fold in folds:
        result = fold.result
        validation_economics = result.validation_economics
        test_economics = result.test_economics
        lines.append(
            ",".join(
                [
                    str(fold.fold),
                    str(fold.train_rows),
                    str(fold.validation_rows),
                    str(fold.test_rows),
                    str(fold.purged_train_rows),
                    str(fold.purged_validation_rows),
                    result.name,
                    result.feature,
                    f"{result.threshold:.6g}",
                    _fmt(result.validation.macro_f1),
                    str(validation_economics.signals),
                    str(validation_economics.trades),
                    _fmt(validation_economics.fill_rate),
                    _fmt(validation_economics.net_pnl),
                    _fmt(validation_economics.break_even_taker_fee_bps),
                    _fmt(result.test.macro_f1),
                    _fmt(result.test.balanced_accuracy),
                    _fmt(result.test.accuracy),
                    _fmt(result.test.coverage),
                    str(test_economics.signals),
                    str(test_economics.trades),
                    _fmt(test_economics.fill_rate),
                    _fmt(test_economics.gross_pnl),
                    _fmt(test_economics.net_pnl),
                    _fmt(test_economics.break_even_taker_fee_bps),
                    _fmt(test_economics.mean_net_return_bps_per_trade),
                    _fmt(test_economics.median_net_return_bps_per_trade),
                    _fmt(test_economics.profit_factor),
                    _fmt(test_economics.max_drawdown_pnl),
                    _fmt(test_economics.sharpe_per_trade),
                    _fmt(test_economics.win_rate),
                ]
            )
        )
        total_test_rows += fold.test_rows
        total_test_signals += test_economics.signals
        total_test_trades += test_economics.trades
        total_test_gross_pnl += test_economics.gross_pnl
        total_test_net_pnl += test_economics.net_pnl
        total_test_fee_turnover += test_economics.fee_turnover
        weighted_test_macro_f1 += result.test.macro_f1 * fold.test_rows
        weighted_test_bal_acc += result.test.balanced_accuracy * fold.test_rows
        weighted_test_accuracy += result.test.accuracy * fold.test_rows
        weighted_test_coverage += result.test.coverage * fold.test_rows

    summary_break_even_fee_bps = 0.0
    if total_test_fee_turnover:
        summary_break_even_fee_bps = sum(
            fold.result.test_economics.break_even_taker_fee_bps
            * fold.result.test_economics.fee_turnover
            for fold in folds
        ) / total_test_fee_turnover
    lines.append(
        ",".join(
            [
                "summary",
                "",
                "",
                str(total_test_rows),
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                _fmt(weighted_test_macro_f1 / total_test_rows if total_test_rows else 0.0),
                _fmt(weighted_test_bal_acc / total_test_rows if total_test_rows else 0.0),
                _fmt(weighted_test_accuracy / total_test_rows if total_test_rows else 0.0),
                _fmt(weighted_test_coverage / total_test_rows if total_test_rows else 0.0),
                str(total_test_signals),
                str(total_test_trades),
                _fmt(total_test_trades / total_test_signals if total_test_signals else 0.0),
                _fmt(total_test_gross_pnl),
                _fmt(total_test_net_pnl),
                _fmt(summary_break_even_fee_bps),
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )
    )
    return "\n".join(lines)


def format_calendar_walk_forward_results(folds: list[CalendarWalkForwardFoldResult]) -> str:
    lines = [
        "fold,train_dates,validation_dates,test_dates,train_rows,validation_rows,test_rows,purged_train_rows,purged_validation_rows,name,feature,threshold,val_macro_f1,val_signals,val_trades,val_fill_rate,val_net_pnl,val_break_even_fee_bps,test_macro_f1,test_bal_acc,test_accuracy,test_coverage,test_signals,test_trades,test_fill_rate,test_gross_pnl,test_net_pnl,test_break_even_fee_bps,test_mean_net_bps,test_median_net_bps,test_profit_factor,test_max_drawdown_pnl,test_sharpe_per_trade,test_win_rate"
    ]
    total_test_signals = 0
    total_test_trades = 0
    total_test_gross_pnl = 0.0
    total_test_net_pnl = 0.0
    total_test_fee_turnover = 0.0
    total_test_rows = 0
    weighted_test_macro_f1 = 0.0
    weighted_test_bal_acc = 0.0
    weighted_test_accuracy = 0.0
    weighted_test_coverage = 0.0

    for fold in folds:
        result = fold.result
        validation_economics = result.validation_economics
        test_economics = result.test_economics
        lines.append(
            ",".join(
                [
                    str(fold.fold),
                    "|".join(fold.train_dates),
                    "|".join(fold.validation_dates),
                    "|".join(fold.test_dates),
                    str(fold.train_rows),
                    str(fold.validation_rows),
                    str(fold.test_rows),
                    str(fold.purged_train_rows),
                    str(fold.purged_validation_rows),
                    result.name,
                    result.feature,
                    f"{result.threshold:.6g}",
                    _fmt(result.validation.macro_f1),
                    str(validation_economics.signals),
                    str(validation_economics.trades),
                    _fmt(validation_economics.fill_rate),
                    _fmt(validation_economics.net_pnl),
                    _fmt(validation_economics.break_even_taker_fee_bps),
                    _fmt(result.test.macro_f1),
                    _fmt(result.test.balanced_accuracy),
                    _fmt(result.test.accuracy),
                    _fmt(result.test.coverage),
                    str(test_economics.signals),
                    str(test_economics.trades),
                    _fmt(test_economics.fill_rate),
                    _fmt(test_economics.gross_pnl),
                    _fmt(test_economics.net_pnl),
                    _fmt(test_economics.break_even_taker_fee_bps),
                    _fmt(test_economics.mean_net_return_bps_per_trade),
                    _fmt(test_economics.median_net_return_bps_per_trade),
                    _fmt(test_economics.profit_factor),
                    _fmt(test_economics.max_drawdown_pnl),
                    _fmt(test_economics.sharpe_per_trade),
                    _fmt(test_economics.win_rate),
                ]
            )
        )
        total_test_rows += fold.test_rows
        total_test_signals += test_economics.signals
        total_test_trades += test_economics.trades
        total_test_gross_pnl += test_economics.gross_pnl
        total_test_net_pnl += test_economics.net_pnl
        total_test_fee_turnover += test_economics.fee_turnover
        weighted_test_macro_f1 += result.test.macro_f1 * fold.test_rows
        weighted_test_bal_acc += result.test.balanced_accuracy * fold.test_rows
        weighted_test_accuracy += result.test.accuracy * fold.test_rows
        weighted_test_coverage += result.test.coverage * fold.test_rows

    summary_break_even_fee_bps = 0.0
    if total_test_fee_turnover:
        summary_break_even_fee_bps = sum(
            fold.result.test_economics.break_even_taker_fee_bps
            * fold.result.test_economics.fee_turnover
            for fold in folds
        ) / total_test_fee_turnover
    lines.append(
        ",".join(
            [
                "summary",
                "",
                "",
                "",
                "",
                "",
                str(total_test_rows),
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                _fmt(weighted_test_macro_f1 / total_test_rows if total_test_rows else 0.0),
                _fmt(weighted_test_bal_acc / total_test_rows if total_test_rows else 0.0),
                _fmt(weighted_test_accuracy / total_test_rows if total_test_rows else 0.0),
                _fmt(weighted_test_coverage / total_test_rows if total_test_rows else 0.0),
                str(total_test_signals),
                str(total_test_trades),
                _fmt(total_test_trades / total_test_signals if total_test_signals else 0.0),
                _fmt(total_test_gross_pnl),
                _fmt(total_test_net_pnl),
                _fmt(summary_break_even_fee_bps),
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )
    )
    return "\n".join(lines)


def format_conditional_walk_forward_results(folds: list[ConditionalWalkForwardFoldResult]) -> str:
    lines = [
        "fold,train_rows,validation_rows,test_rows,purged_train_rows,purged_validation_rows,name,feature,threshold,regime_feature,regime_bucket,regime_lower_bound,regime_upper_bound,val_macro_f1,val_signals,val_trades,val_fill_rate,val_net_pnl,val_break_even_fee_bps,test_macro_f1,test_bal_acc,test_accuracy,test_coverage,test_signals,test_trades,test_fill_rate,test_gross_pnl,test_net_pnl,test_break_even_fee_bps,test_mean_net_bps,test_median_net_bps,test_profit_factor,test_max_drawdown_pnl,test_sharpe_per_trade,test_win_rate"
    ]
    total_test_signals = 0
    total_test_trades = 0
    total_test_gross_pnl = 0.0
    total_test_net_pnl = 0.0
    total_test_fee_turnover = 0.0
    total_test_rows = 0
    weighted_test_macro_f1 = 0.0
    weighted_test_bal_acc = 0.0
    weighted_test_accuracy = 0.0
    weighted_test_coverage = 0.0

    for fold in folds:
        result = fold.result
        validation_economics = result.validation_economics
        test_economics = result.test_economics
        lines.append(
            ",".join(
                [
                    str(fold.fold),
                    str(fold.train_rows),
                    str(fold.validation_rows),
                    str(fold.test_rows),
                    str(fold.purged_train_rows),
                    str(fold.purged_validation_rows),
                    result.name,
                    result.feature,
                    f"{result.threshold:.6g}",
                    result.regime_feature,
                    str(result.regime_bucket),
                    _fmt(result.regime_lower_bound),
                    _fmt(result.regime_upper_bound),
                    _fmt(result.validation.macro_f1),
                    str(validation_economics.signals),
                    str(validation_economics.trades),
                    _fmt(validation_economics.fill_rate),
                    _fmt(validation_economics.net_pnl),
                    _fmt(validation_economics.break_even_taker_fee_bps),
                    _fmt(result.test.macro_f1),
                    _fmt(result.test.balanced_accuracy),
                    _fmt(result.test.accuracy),
                    _fmt(result.test.coverage),
                    str(test_economics.signals),
                    str(test_economics.trades),
                    _fmt(test_economics.fill_rate),
                    _fmt(test_economics.gross_pnl),
                    _fmt(test_economics.net_pnl),
                    _fmt(test_economics.break_even_taker_fee_bps),
                    _fmt(test_economics.mean_net_return_bps_per_trade),
                    _fmt(test_economics.median_net_return_bps_per_trade),
                    _fmt(test_economics.profit_factor),
                    _fmt(test_economics.max_drawdown_pnl),
                    _fmt(test_economics.sharpe_per_trade),
                    _fmt(test_economics.win_rate),
                ]
            )
        )
        total_test_rows += fold.test_rows
        total_test_signals += test_economics.signals
        total_test_trades += test_economics.trades
        total_test_gross_pnl += test_economics.gross_pnl
        total_test_net_pnl += test_economics.net_pnl
        total_test_fee_turnover += test_economics.fee_turnover
        weighted_test_macro_f1 += result.test.macro_f1 * fold.test_rows
        weighted_test_bal_acc += result.test.balanced_accuracy * fold.test_rows
        weighted_test_accuracy += result.test.accuracy * fold.test_rows
        weighted_test_coverage += result.test.coverage * fold.test_rows

    summary_break_even_fee_bps = 0.0
    if total_test_fee_turnover:
        summary_break_even_fee_bps = sum(
            fold.result.test_economics.break_even_taker_fee_bps
            * fold.result.test_economics.fee_turnover
            for fold in folds
        ) / total_test_fee_turnover
    lines.append(
        ",".join(
            [
                "summary",
                "",
                "",
                str(total_test_rows),
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                _fmt(weighted_test_macro_f1 / total_test_rows if total_test_rows else 0.0),
                _fmt(weighted_test_bal_acc / total_test_rows if total_test_rows else 0.0),
                _fmt(weighted_test_accuracy / total_test_rows if total_test_rows else 0.0),
                _fmt(weighted_test_coverage / total_test_rows if total_test_rows else 0.0),
                str(total_test_signals),
                str(total_test_trades),
                _fmt(total_test_trades / total_test_signals if total_test_signals else 0.0),
                _fmt(total_test_gross_pnl),
                _fmt(total_test_net_pnl),
                _fmt(summary_break_even_fee_bps),
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )
    )
    return "\n".join(lines)


def format_regime_analysis(evaluations: list[RegimeEvaluation]) -> str:
    lines = [
        "regime_feature,bucket,lower_bound,upper_bound,rows,label_down,label_flat,label_up,macro_f1,balanced_accuracy,accuracy,coverage,signals,trades,fill_rate,gross_pnl,net_pnl,break_even_fee_bps,mean_net_bps,median_net_bps,profit_factor,max_drawdown_pnl,sharpe_per_trade,win_rate,maker_long_fillable_rate,maker_short_fillable_rate"
    ]
    for evaluation in evaluations:
        metrics = evaluation.metrics
        economics = evaluation.economics
        lines.append(
            ",".join(
                [
                    evaluation.regime_feature,
                    str(evaluation.bucket),
                    _fmt(evaluation.lower_bound),
                    _fmt(evaluation.upper_bound),
                    str(evaluation.row_count),
                    str(evaluation.label_down),
                    str(evaluation.label_flat),
                    str(evaluation.label_up),
                    _fmt(metrics.macro_f1),
                    _fmt(metrics.balanced_accuracy),
                    _fmt(metrics.accuracy),
                    _fmt(metrics.coverage),
                    str(economics.signals),
                    str(economics.trades),
                    _fmt(economics.fill_rate),
                    _fmt(economics.gross_pnl),
                    _fmt(economics.net_pnl),
                    _fmt(economics.break_even_taker_fee_bps),
                    _fmt(economics.mean_net_return_bps_per_trade),
                    _fmt(economics.median_net_return_bps_per_trade),
                    _fmt(economics.profit_factor),
                    _fmt(economics.max_drawdown_pnl),
                    _fmt(economics.sharpe_per_trade),
                    _fmt(economics.win_rate),
                    _fmt(evaluation.maker_long_fillable_rate),
                    _fmt(evaluation.maker_short_fillable_rate),
                ]
            )
        )
    return "\n".join(lines)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _quantile_buckets(
    rows: list[dict[str, str]],
    feature: str,
    bins: int,
) -> list[list[dict[str, str]]]:
    finite_rows = [
        (value, row)
        for row in rows
        if (value := _safe_float(row.get(feature, ""))) is not None
    ]
    if not finite_rows:
        raise ValueError(f"no finite values available for regime feature: {feature}")
    finite_rows.sort(key=lambda item: item[0])
    bucket_count = min(bins, len(finite_rows))
    buckets: list[list[dict[str, str]]] = []
    for bucket_idx in range(bucket_count):
        start = bucket_idx * len(finite_rows) // bucket_count
        end = (bucket_idx + 1) * len(finite_rows) // bucket_count
        buckets.append([row for _, row in finite_rows[start:end]])
    return buckets


def _quantile_bucket_specs(
    rows: list[dict[str, str]],
    feature: str,
    bins: int,
) -> list[RegimeBucketSpec]:
    finite_values = sorted(
        value
        for row in rows
        if (value := _safe_float(row.get(feature, ""))) is not None
    )
    if not finite_values:
        raise ValueError(f"no finite values available for regime feature: {feature}")
    bucket_count = min(bins, len(finite_values))
    specs: list[RegimeBucketSpec] = []
    for bucket_idx in range(bucket_count):
        start = bucket_idx * len(finite_values) // bucket_count
        end = (bucket_idx + 1) * len(finite_values) // bucket_count
        bucket_values = finite_values[start:end]
        specs.append(
            RegimeBucketSpec(
                feature=feature,
                bucket=bucket_idx + 1,
                lower_bound=bucket_values[0],
                upper_bound=bucket_values[-1],
                is_last=bucket_idx == bucket_count - 1,
            )
        )
    return specs


def _row_in_regime_bucket(row: dict[str, str], spec: RegimeBucketSpec) -> bool:
    value = _safe_float(row.get(spec.feature))
    if value is None:
        return False
    if spec.lower_bound == spec.upper_bound:
        return value == spec.lower_bound
    if spec.is_last:
        return spec.lower_bound <= value <= spec.upper_bound
    return spec.lower_bound <= value < spec.upper_bound


def _safe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _truthy_rate(rows: list[dict[str, str]], column: str) -> float:
    values = [row.get(column) for row in rows if column in row]
    if not values:
        return 0.0
    return sum(1 for value in values if _row_truthy(value)) / len(values)


def _row_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes"}


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _profit_factor(net_pnls: list[float]) -> float:
    gross_profit = sum(value for value in net_pnls if value > 0.0)
    gross_loss = -sum(value for value in net_pnls if value < 0.0)
    if gross_loss > 0.0:
        return gross_profit / gross_loss
    return math.inf if gross_profit > 0.0 else 0.0


def _max_drawdown(net_pnls: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl in net_pnls:
        equity += pnl
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    return max_drawdown


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _sharpe(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    std_value = _sample_std(values)
    if std_value <= 1e-12:
        return 0.0
    return (sum(values) / len(values)) / std_value


def _fmt(value: float) -> str:
    return f"{value:.6f}"
