from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from lob_forge.baselines import (
    CLASSES,
    DEFAULT_FEATURES,
    EconomicMetrics,
    Metrics,
    compute_metrics,
    evaluate_economics,
)


DEFAULT_ALPHA_THRESHOLDS = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75]


@dataclass(frozen=True)
class Standardizer:
    means: list[float]
    scales: list[float]


@dataclass(frozen=True)
class SoftmaxModel:
    features: list[str]
    standardizer: Standardizer
    weights: list[list[float]]


@dataclass(frozen=True)
class LogisticResult:
    name: str
    features: list[str]
    alpha_threshold: float
    train: Metrics
    validation: Metrics
    test: Metrics
    train_economics: EconomicMetrics
    validation_economics: EconomicMetrics
    test_economics: EconomicMetrics


@dataclass(frozen=True)
class LogisticWalkForwardFold:
    fold: int
    train_rows: int
    validation_rows: int
    test_rows: int
    purged_train_rows: int
    purged_validation_rows: int
    result: LogisticResult


def run_logistic_walk_forward(
    feature_csv: Path | str,
    *,
    features: list[str] | None = None,
    alpha_thresholds: list[float] | None = None,
    train_size: int = 2400,
    validation_size: int = 1200,
    test_size: int = 1200,
    step_size: int | None = None,
    purge_label_overlap: bool = True,
    epochs: int = 120,
    learning_rate: float = 0.05,
    l2: float = 0.001,
    class_weighting: str = "balanced",
    execution_model: str = "taker",
    maker_fee_bps: float = 0.0,
    taker_fee_bps: float = 5.0,
    slippage_bps: float = 0.0,
    sort_by: str = "validation_net_pnl",
) -> list[LogisticWalkForwardFold]:
    rows = _read_rows(Path(feature_csv))
    if not rows:
        raise ValueError("no rows available for logistic walk-forward evaluation")
    if train_size <= 0 or validation_size <= 0 or test_size <= 0:
        raise ValueError("train_size, validation_size, and test_size must be positive")
    effective_step = step_size or test_size
    if effective_step <= 0:
        raise ValueError("step_size must be positive")

    feature_names = _available_features(rows, features)
    threshold_values = alpha_thresholds or DEFAULT_ALPHA_THRESHOLDS
    total_window = train_size + validation_size + test_size
    if len(rows) < total_window:
        raise ValueError("not enough rows for one logistic walk-forward fold")

    folds: list[LogisticWalkForwardFold] = []
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

        model = fit_softmax_model(
            train_rows,
            feature_names,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            class_weighting=class_weighting,
        )
        result = select_logistic_threshold(
            model=model,
            train_rows=train_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
            alpha_thresholds=threshold_values,
            execution_model=execution_model,
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
            sort_by=sort_by,
        )
        folds.append(
            LogisticWalkForwardFold(
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
        raise ValueError("no valid logistic walk-forward folds after purging")
    return folds


def fit_softmax_model(
    rows: list[dict[str, str]],
    features: list[str],
    *,
    epochs: int = 120,
    learning_rate: float = 0.05,
    l2: float = 0.001,
    class_weighting: str = "balanced",
) -> SoftmaxModel:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if l2 < 0:
        raise ValueError("l2 must be non-negative")
    if class_weighting not in {"balanced", "none"}:
        raise ValueError("class_weighting must be one of: balanced, none")

    standardizer = fit_standardizer(rows, features)
    x_rows = [standardize_row(row, features, standardizer) for row in rows]
    y_indices = [CLASSES.index(int(row["label"])) for row in rows]
    sample_weights = _sample_weights(y_indices, class_weighting=class_weighting)
    feature_count = len(features) + 1
    weights = [[0.0 for _ in range(feature_count)] for _ in CLASSES]

    for epoch in range(epochs):
        gradients = [[0.0 for _ in range(feature_count)] for _ in CLASSES]
        for x_values, label_idx, sample_weight in zip(x_rows, y_indices, sample_weights):
            x = [1.0, *x_values]
            probabilities = _softmax([_dot(class_weights, x) for class_weights in weights])
            for class_idx in range(len(CLASSES)):
                error = (probabilities[class_idx] - (1.0 if class_idx == label_idx else 0.0)) * sample_weight
                for feature_idx, value in enumerate(x):
                    gradients[class_idx][feature_idx] += error * value

        step = learning_rate / (1.0 + 0.01 * epoch)
        n = float(len(rows))
        for class_idx in range(len(CLASSES)):
            for feature_idx in range(feature_count):
                gradient = gradients[class_idx][feature_idx] / n
                if feature_idx > 0:
                    gradient += l2 * weights[class_idx][feature_idx]
                weights[class_idx][feature_idx] -= step * gradient

    return SoftmaxModel(features=features, standardizer=standardizer, weights=weights)


def select_logistic_threshold(
    *,
    model: SoftmaxModel,
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    alpha_thresholds: list[float],
    execution_model: str,
    maker_fee_bps: float,
    taker_fee_bps: float,
    slippage_bps: float,
    sort_by: str,
) -> LogisticResult:
    results = [
        _make_constant_result(
            model=model,
            train_rows=train_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
            execution_model=execution_model,
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
        ),
        *[
        _make_logistic_result(
            model=model,
            alpha_threshold=threshold,
            train_rows=train_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
            execution_model=execution_model,
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
        )
        for threshold in alpha_thresholds
        ],
    ]
    return sorted(results, key=lambda result: _sort_key(result, sort_by), reverse=True)[0]


def predict_alpha(model: SoftmaxModel, row: dict[str, str]) -> float:
    probabilities = predict_probabilities(model, row)
    return probabilities[CLASSES.index(1)] - probabilities[CLASSES.index(-1)]


def predict_side(model: SoftmaxModel, row: dict[str, str], alpha_threshold: float) -> int:
    alpha = predict_alpha(model, row)
    if alpha > alpha_threshold:
        return 1
    if alpha < -alpha_threshold:
        return -1
    return 0


def predict_probabilities(model: SoftmaxModel, row: dict[str, str]) -> list[float]:
    x = [1.0, *standardize_row(row, model.features, model.standardizer)]
    return _softmax([_dot(class_weights, x) for class_weights in model.weights])


def fit_standardizer(rows: list[dict[str, str]], features: list[str]) -> Standardizer:
    means: list[float] = []
    scales: list[float] = []
    for feature in features:
        values = [float(row[feature]) for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        scale = variance ** 0.5
        means.append(mean)
        scales.append(scale if scale > 1e-12 else 1.0)
    return Standardizer(means=means, scales=scales)


def standardize_row(row: dict[str, str], features: list[str], standardizer: Standardizer) -> list[float]:
    return [
        (float(row[feature]) - mean) / scale
        for feature, mean, scale in zip(features, standardizer.means, standardizer.scales)
    ]


def format_logistic_walk_forward_results(folds: list[LogisticWalkForwardFold]) -> str:
    lines = [
        "fold,train_rows,validation_rows,test_rows,purged_train_rows,purged_validation_rows,name,feature_count,alpha_threshold,val_macro_f1,val_signals,val_trades,val_fill_rate,val_net_pnl,val_break_even_fee_bps,test_macro_f1,test_bal_acc,test_accuracy,test_coverage,test_signals,test_trades,test_fill_rate,test_gross_pnl,test_net_pnl,test_break_even_fee_bps,test_mean_net_bps,test_median_net_bps,test_profit_factor,test_max_drawdown_pnl,test_sharpe_per_trade,test_win_rate"
    ]
    total_test_rows = 0
    total_test_signals = 0
    total_test_trades = 0
    total_test_gross_pnl = 0.0
    total_test_net_pnl = 0.0
    total_test_fee_turnover = 0.0
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
                    str(len(result.features)),
                    f"{result.alpha_threshold:.6g}",
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


def _make_logistic_result(
    *,
    model: SoftmaxModel,
    alpha_threshold: float,
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    execution_model: str,
    maker_fee_bps: float,
    taker_fee_bps: float,
    slippage_bps: float,
) -> LogisticResult:
    predictor = lambda row: predict_side(model, row, alpha_threshold)
    return LogisticResult(
        name="softmax_logistic",
        features=model.features,
        alpha_threshold=alpha_threshold,
        train=_evaluate_predictor(train_rows, predictor),
        validation=_evaluate_predictor(validation_rows, predictor),
        test=_evaluate_predictor(test_rows, predictor),
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


def _make_constant_result(
    *,
    model: SoftmaxModel,
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    execution_model: str,
    maker_fee_bps: float,
    taker_fee_bps: float,
    slippage_bps: float,
) -> LogisticResult:
    predictor = lambda row: 0
    return LogisticResult(
        name="always_flat",
        features=model.features,
        alpha_threshold=0.0,
        train=_evaluate_predictor(train_rows, predictor),
        validation=_evaluate_predictor(validation_rows, predictor),
        test=_evaluate_predictor(test_rows, predictor),
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


def _evaluate_predictor(rows: list[dict[str, str]], predictor: Callable[[dict[str, str]], int]) -> Metrics:
    return compute_metrics([int(row["label"]) for row in rows], [predictor(row) for row in rows])


def _sample_weights(y_indices: list[int], *, class_weighting: str) -> list[float]:
    if class_weighting == "none":
        return [1.0 for _ in y_indices]
    counts = {class_idx: y_indices.count(class_idx) for class_idx in range(len(CLASSES))}
    total = len(y_indices)
    return [total / (len(CLASSES) * counts[label_idx]) if counts[label_idx] else 1.0 for label_idx in y_indices]


def _sort_key(result: LogisticResult, sort_by: str) -> float:
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


def _purge_rows_crossing_boundary(
    rows: list[dict[str, str]],
    next_window_start_time_ms: int,
) -> list[dict[str, str]]:
    return [row for row in rows if _future_time_ms(row) < next_window_start_time_ms]


def _event_time_ms(row: dict[str, str]) -> int:
    return int(float(row["event_time"]))


def _future_time_ms(row: dict[str, str]) -> int:
    return int(float(row.get("future_event_time") or row["event_time"]))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _softmax(logits: list[float]) -> list[float]:
    offset = max(logits)
    exponentials = [math.exp(value - offset) for value in logits]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def _fmt(value: float) -> str:
    return f"{value:.6f}"
