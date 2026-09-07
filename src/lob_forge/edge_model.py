from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from lob_forge.baselines import DEFAULT_FEATURES, EconomicMetrics, Metrics, compute_metrics, evaluate_economics, pool_metrics
from lob_forge.live_validation import ShadowDecision, write_shadow_decisions
from lob_forge.logistic import Standardizer, fit_standardizer, standardize_row
from lob_forge.protocol import assert_valid_selection_metric


DEFAULT_EDGE_THRESHOLDS_BPS = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0]


@dataclass(frozen=True)
class EdgeModel:
    features: list[str]
    standardizer: Standardizer
    long_weights: list[float]
    short_weights: list[float]


@dataclass(frozen=True)
class EdgeResult:
    name: str
    features: list[str]
    edge_threshold_bps: float
    train: Metrics
    validation: Metrics
    test: Metrics
    train_economics: EconomicMetrics
    validation_economics: EconomicMetrics
    test_economics: EconomicMetrics


@dataclass(frozen=True)
class EdgeWalkForwardFold:
    fold: int
    train_rows: int
    validation_rows: int
    test_rows: int
    purged_train_rows: int
    purged_validation_rows: int
    result: EdgeResult


@dataclass(frozen=True)
class EdgeFoldFit:
    fold: int
    train_rows: int
    validation_rows: int
    test_rows: int
    purged_train_rows: int
    purged_validation_rows: int
    model: EdgeModel
    result: EdgeResult
    test_window_rows: list[dict[str, str]]


def available_edge_features(rows: list[dict[str, str]], features: list[str] | None = None) -> list[str]:
    return _available_features(rows, features)


def run_edge_walk_forward(
    feature_csv: Path | str,
    *,
    features: list[str] | None = None,
    edge_thresholds_bps: list[float] | None = None,
    train_size: int = 2400,
    validation_size: int = 1200,
    test_size: int = 1200,
    step_size: int | None = None,
    purge_label_overlap: bool = True,
    l2: float = 1.0,
    taker_fee_bps: float = 5.0,
    slippage_bps: float = 0.0,
    sort_by: str = "validation_net_pnl",
    max_folds: int | None = None,
) -> list[EdgeWalkForwardFold]:
    assert_valid_selection_metric(sort_by)
    rows = _read_rows(Path(feature_csv))
    if not rows:
        raise ValueError("no rows available for expected-edge walk-forward evaluation")
    effective_step, threshold_values = _validate_walk_forward_config(
        edge_thresholds_bps=edge_thresholds_bps,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        step_size=step_size,
        l2=l2,
        max_folds=max_folds,
    )

    feature_names = _available_features(rows, features)
    total_window = train_size + validation_size + test_size
    if len(rows) < total_window:
        raise ValueError("not enough rows for one expected-edge walk-forward fold")

    folds: list[EdgeWalkForwardFold] = []
    start = 0
    fold_idx = 1
    while start + total_window <= len(rows) and _should_continue_folds(folds, max_folds):
        fold = _evaluate_fold_window(
            rows[start : start + total_window],
            fold_idx=fold_idx,
            train_size=train_size,
            validation_size=validation_size,
            feature_names=feature_names,
            threshold_values=threshold_values,
            purge_label_overlap=purge_label_overlap,
            l2=l2,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
            sort_by=sort_by,
        )
        if fold is not None:
            folds.append(fold)
        start += effective_step
        fold_idx += 1

    if not folds:
        raise ValueError("no valid expected-edge walk-forward folds after purging")
    return folds


def run_edge_walk_forward_streaming(
    feature_csv: Path | str,
    *,
    features: list[str] | None = None,
    edge_thresholds_bps: list[float] | None = None,
    train_size: int = 2400,
    validation_size: int = 1200,
    test_size: int = 1200,
    step_size: int | None = None,
    purge_label_overlap: bool = True,
    l2: float = 1.0,
    taker_fee_bps: float = 5.0,
    slippage_bps: float = 0.0,
    sort_by: str = "validation_net_pnl",
    max_folds: int | None = None,
) -> list[EdgeWalkForwardFold]:
    assert_valid_selection_metric(sort_by)
    effective_step, threshold_values = _validate_walk_forward_config(
        edge_thresholds_bps=edge_thresholds_bps,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        step_size=step_size,
        l2=l2,
        max_folds=max_folds,
    )
    total_window = train_size + validation_size + test_size
    folds: list[EdgeWalkForwardFold] = []

    with Path(feature_csv).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("no rows available for expected-edge walk-forward evaluation")
        window = _read_next_rows(reader, total_window)
        if not window:
            raise ValueError("no rows available for expected-edge walk-forward evaluation")
        feature_names = _available_features_from_columns(list(reader.fieldnames), features)
        if len(window) < total_window:
            raise ValueError("not enough rows for one expected-edge walk-forward fold")

        fold_idx = 1
        while len(window) == total_window and _should_continue_folds(folds, max_folds):
            fold = _evaluate_fold_window(
                window,
                fold_idx=fold_idx,
                train_size=train_size,
                validation_size=validation_size,
                feature_names=feature_names,
                threshold_values=threshold_values,
                purge_label_overlap=purge_label_overlap,
                l2=l2,
                taker_fee_bps=taker_fee_bps,
                slippage_bps=slippage_bps,
                sort_by=sort_by,
            )
            if fold is not None:
                folds.append(fold)
            fold_idx += 1
            window = _advance_window(window, reader, step=effective_step, total_window=total_window)

    if not folds:
        raise ValueError("no valid expected-edge walk-forward folds after purging")
    return folds


def run_edge_shadow_decisions_streaming(
    feature_csv: Path | str,
    *,
    venue: str,
    symbol: str,
    intended_size: float | None = None,
    intended_notional: float | None = None,
    order_type: str = "paper_taker",
    model_name: str = "ridge_expected_edge",
    include_flat: bool = False,
    features: list[str] | None = None,
    edge_thresholds_bps: list[float] | None = None,
    train_size: int = 2400,
    validation_size: int = 1200,
    test_size: int = 1200,
    step_size: int | None = None,
    purge_label_overlap: bool = True,
    l2: float = 1.0,
    taker_fee_bps: float = 5.0,
    slippage_bps: float = 0.0,
    sort_by: str = "validation_net_pnl",
    max_folds: int | None = None,
) -> list[ShadowDecision]:
    assert_valid_selection_metric(sort_by)
    if intended_size is None and intended_notional is None:
        raise ValueError("provide intended_size or intended_notional")
    if intended_size is not None and intended_size <= 0:
        raise ValueError("intended_size must be positive")
    if intended_notional is not None and intended_notional <= 0:
        raise ValueError("intended_notional must be positive")
    if order_type not in {"paper_taker", "paper_limit"}:
        raise ValueError("order_type must be paper_taker or paper_limit")

    decisions: list[ShadowDecision] = []
    for fold_fit in _iter_edge_fold_fits_streaming(
        feature_csv,
        features=features,
        edge_thresholds_bps=edge_thresholds_bps,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        step_size=step_size,
        purge_label_overlap=purge_label_overlap,
        l2=l2,
        taker_fee_bps=taker_fee_bps,
        slippage_bps=slippage_bps,
        sort_by=sort_by,
        max_folds=max_folds,
    ):
        if fold_fit.result.name == "always_flat" and not include_flat:
            continue
        for row_index, row in enumerate(fold_fit.test_window_rows):
            side, predicted_edge_bps = _predict_side_and_net_edge(
                fold_fit.model,
                row,
                edge_threshold_bps=fold_fit.result.edge_threshold_bps,
                taker_fee_bps=taker_fee_bps,
                slippage_bps=slippage_bps,
            )
            if fold_fit.result.name == "always_flat":
                side, predicted_edge_bps = 0, 0.0
            if side == 0 and not include_flat:
                continue
            intended_price = _intended_order_price(row, side=side, order_type=order_type)
            if intended_size is not None:
                size = intended_size
            else:
                assert intended_notional is not None
                size = intended_notional / intended_price
            decisions.append(
                ShadowDecision(
                    decision_id=_decision_id(symbol=symbol, fold=fold_fit.fold, row_index=row_index, row=row),
                    timestamp_ms=_event_time_ms(row),
                    venue=venue,
                    symbol=symbol.upper(),
                    model_name="always_flat" if fold_fit.result.name == "always_flat" else model_name,
                    predicted_side=side,
                    predicted_edge_bps=predicted_edge_bps,
                    order_type=order_type,
                    intended_price=intended_price,
                    intended_size=size,
                    notes=f"fold={fold_fit.fold};threshold_bps={fold_fit.result.edge_threshold_bps:.12g}",
                )
            )
    return decisions


def write_edge_shadow_decisions_streaming(output_path: Path | str, feature_csv: Path | str, **kwargs) -> Path:
    decisions = run_edge_shadow_decisions_streaming(feature_csv, **kwargs)
    return write_shadow_decisions(output_path, decisions)


def fit_edge_model(rows: list[dict[str, str]], features: list[str], *, l2: float = 1.0) -> EdgeModel:
    if not rows:
        raise ValueError("cannot fit expected-edge model on empty rows")
    if l2 < 0:
        raise ValueError("l2 must be non-negative")

    standardizer = fit_standardizer(rows, features)
    x_rows = [[1.0, *standardize_row(row, features, standardizer)] for row in rows]
    long_targets = [_long_taker_gross_bps(row) for row in rows]
    short_targets = [_short_taker_gross_bps(row) for row in rows]
    long_weights = _fit_ridge_weights(x_rows, long_targets, l2=l2)
    short_weights = _fit_ridge_weights(x_rows, short_targets, l2=l2)
    return EdgeModel(
        features=features,
        standardizer=standardizer,
        long_weights=long_weights,
        short_weights=short_weights,
    )


def select_edge_threshold(
    *,
    model: EdgeModel,
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    edge_thresholds_bps: list[float],
    taker_fee_bps: float,
    slippage_bps: float,
    sort_by: str,
) -> EdgeResult:
    assert_valid_selection_metric(sort_by)
    if sort_by.startswith("validation_"):
        return _select_edge_threshold_by_validation(
            model=model,
            train_rows=train_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
            edge_thresholds_bps=edge_thresholds_bps,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
            sort_by=sort_by,
        )

    results = [
        _make_constant_result(
            model=model,
            train_rows=train_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
        ),
        *[
            _make_edge_result(
                model=model,
                edge_threshold_bps=threshold,
                train_rows=train_rows,
                validation_rows=validation_rows,
                test_rows=test_rows,
                taker_fee_bps=taker_fee_bps,
                slippage_bps=slippage_bps,
            )
            for threshold in edge_thresholds_bps
        ],
    ]
    return sorted(results, key=lambda result: _sort_key(result, sort_by), reverse=True)[0]


def _select_edge_threshold_by_validation(
    *,
    model: EdgeModel,
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    edge_thresholds_bps: list[float],
    taker_fee_bps: float,
    slippage_bps: float,
    sort_by: str,
) -> EdgeResult:
    best_name = "always_flat"
    best_threshold = 0.0
    best_score = _validation_sort_score(
        validation_rows,
        lambda row: 0,
        taker_fee_bps=taker_fee_bps,
        slippage_bps=slippage_bps,
        sort_by=sort_by,
    )

    for threshold in edge_thresholds_bps:

        def predictor(row: dict[str, str], threshold: float = threshold) -> int:
            return predict_side(
                model,
                row,
                edge_threshold_bps=threshold,
                taker_fee_bps=taker_fee_bps,
                slippage_bps=slippage_bps,
            )

        score = _validation_sort_score(
            validation_rows,
            predictor,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
            sort_by=sort_by,
        )
        if score > best_score:
            best_name = "ridge_expected_edge"
            best_threshold = threshold
            best_score = score

    if best_name == "always_flat":
        return _make_constant_result(
            model=model,
            train_rows=train_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
        )
    return _make_edge_result(
        model=model,
        edge_threshold_bps=best_threshold,
        train_rows=train_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
        taker_fee_bps=taker_fee_bps,
        slippage_bps=slippage_bps,
    )


def predict_side(
    model: EdgeModel,
    row: dict[str, str],
    *,
    edge_threshold_bps: float,
    taker_fee_bps: float,
    slippage_bps: float,
) -> int:
    long_net_bps, short_net_bps = predict_net_edges_bps(
        model, row, taker_fee_bps=taker_fee_bps, slippage_bps=slippage_bps
    )
    if long_net_bps > edge_threshold_bps and long_net_bps >= short_net_bps:
        return 1
    if short_net_bps > edge_threshold_bps and short_net_bps > long_net_bps:
        return -1
    return 0


def predict_net_edges_bps(
    model: EdgeModel,
    row: dict[str, str],
    *,
    taker_fee_bps: float,
    slippage_bps: float,
) -> tuple[float, float]:
    long_gross_bps, short_gross_bps = predict_gross_edges_bps(model, row)
    if any(not math.isfinite(value) or value < 0 for value in (taker_fee_bps, slippage_bps)):
        raise ValueError("taker_fee_bps and slippage_bps must be finite and non-negative")
    cost = taker_fee_bps + slippage_bps
    return (1.0-cost/10_000.0)*long_gross_bps-2.0*cost, (1.0+cost/10_000.0)*short_gross_bps-2.0*cost


def predict_gross_edges_bps(model: EdgeModel, row: dict[str, str]) -> tuple[float, float]:
    x = [1.0, *standardize_row(row, model.features, model.standardizer)]
    return _dot(model.long_weights, x), _dot(model.short_weights, x)


def format_edge_walk_forward_results(folds: list[EdgeWalkForwardFold]) -> str:
    lines = [
        "fold,train_rows,validation_rows,test_rows,purged_train_rows,purged_validation_rows,name,feature_count,edge_threshold_bps,val_macro_f1,val_signals,val_trades,val_fill_rate,val_net_pnl,val_break_even_fee_bps,test_macro_f1,test_bal_acc,test_accuracy,test_coverage,test_signals,test_trades,test_fill_rate,test_gross_pnl,test_net_pnl,test_break_even_fee_bps,test_mean_net_bps,test_median_net_bps,test_profit_factor,test_max_drawdown_pnl,test_sharpe_per_trade,test_win_rate"
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
                    f"{result.edge_threshold_bps:.6g}",
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
        summary_break_even_fee_bps = (
            sum(
                fold.result.test_economics.break_even_taker_fee_bps * fold.result.test_economics.fee_turnover
                for fold in folds
            )
            / total_test_fee_turnover
        )

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
                _fmt(pool_metrics([fold.result.test for fold in folds]).macro_f1),
                _fmt(pool_metrics([fold.result.test for fold in folds]).balanced_accuracy),
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


def _make_edge_result(
    *,
    model: EdgeModel,
    edge_threshold_bps: float,
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    taker_fee_bps: float,
    slippage_bps: float,
) -> EdgeResult:
    def predictor(row: dict[str, str]) -> int:
        return predict_side(
            model,
            row,
            edge_threshold_bps=edge_threshold_bps,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
        )

    return EdgeResult(
        name="ridge_expected_edge",
        features=model.features,
        edge_threshold_bps=edge_threshold_bps,
        train=_evaluate_predictor(train_rows, predictor),
        validation=_evaluate_predictor(validation_rows, predictor),
        test=_evaluate_predictor(test_rows, predictor),
        train_economics=evaluate_economics(
            train_rows,
            predictor,
            execution_model="taker",
            maker_fee_bps=0.0,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
        ),
        validation_economics=evaluate_economics(
            validation_rows,
            predictor,
            execution_model="taker",
            maker_fee_bps=0.0,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
        ),
        test_economics=evaluate_economics(
            test_rows,
            predictor,
            execution_model="taker",
            maker_fee_bps=0.0,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
        ),
    )


def _make_constant_result(
    *,
    model: EdgeModel,
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    taker_fee_bps: float,
    slippage_bps: float,
) -> EdgeResult:
    def predictor(row: dict[str, str]) -> int:
        return 0

    return EdgeResult(
        name="always_flat",
        features=model.features,
        edge_threshold_bps=0.0,
        train=_evaluate_predictor(train_rows, predictor),
        validation=_evaluate_predictor(validation_rows, predictor),
        test=_evaluate_predictor(test_rows, predictor),
        train_economics=evaluate_economics(
            train_rows,
            predictor,
            execution_model="taker",
            maker_fee_bps=0.0,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
        ),
        validation_economics=evaluate_economics(
            validation_rows,
            predictor,
            execution_model="taker",
            maker_fee_bps=0.0,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
        ),
        test_economics=evaluate_economics(
            test_rows,
            predictor,
            execution_model="taker",
            maker_fee_bps=0.0,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
        ),
    )


def _fit_ridge_weights(x_rows: list[list[float]], y_values: list[float], *, l2: float) -> list[float]:
    """Solve the augmented least-squares problem using Householder QR.

    This avoids squaring the condition number in X.T @ X. The intercept is
    unpenalized. A singular unregularized fit fails rather than adding hidden
    jitter or returning an arbitrary coefficient vector.
    """
    columns = len(x_rows[0])
    if len(x_rows) != len(y_values) or any(len(row) != columns for row in x_rows):
        raise ValueError("ridge design and target dimensions do not match")
    if not math.isfinite(l2) or l2 < 0:
        raise ValueError("l2 must be finite and non-negative")
    if not all(math.isfinite(v) for row in x_rows for v in row) or not all(math.isfinite(v) for v in y_values):
        raise ValueError("ridge inputs must be finite")
    matrix = [list(row) for row in x_rows]
    target = list(y_values)
    if l2 > 0:
        penalty = math.sqrt(l2)
        for column in range(1, columns):
            matrix.append([penalty if j == column else 0.0 for j in range(columns)])
            target.append(0.0)
    if len(matrix) < columns:
        raise ValueError("singular unregularized ridge design; use positive l2")
    for column in range(columns):
        vector = [matrix[i][column] for i in range(column, len(matrix))]
        norm = math.hypot(*vector)
        if norm <= 1e-12:
            raise ValueError("singular unregularized ridge design; use positive l2")
        vector = [v/norm for v in vector]
        vector[0] += math.copysign(1.0, vector[0])
        scale = math.hypot(*vector)
        vector = [v/scale for v in vector]
        for j in range(column, columns):
            projection = 2*math.fsum(v*matrix[column+i][j] for i,v in enumerate(vector))
            for i,v in enumerate(vector):
                matrix[column+i][j] -= projection*v
        projection = 2*math.fsum(v*target[column+i] for i,v in enumerate(vector))
        for i,v in enumerate(vector):
            target[column+i] -= projection*v
    weights = [0.0]*columns
    for i in range(columns-1, -1, -1):
        weights[i] = (target[i]-math.fsum(matrix[i][j]*weights[j] for j in range(i+1,columns)))/matrix[i][i]
    return weights


def _long_taker_gross_bps(row: dict[str, str]) -> float:
    entry = float(row.get("entry_ask") or row["ask"])
    exit_price = float(row["future_bid"])
    if not all(math.isfinite(x) and x > 0 for x in (entry, exit_price)):
        raise ValueError("edge targets require finite positive executable prices")
    return 10_000.0 * (exit_price - entry) / entry


def _short_taker_gross_bps(row: dict[str, str]) -> float:
    entry = float(row.get("entry_bid") or row["bid"])
    exit_price = float(row["future_ask"])
    if not all(math.isfinite(x) and x > 0 for x in (entry, exit_price)):
        raise ValueError("edge targets require finite positive executable prices")
    return 10_000.0 * (entry - exit_price) / entry


def _evaluate_predictor(rows: list[dict[str, str]], predictor: Callable[[dict[str, str]], int]) -> Metrics:
    return compute_metrics([int(row["label"]) for row in rows], [predictor(row) for row in rows])


def _sort_key(result: EdgeResult, sort_by: str) -> float:
    assert_valid_selection_metric(sort_by)
    if sort_by == "validation_macro_f1":
        return result.validation.macro_f1
    if sort_by == "validation_balanced_accuracy":
        return result.validation.balanced_accuracy
    if sort_by == "validation_net_pnl":
        return result.validation_economics.net_pnl
    if sort_by == "validation_gross_pnl":
        return result.validation_economics.gross_pnl
    raise ValueError(
        "sort_by must be one of: validation_macro_f1, validation_balanced_accuracy, "
        "validation_net_pnl, validation_gross_pnl"
    )


def _validation_sort_score(
    rows: list[dict[str, str]],
    predictor: Callable[[dict[str, str]], int],
    *,
    taker_fee_bps: float,
    slippage_bps: float,
    sort_by: str,
) -> float:
    if sort_by == "validation_macro_f1":
        return _evaluate_predictor(rows, predictor).macro_f1
    if sort_by == "validation_balanced_accuracy":
        return _evaluate_predictor(rows, predictor).balanced_accuracy
    if sort_by in {"validation_net_pnl", "validation_gross_pnl"}:
        economics = evaluate_economics(
            rows,
            predictor,
            execution_model="taker",
            maker_fee_bps=0.0,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
        )
        if sort_by == "validation_net_pnl":
            return economics.net_pnl
        return economics.gross_pnl
    raise ValueError(
        "sort_by must be one of: validation_macro_f1, validation_balanced_accuracy, "
        "validation_net_pnl, validation_gross_pnl"
    )


def _validate_walk_forward_config(
    *,
    edge_thresholds_bps: list[float] | None,
    train_size: int,
    validation_size: int,
    test_size: int,
    step_size: int | None,
    l2: float,
    max_folds: int | None,
) -> tuple[int, list[float]]:
    if train_size <= 0 or validation_size <= 0 or test_size <= 0:
        raise ValueError("train_size, validation_size, and test_size must be positive")
    if l2 < 0:
        raise ValueError("l2 must be non-negative")
    effective_step = test_size if step_size is None else step_size
    if effective_step <= 0:
        raise ValueError("step_size must be positive")
    if effective_step < test_size:
        raise ValueError("step_size must be at least test_size so OOS test windows do not overlap")
    if max_folds is not None and max_folds <= 0:
        raise ValueError("max_folds must be positive")
    return effective_step, edge_thresholds_bps or DEFAULT_EDGE_THRESHOLDS_BPS


def _should_continue_folds(folds: list[EdgeWalkForwardFold], max_folds: int | None) -> bool:
    return max_folds is None or len(folds) < max_folds


def _should_continue_fits(fits: list[EdgeFoldFit], max_folds: int | None) -> bool:
    return max_folds is None or len(fits) < max_folds


def _iter_edge_fold_fits_streaming(
    feature_csv: Path | str,
    *,
    features: list[str] | None,
    edge_thresholds_bps: list[float] | None,
    train_size: int,
    validation_size: int,
    test_size: int,
    step_size: int | None,
    purge_label_overlap: bool,
    l2: float,
    taker_fee_bps: float,
    slippage_bps: float,
    sort_by: str,
    max_folds: int | None,
) -> list[EdgeFoldFit]:
    effective_step, threshold_values = _validate_walk_forward_config(
        edge_thresholds_bps=edge_thresholds_bps,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        step_size=step_size,
        l2=l2,
        max_folds=max_folds,
    )
    total_window = train_size + validation_size + test_size
    fits: list[EdgeFoldFit] = []

    with Path(feature_csv).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("no rows available for expected-edge shadow decision export")
        window = _read_next_rows(reader, total_window)
        if not window:
            raise ValueError("no rows available for expected-edge shadow decision export")
        feature_names = _available_features_from_columns(list(reader.fieldnames), features)
        if len(window) < total_window:
            raise ValueError("not enough rows for one expected-edge walk-forward fold")

        fold_idx = 1
        while len(window) == total_window and _should_continue_fits(fits, max_folds):
            fit = _fit_fold_window(
                window,
                fold_idx=fold_idx,
                train_size=train_size,
                validation_size=validation_size,
                feature_names=feature_names,
                threshold_values=threshold_values,
                purge_label_overlap=purge_label_overlap,
                l2=l2,
                taker_fee_bps=taker_fee_bps,
                slippage_bps=slippage_bps,
                sort_by=sort_by,
            )
            if fit is not None:
                fits.append(fit)
            fold_idx += 1
            window = _advance_window(window, reader, step=effective_step, total_window=total_window)

    if not fits:
        raise ValueError("no valid expected-edge walk-forward folds after purging")
    return fits


def _evaluate_fold_window(
    rows: list[dict[str, str]],
    *,
    fold_idx: int,
    train_size: int,
    validation_size: int,
    feature_names: list[str],
    threshold_values: list[float],
    purge_label_overlap: bool,
    l2: float,
    taker_fee_bps: float,
    slippage_bps: float,
    sort_by: str,
) -> EdgeWalkForwardFold | None:
    fit = _fit_fold_window(
        rows,
        fold_idx=fold_idx,
        train_size=train_size,
        validation_size=validation_size,
        feature_names=feature_names,
        threshold_values=threshold_values,
        purge_label_overlap=purge_label_overlap,
        l2=l2,
        taker_fee_bps=taker_fee_bps,
        slippage_bps=slippage_bps,
        sort_by=sort_by,
    )
    if fit is None:
        return None
    return EdgeWalkForwardFold(
        fold=fit.fold,
        train_rows=fit.train_rows,
        validation_rows=fit.validation_rows,
        test_rows=fit.test_rows,
        purged_train_rows=fit.purged_train_rows,
        purged_validation_rows=fit.purged_validation_rows,
        result=fit.result,
    )


def _fit_fold_window(
    rows: list[dict[str, str]],
    *,
    fold_idx: int,
    train_size: int,
    validation_size: int,
    feature_names: list[str],
    threshold_values: list[float],
    purge_label_overlap: bool,
    l2: float,
    taker_fee_bps: float,
    slippage_bps: float,
    sort_by: str,
) -> EdgeFoldFit | None:
    train_raw = rows[:train_size]
    validation_raw = rows[train_size : train_size + validation_size]
    test_rows = rows[train_size + validation_size :]

    train_rows = train_raw
    validation_rows = validation_raw
    if purge_label_overlap:
        validation_start_time = _event_time_ms(validation_raw[0])
        test_start_time = _event_time_ms(test_rows[0])
        train_rows = _purge_rows_crossing_boundary(train_raw, validation_start_time)
        validation_rows = _purge_rows_crossing_boundary(validation_raw, test_start_time)

    if not train_rows or not validation_rows or not test_rows:
        return None

    model = fit_edge_model(train_rows, feature_names, l2=l2)
    result = select_edge_threshold(
        model=model,
        train_rows=train_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
        edge_thresholds_bps=threshold_values,
        taker_fee_bps=taker_fee_bps,
        slippage_bps=slippage_bps,
        sort_by=sort_by,
    )
    return EdgeFoldFit(
        fold=fold_idx,
        train_rows=len(train_rows),
        validation_rows=len(validation_rows),
        test_rows=len(test_rows),
        purged_train_rows=len(train_raw) - len(train_rows),
        purged_validation_rows=len(validation_raw) - len(validation_rows),
        model=model,
        result=result,
        test_window_rows=test_rows,
    )


def _available_features(rows: list[dict[str, str]], features: list[str] | None) -> list[str]:
    return _available_features_from_columns(list(rows[0]), features)


def _available_features_from_columns(columns: Sequence[str], features: list[str] | None) -> list[str]:
    row_features = set(columns)
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


def _predict_side_and_net_edge(
    model: EdgeModel,
    row: dict[str, str],
    *,
    edge_threshold_bps: float,
    taker_fee_bps: float,
    slippage_bps: float,
) -> tuple[int, float]:
    long_net_bps, short_net_bps = predict_net_edges_bps(
        model,
        row,
        taker_fee_bps=taker_fee_bps,
        slippage_bps=slippage_bps,
    )
    if long_net_bps > edge_threshold_bps and long_net_bps >= short_net_bps:
        return 1, long_net_bps
    if short_net_bps > edge_threshold_bps and short_net_bps > long_net_bps:
        return -1, short_net_bps
    return 0, max(long_net_bps, short_net_bps)


def _intended_order_price(row: dict[str, str], *, side: int, order_type: str) -> float:
    if order_type == "paper_taker":
        if side == 1:
            return float(row.get("entry_ask") or row["ask"])
        if side == -1:
            return float(row.get("entry_bid") or row["bid"])
        return _entry_mid(row)
    if order_type == "paper_limit":
        if side == 1:
            return float(row.get("entry_bid") or row["bid"])
        if side == -1:
            return float(row.get("entry_ask") or row["ask"])
        return _entry_mid(row)
    raise ValueError("order_type must be paper_taker or paper_limit")


def _entry_mid(row: dict[str, str]) -> float:
    if row.get("entry_mid"):
        return float(row["entry_mid"])
    bid = float(row.get("entry_bid") or row["bid"])
    ask = float(row.get("entry_ask") or row["ask"])
    return (bid + ask) / 2.0


def _decision_id(*, symbol: str, fold: int, row_index: int, row: dict[str, str]) -> str:
    source_date = row.get("source_date", "")
    parts = [symbol.upper()]
    if source_date:
        parts.append(source_date)
    parts.extend([f"f{fold}", str(_event_time_ms(row)), str(row_index)])
    return "-".join(parts)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_next_rows(reader: csv.DictReader, count: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for _ in range(count):
        try:
            rows.append(next(reader))
        except StopIteration:
            break
    return rows


def _advance_window(
    window: list[dict[str, str]],
    reader: csv.DictReader,
    *,
    step: int,
    total_window: int,
) -> list[dict[str, str]]:
    if step < total_window:
        next_window = window[step:]
        next_window.extend(_read_next_rows(reader, step))
        return next_window

    for _ in range(step - total_window):
        try:
            next(reader)
        except StopIteration:
            return []
    return _read_next_rows(reader, total_window)


def _dot(left: list[float], right: list[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def _fmt(value: float) -> str:
    return f"{value:.6f}"
