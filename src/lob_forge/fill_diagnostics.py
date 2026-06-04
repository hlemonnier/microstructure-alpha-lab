from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from lob_forge.baselines import predict_feature_threshold


@dataclass(frozen=True)
class FillDiagnostics:
    group: str
    side: str
    signals: int
    fills: int
    fill_rate: float
    gross_pnl: float
    net_pnl: float
    exit_fee_turnover: float
    break_even_exit_taker_fee_bps: float
    mean_net_pnl_per_fill: float
    mean_net_return_bps_per_fill: float
    win_rate: float
    mean_fill_latency_ms: float


@dataclass(frozen=True)
class FillRegimeDiagnostics:
    regime_feature: str
    bucket: int
    lower_bound: float
    upper_bound: float
    side: str
    diagnostics: FillDiagnostics


def run_fill_diagnostics(
    feature_csv: Path | str,
    *,
    feature: str,
    threshold: float,
    maker_fee_bps: float = 0.0,
    taker_fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    source_date: str | None = None,
    by_source_date: bool = False,
) -> list[FillDiagnostics]:
    rows = _read_rows(Path(feature_csv))
    if source_date is not None:
        rows = [row for row in rows if row.get("source_date") == source_date]
    if not rows:
        raise ValueError("no rows available for fill diagnostics")
    if feature not in rows[0]:
        raise ValueError(f"feature column not found: {feature}")

    groups: list[tuple[str, list[dict[str, str]]]]
    if by_source_date:
        dates = sorted({row.get("source_date", "unknown") or "unknown" for row in rows})
        groups = [(date, [row for row in rows if (row.get("source_date", "unknown") or "unknown") == date]) for date in dates]
        groups.append(("all", rows))
    else:
        groups = [("all", rows)]

    diagnostics: list[FillDiagnostics] = []
    for group_name, group_rows in groups:
        for side_name, side_filter in [
            ("all", 0),
            ("long", 1),
            ("short", -1),
        ]:
            diagnostics.append(
                _compute_group(
                    group_name,
                    side_name,
                    group_rows,
                    feature=feature,
                    threshold=threshold,
                    side_filter=side_filter,
                    maker_fee_bps=maker_fee_bps,
                    taker_fee_bps=taker_fee_bps,
                    slippage_bps=slippage_bps,
                )
            )
    return diagnostics


def run_fill_regime_diagnostics(
    feature_csv: Path | str,
    *,
    feature: str,
    threshold: float,
    regime_features: list[str],
    bins: int = 3,
    maker_fee_bps: float = 0.0,
    taker_fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    source_date: str | None = None,
) -> list[FillRegimeDiagnostics]:
    rows = _read_rows(Path(feature_csv))
    if source_date is not None:
        rows = [row for row in rows if row.get("source_date") == source_date]
    if not rows:
        raise ValueError("no rows available for fill regime diagnostics")
    if feature not in rows[0]:
        raise ValueError(f"feature column not found: {feature}")
    if bins <= 0:
        raise ValueError("bins must be positive")
    missing = [name for name in regime_features if name not in rows[0]]
    if missing:
        raise ValueError(f"regime feature columns not found: {', '.join(missing)}")

    diagnostics: list[FillRegimeDiagnostics] = []
    for regime_feature in regime_features:
        for bucket, lower_bound, upper_bound, bucket_rows in _quantile_buckets(rows, regime_feature, bins):
            for side_name, side_filter in [
                ("all", 0),
                ("long", 1),
                ("short", -1),
            ]:
                item = _compute_group(
                    f"{regime_feature}_bucket_{bucket}",
                    side_name,
                    bucket_rows,
                    feature=feature,
                    threshold=threshold,
                    side_filter=side_filter,
                    maker_fee_bps=maker_fee_bps,
                    taker_fee_bps=taker_fee_bps,
                    slippage_bps=slippage_bps,
                )
                diagnostics.append(
                    FillRegimeDiagnostics(
                        regime_feature=regime_feature,
                        bucket=bucket,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        side=side_name,
                        diagnostics=item,
                    )
                )
    return diagnostics


def format_fill_diagnostics(diagnostics: list[FillDiagnostics]) -> str:
    lines = [
        "group,side,signals,fills,fill_rate,gross_pnl,net_pnl,break_even_exit_taker_fee_bps,mean_net_pnl_per_fill,mean_net_bps_per_fill,win_rate,mean_fill_latency_ms"
    ]
    for item in diagnostics:
        lines.append(
            ",".join(
                [
                    item.group,
                    item.side,
                    str(item.signals),
                    str(item.fills),
                    _fmt(item.fill_rate),
                    _fmt(item.gross_pnl),
                    _fmt(item.net_pnl),
                    _fmt(item.break_even_exit_taker_fee_bps),
                    _fmt(item.mean_net_pnl_per_fill),
                    _fmt(item.mean_net_return_bps_per_fill),
                    _fmt(item.win_rate),
                    _fmt(item.mean_fill_latency_ms),
                ]
            )
        )
    return "\n".join(lines)


def format_fill_regime_diagnostics(diagnostics: list[FillRegimeDiagnostics]) -> str:
    lines = [
        "regime_feature,bucket,lower_bound,upper_bound,side,signals,fills,fill_rate,gross_pnl,net_pnl,break_even_exit_taker_fee_bps,mean_net_pnl_per_fill,mean_net_bps_per_fill,win_rate,mean_fill_latency_ms"
    ]
    for item in diagnostics:
        diagnostic = item.diagnostics
        lines.append(
            ",".join(
                [
                    item.regime_feature,
                    str(item.bucket),
                    _fmt(item.lower_bound),
                    _fmt(item.upper_bound),
                    item.side,
                    str(diagnostic.signals),
                    str(diagnostic.fills),
                    _fmt(diagnostic.fill_rate),
                    _fmt(diagnostic.gross_pnl),
                    _fmt(diagnostic.net_pnl),
                    _fmt(diagnostic.break_even_exit_taker_fee_bps),
                    _fmt(diagnostic.mean_net_pnl_per_fill),
                    _fmt(diagnostic.mean_net_return_bps_per_fill),
                    _fmt(diagnostic.win_rate),
                    _fmt(diagnostic.mean_fill_latency_ms),
                ]
            )
        )
    return "\n".join(lines)


def _compute_group(
    group: str,
    side_name: str,
    rows: list[dict[str, str]],
    *,
    feature: str,
    threshold: float,
    side_filter: int,
    maker_fee_bps: float,
    taker_fee_bps: float,
    slippage_bps: float,
) -> FillDiagnostics:
    maker_fee_rate = maker_fee_bps / 10_000.0
    taker_fee_rate = taker_fee_bps / 10_000.0
    slippage_rate = slippage_bps / 10_000.0
    signals = 0
    fills = 0
    gross_pnl = 0.0
    net_pnl = 0.0
    net_return_bps = 0.0
    exit_fee_turnover = 0.0
    maker_fee_turnover = 0.0
    wins = 0
    fill_latency_ms = 0.0

    for row in rows:
        side = predict_feature_threshold(row, feature, threshold)
        if side == 0:
            continue
        if side_filter and side != side_filter:
            continue
        signals += 1

        if side > 0:
            if not _row_truthy(row.get("maker_long_fillable")):
                continue
            entry = float(row.get("entry_bid") or row["bid"])
            exit_price = float(row["future_bid"])
            trade_gross = exit_price - entry
            fill_event_time = row.get("maker_long_fill_event_time")
        else:
            if not _row_truthy(row.get("maker_short_fillable")):
                continue
            entry = float(row.get("entry_ask") or row["ask"])
            exit_price = float(row["future_ask"])
            trade_gross = entry - exit_price
            fill_event_time = row.get("maker_short_fill_event_time")

        maker_fee_cost = maker_fee_rate * entry
        exit_fee_cost = taker_fee_rate * exit_price
        slippage_cost = slippage_rate * exit_price
        trade_net = trade_gross - maker_fee_cost - exit_fee_cost - slippage_cost

        fills += 1
        gross_pnl += trade_gross
        net_pnl += trade_net
        maker_fee_turnover += entry
        exit_fee_turnover += exit_price
        net_return_bps += 10_000.0 * trade_net / entry if entry else 0.0
        if trade_net > 0:
            wins += 1
        fill_latency_ms += _fill_latency_ms(row, fill_event_time)

    break_even_exit_taker_fee_bps = (
        ((gross_pnl - maker_fee_rate * maker_fee_turnover - slippage_rate * exit_fee_turnover)
         / exit_fee_turnover
         * 10_000.0)
        if exit_fee_turnover
        else 0.0
    )
    return FillDiagnostics(
        group=group,
        side=side_name,
        signals=signals,
        fills=fills,
        fill_rate=fills / signals if signals else 0.0,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        exit_fee_turnover=exit_fee_turnover,
        break_even_exit_taker_fee_bps=break_even_exit_taker_fee_bps,
        mean_net_pnl_per_fill=net_pnl / fills if fills else 0.0,
        mean_net_return_bps_per_fill=net_return_bps / fills if fills else 0.0,
        win_rate=wins / fills if fills else 0.0,
        mean_fill_latency_ms=fill_latency_ms / fills if fills else 0.0,
    )


def _fill_latency_ms(row: dict[str, str], fill_event_time: str | None) -> float:
    if not fill_event_time:
        return 0.0
    return max(0.0, float(fill_event_time) - float(row.get("entry_event_time") or row["event_time"]))


def _quantile_buckets(
    rows: list[dict[str, str]],
    feature: str,
    bins: int,
) -> list[tuple[int, float, float, list[dict[str, str]]]]:
    finite_rows = [
        (value, row)
        for row in rows
        if (value := _safe_float(row.get(feature, ""))) is not None
    ]
    if not finite_rows:
        raise ValueError(f"no finite values available for regime feature: {feature}")
    finite_rows.sort(key=lambda item: item[0])
    bucket_count = min(bins, len(finite_rows))
    buckets: list[tuple[int, float, float, list[dict[str, str]]]] = []
    for bucket_idx in range(bucket_count):
        start = bucket_idx * len(finite_rows) // bucket_count
        end = (bucket_idx + 1) * len(finite_rows) // bucket_count
        bucket_items = finite_rows[start:end]
        values = [value for value, _ in bucket_items]
        buckets.append((bucket_idx + 1, values[0], values[-1], [row for _, row in bucket_items]))
    return buckets


def _safe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _row_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes"}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _fmt(value: float) -> str:
    return f"{value:.6f}"
