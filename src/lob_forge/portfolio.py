from __future__ import annotations

import csv
import heapq
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from lob_forge.statistics import autocorrelation


@dataclass(frozen=True)
class PortfolioConfig:
    capital: float
    base_notional: float
    max_notional: float
    taker_fee_bps: float = 0.0
    slippage_bps: float = 0.0
    initial_margin_rate: float = 1.0
    inventory_penalty_bps: float = 0.0
    max_inventory_notional: float | None = None
    daily_loss_limit: float | None = None
    rolling_loss_limit: float | None = None
    rolling_window: int = 20


@dataclass(frozen=True)
class Trade:
    entry_time_ms: int
    exit_time_ms: int
    side: int
    entry_price: float
    exit_price: float
    predicted_edge_bps: float = 0.0
    notional: float | None = None


@dataclass(frozen=True)
class SimulatedTrade:
    trade: Trade
    notional: float
    quantity: float
    gross_pnl: float
    net_pnl: float
    return_on_capital: float
    inventory_notional: float  # Signed net inventory immediately after this entry decision.
    margin_used: float  # Initial margin on gross active exposure after this entry decision.
    rejected: bool = False
    rejection_reason: str = ""


@dataclass(frozen=True)
class DailyRisk:
    day: str
    pnl: float
    return_on_capital: float
    trade_count: int
    kill_switch_triggered: bool


@dataclass(frozen=True)
class PortfolioResult:
    trades: list[SimulatedTrade]
    daily_risk: list[DailyRisk]
    total_net_pnl: float
    return_on_capital: float
    max_leverage: float
    max_exposure: float  # Peak gross active notional.
    max_margin_used: float
    max_concurrency: int
    max_inventory: float  # Peak absolute signed net inventory.
    daily_sharpe: float
    daily_autocorrelation: float
    calmar_like: float
    kill_switch_triggered: bool


@dataclass(frozen=True)
class VarianceStabilityReport:
    path: Path
    column: str
    observations: int
    window_size: int
    window_variances: tuple[float, ...]
    variance_mean: float
    variance_std: float
    variance_cv: float
    min_observations: int
    max_variance_cv: float

    @property
    def passed(self) -> bool:
        return (
            self.observations >= self.min_observations
            and len(self.window_variances) >= 2
            and self.variance_mean > 0.0
            and self.variance_cv <= self.max_variance_cv
        )


def simulate_fixed_notional_portfolio(trades: list[Trade], config: PortfolioConfig) -> PortfolioResult:
    if not math.isfinite(config.capital) or config.capital <= 0.0:
        raise ValueError("capital must be positive")
    if (
        not math.isfinite(config.base_notional)
        or not math.isfinite(config.max_notional)
        or config.base_notional <= 0.0
        or config.max_notional <= 0.0
    ):
        raise ValueError("notional limits must be positive")
    if not math.isfinite(config.initial_margin_rate) or config.initial_margin_rate < 0.0:
        raise ValueError("initial_margin_rate must be non-negative")
    if not math.isfinite(config.inventory_penalty_bps) or config.inventory_penalty_bps < 0.0:
        raise ValueError("inventory_penalty_bps must be non-negative")
    if (
        not math.isfinite(config.taker_fee_bps)
        or not math.isfinite(config.slippage_bps)
        or config.taker_fee_bps < 0.0
        or config.slippage_bps < 0.0
    ):
        raise ValueError("fees and slippage must be non-negative")
    if config.max_inventory_notional is not None and (
        not math.isfinite(config.max_inventory_notional) or config.max_inventory_notional < 0.0
    ):
        raise ValueError("max_inventory_notional must be non-negative")
    if config.daily_loss_limit is not None and not math.isfinite(config.daily_loss_limit):
        raise ValueError("daily_loss_limit must be finite")
    if config.rolling_loss_limit is not None and not math.isfinite(config.rolling_loss_limit):
        raise ValueError("rolling_loss_limit must be finite")
    if config.rolling_window <= 0:
        raise ValueError("rolling_window must be positive")
    for trade in trades:
        _validate_trade(trade)

    simulated: list[SimulatedTrade] = []
    active: list[tuple[int, int, SimulatedTrade]] = []
    kill_switch = False
    rolling_pnl: list[float] = []
    daily_pnl: dict[str, float] = {}
    daily_counts: dict[str, int] = {}
    daily_kill_days: set[str] = set()
    current_inventory = 0.0
    current_exposure = 0.0
    max_inventory = 0.0
    max_exposure = 0.0

    def update_peaks() -> None:
        nonlocal max_inventory, max_exposure
        max_inventory = max(max_inventory, abs(current_inventory))
        max_exposure = max(max_exposure, current_exposure)

    def realize(item: SimulatedTrade) -> None:
        nonlocal current_inventory, current_exposure, kill_switch
        current_inventory -= item.trade.side * item.notional
        current_exposure = max(0.0, current_exposure - item.notional)
        if abs(current_inventory) <= 1e-12 * max(1.0, current_exposure):
            current_inventory = 0.0
        update_peaks()

        day = _day(item.trade.exit_time_ms)
        daily_pnl[day] = daily_pnl.get(day, 0.0) + item.net_pnl
        daily_counts[day] = daily_counts.get(day, 0) + 1
        rolling_pnl.append(item.net_pnl)
        if len(rolling_pnl) > config.rolling_window:
            rolling_pnl.pop(0)
        if config.daily_loss_limit is not None and daily_pnl[day] <= -abs(config.daily_loss_limit):
            daily_kill_days.add(day)
            kill_switch = True
        if config.rolling_loss_limit is not None and sum(rolling_pnl) <= -abs(config.rolling_loss_limit):
            kill_switch = True

    def realize_through(timestamp_ms: int) -> None:
        while active and active[0][0] <= timestamp_ms:
            _, _, item = heapq.heappop(active)
            realize(item)

    for sequence, trade in enumerate(sorted(trades, key=lambda item: item.entry_time_ms)):
        # Exits at a timestamp are observable before a new entry at that same timestamp.
        realize_through(trade.entry_time_ms)
        if kill_switch:
            simulated.append(
                SimulatedTrade(
                    trade=trade,
                    notional=0.0,
                    quantity=0.0,
                    gross_pnl=0.0,
                    net_pnl=0.0,
                    return_on_capital=0.0,
                    inventory_notional=current_inventory,
                    margin_used=current_exposure * config.initial_margin_rate,
                    rejected=True,
                    rejection_reason="kill_switch",
                )
            )
            continue
        notional = (
            trade.notional
            if trade.notional is not None
            else capped_expected_edge_notional(
                trade.predicted_edge_bps,
                base_notional=config.base_notional,
                max_notional=config.max_notional,
            )
        )
        proposed_inventory = current_inventory + trade.side * notional
        proposed_exposure = current_exposure + notional
        if config.max_inventory_notional is not None and abs(proposed_inventory) > config.max_inventory_notional:
            simulated.append(
                SimulatedTrade(
                    trade=trade,
                    notional=0.0,
                    quantity=0.0,
                    gross_pnl=0.0,
                    net_pnl=0.0,
                    return_on_capital=0.0,
                    inventory_notional=current_inventory,
                    margin_used=current_exposure * config.initial_margin_rate,
                    rejected=True,
                    rejection_reason="max_inventory_notional",
                )
            )
            continue

        quantity = notional / trade.entry_price
        gross_return = trade.side * (trade.exit_price - trade.entry_price) / trade.entry_price
        gross_pnl = gross_return * notional
        cost = 2.0 * notional * (config.taker_fee_bps + config.slippage_bps) / 10000.0
        inventory_penalty = abs(proposed_inventory) * config.inventory_penalty_bps / 10000.0
        net_pnl = gross_pnl - cost
        net_pnl -= inventory_penalty

        item = SimulatedTrade(
            trade=trade,
            notional=notional,
            quantity=quantity,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            return_on_capital=net_pnl / config.capital,
            inventory_notional=proposed_inventory,
            margin_used=proposed_exposure * config.initial_margin_rate,
        )
        simulated.append(item)
        current_inventory = proposed_inventory
        current_exposure = proposed_exposure
        update_peaks()
        heapq.heappush(active, (trade.exit_time_ms, sequence, item))

    while active:
        _, _, item = heapq.heappop(active)
        realize(item)

    daily_risk = [
        DailyRisk(
            day=day,
            pnl=pnl,
            return_on_capital=pnl / config.capital,
            trade_count=daily_counts[day],
            kill_switch_triggered=day in daily_kill_days,
        )
        for day, pnl in sorted(daily_pnl.items())
    ]
    daily_returns = [item.return_on_capital for item in daily_risk]
    total_net = sum(item.net_pnl for item in simulated)
    max_margin_used = max((item.margin_used for item in simulated), default=0.0)
    max_concurrency = _max_concurrency([item.trade for item in simulated if not item.rejected])
    max_drawdown = _max_drawdown([item.pnl for item in daily_risk])
    return PortfolioResult(
        trades=simulated,
        daily_risk=daily_risk,
        total_net_pnl=total_net,
        return_on_capital=total_net / config.capital,
        max_leverage=max_exposure / config.capital,
        max_exposure=max_exposure,
        max_margin_used=max_margin_used,
        max_concurrency=max_concurrency,
        max_inventory=max_inventory,
        daily_sharpe=_sharpe(daily_returns),
        daily_autocorrelation=autocorrelation(daily_returns),
        calmar_like=(total_net / config.capital) / abs(max_drawdown) if max_drawdown else 0.0,
        kill_switch_triggered=kill_switch,
    )


def capped_expected_edge_notional(
    edge_bps: float,
    *,
    base_notional: float,
    max_notional: float,
    full_size_edge_bps: float = 1.0,
) -> float:
    if base_notional <= 0.0 or max_notional <= 0.0:
        raise ValueError("notional limits must be positive")
    if full_size_edge_bps <= 0.0:
        raise ValueError("full_size_edge_bps must be positive")
    scale = max(0.0, min(1.0, abs(edge_bps) / full_size_edge_bps))
    return min(max_notional, base_notional * scale)


def fractional_kelly_notional(
    *,
    mean_edge: float,
    variance: float,
    capital: float,
    fraction: float = 0.25,
    cap_fraction: float = 0.02,
) -> float:
    if variance <= 0.0 or capital <= 0.0:
        return 0.0
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    if cap_fraction < 0.0:
        raise ValueError("cap_fraction must be non-negative")
    kelly_fraction = max(0.0, mean_edge / variance) * fraction
    return min(capital * cap_fraction, capital * kelly_fraction)


def gated_fractional_kelly_notional(
    *,
    mean_edge: float,
    capital: float,
    variance_report: VarianceStabilityReport,
    fraction: float = 0.25,
    cap_fraction: float = 0.02,
) -> float:
    if not variance_report.passed:
        return 0.0
    return fractional_kelly_notional(
        mean_edge=mean_edge,
        variance=variance_report.variance_mean,
        capital=capital,
        fraction=fraction,
        cap_fraction=cap_fraction,
    )


def evaluate_oos_variance_stability(
    path: Path | str,
    *,
    column: str = "validation_net_pnl",
    min_observations: int = 20,
    window_size: int = 5,
    max_variance_cv: float = 0.5,
) -> VarianceStabilityReport:
    if min_observations <= 1:
        raise ValueError("min_observations must be greater than 1")
    if window_size <= 1:
        raise ValueError("window_size must be greater than 1")
    if max_variance_cv < 0.0:
        raise ValueError("max_variance_cv must be non-negative")
    path = Path(path)
    values = _read_numeric_column(path, column)
    windows = [
        values[index : index + window_size]
        for index in range(0, len(values), window_size)
        if len(values[index : index + window_size]) == window_size
    ]
    window_variances = tuple(_sample_variance(window) for window in windows)
    variance_mean = sum(window_variances) / len(window_variances) if window_variances else 0.0
    variance_std = _sample_std(window_variances)
    variance_cv = variance_std / variance_mean if variance_mean > 0.0 else math.inf
    return VarianceStabilityReport(
        path=path,
        column=column,
        observations=len(values),
        window_size=window_size,
        window_variances=window_variances,
        variance_mean=variance_mean,
        variance_std=variance_std,
        variance_cv=variance_cv,
        min_observations=min_observations,
        max_variance_cv=max_variance_cv,
    )


def format_variance_stability_report(report: VarianceStabilityReport, *, output_format: str = "text") -> str:
    if output_format == "csv":
        fields = [
            "path",
            "column",
            "observations",
            "window_size",
            "window_count",
            "variance_mean",
            "variance_std",
            "variance_cv",
            "min_observations",
            "max_variance_cv",
            "passed",
        ]
        values = [
            str(report.path),
            report.column,
            str(report.observations),
            str(report.window_size),
            str(len(report.window_variances)),
            f"{report.variance_mean:.12g}",
            f"{report.variance_std:.12g}",
            f"{report.variance_cv:.12g}",
            str(report.min_observations),
            f"{report.max_variance_cv:.12g}",
            str(int(report.passed)),
        ]
        return ",".join(fields) + "\n" + ",".join(values)
    if output_format != "text":
        raise ValueError("output_format must be text or csv")
    return "\n".join(
        [
            f"path={report.path}",
            f"column={report.column}",
            f"observations={report.observations}",
            f"window_size={report.window_size}",
            f"window_count={len(report.window_variances)}",
            f"variance_mean={report.variance_mean:.12g}",
            f"variance_std={report.variance_std:.12g}",
            f"variance_cv={report.variance_cv:.12g}",
            f"min_observations={report.min_observations}",
            f"max_variance_cv={report.max_variance_cv:.12g}",
            f"passed={int(report.passed)}",
        ]
    )


def _validate_trade(trade: Trade) -> None:
    if trade.side not in {-1, 1}:
        raise ValueError("trade side must be -1 or 1")
    if trade.exit_time_ms <= trade.entry_time_ms:
        raise ValueError("trade exit_time_ms must be greater than entry_time_ms")
    if (
        not math.isfinite(trade.entry_price)
        or not math.isfinite(trade.exit_price)
        or trade.entry_price <= 0.0
        or trade.exit_price <= 0.0
    ):
        raise ValueError("trade prices must be positive and finite")
    if not math.isfinite(trade.predicted_edge_bps):
        raise ValueError("trade predicted_edge_bps must be finite")
    if trade.notional is not None and (not math.isfinite(trade.notional) or trade.notional < 0.0):
        raise ValueError("trade notional must be non-negative and finite")


def _day(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).date().isoformat()


def _max_concurrency(trades: list[Trade]) -> int:
    events: list[tuple[int, int]] = []
    for trade in trades:
        events.append((trade.entry_time_ms, 1))
        events.append((trade.exit_time_ms, -1))
    active = 0
    maximum = 0
    for _, delta in sorted(events):
        active += delta
        maximum = max(maximum, active)
    return maximum


def _sharpe(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    std = math.sqrt(max(0.0, variance))
    return mean / std * math.sqrt(252.0) if std else 0.0


def _max_drawdown(values: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    return max_dd


def _read_numeric_column(path: Path, column: str) -> list[float]:
    values: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or column not in reader.fieldnames:
            raise ValueError(f"column {column!r} not found in {path}")
        for row in reader:
            if "fold" in row and not _is_numeric(row.get("fold", "")):
                continue
            raw = row.get(column, "")
            if raw in {"", None}:
                continue
            values.append(float(raw))
    if not values:
        raise ValueError(f"column {column!r} has no numeric values in {path}")
    return values


def _is_numeric(value: str | None) -> bool:
    if value is None:
        return False
    if value == "":
        return False
    try:
        float(value)
    except ValueError:
        return False
    return True


def _sample_variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _sample_std(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return 0.0
    return math.sqrt(max(0.0, _sample_variance(list(values))))
