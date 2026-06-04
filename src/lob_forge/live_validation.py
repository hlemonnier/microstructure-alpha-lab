from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from lob_forge.execution_sim import (
    FillValidation,
    LatencyAssumptions,
    MarketEvent,
    OrderConstraints,
    QueueAssumptions,
    simulate_passive_limit_order,
    simulate_taker_latency_order,
    validate_simulated_vs_live_fills,
)


SHADOW_DECISION_COLUMNS = [
    "decision_id",
    "timestamp_ms",
    "venue",
    "symbol",
    "model_name",
    "predicted_side",
    "predicted_edge_bps",
    "order_type",
    "intended_price",
    "intended_size",
    "observed_fill_price",
    "observed_fill_size",
    "realized_pnl",
    "notes",
]

SIMULATED_FILL_COLUMNS = [
    "decision_id",
    "simulated_fill_price",
    "simulated_fill_size",
]

OBSERVED_DECISION_ID_COLUMNS = (
    "decision_id",
    "client_order_id",
    "clientOrderId",
    "order_link_id",
    "orderLinkId",
    "clOrdId",
)
OBSERVED_FILL_PRICE_COLUMNS = (
    "observed_fill_price",
    "fill_price",
    "avg_fill_price",
    "average_price",
    "avg_price",
    "avgPrice",
    "fillPrice",
    "executed_price",
    "price",
)
OBSERVED_FILL_SIZE_COLUMNS = (
    "observed_fill_size",
    "fill_size",
    "filled_size",
    "executed_qty",
    "executedQty",
    "cum_exec_qty",
    "cumExecQty",
    "filled_qty",
    "filledQty",
    "quantity",
    "qty",
    "size",
)
OBSERVED_REALIZED_PNL_COLUMNS = (
    "realized_pnl",
    "realizedPnl",
    "closed_pnl",
    "closedPnl",
    "pnl",
)
OBSERVED_FILL_TEMPLATE_COLUMNS = [
    "decision_id",
    "client_order_id",
    "venue",
    "symbol",
    "predicted_side",
    "intended_price",
    "intended_size",
    "avgPrice",
    "cumExecQty",
    "realizedPnl",
    "notes",
]

MARKET_EVENT_COLUMNS = [
    "timestamp_ms",
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "trade_side",
    "trade_size",
    "top_imbalance",
    "volatility_bps",
    "trade_intensity",
]


@dataclass(frozen=True)
class ShadowDecision:
    decision_id: str
    timestamp_ms: int
    venue: str
    symbol: str
    model_name: str
    predicted_side: int
    predicted_edge_bps: float
    order_type: str
    intended_price: float
    intended_size: float
    observed_fill_price: float | None = None
    observed_fill_size: float | None = None
    realized_pnl: float | None = None
    notes: str = ""


@dataclass(frozen=True)
class SimulatedFillPrediction:
    decision_id: str
    simulated_fill_price: float | None
    simulated_fill_size: float


@dataclass(frozen=True)
class ShadowFillValidationReport:
    matched_observations: int
    missing_simulated_decisions: tuple[str, ...]
    missing_shadow_decisions: tuple[str, ...]
    validation: FillValidation
    max_price_error: float | None = None
    max_size_error: float | None = None
    max_fill_rate_error: float | None = None

    @property
    def passed(self) -> bool:
        if self.missing_simulated_decisions or self.missing_shadow_decisions:
            return False
        checks = [
            (self.max_price_error, self.validation.mean_abs_price_error),
            (self.max_size_error, self.validation.mean_abs_size_error),
            (self.max_fill_rate_error, self.validation.fill_rate_error),
        ]
        return all(limit is None or value <= limit for limit, value in checks)


@dataclass(frozen=True)
class ObservedFillMergeReport:
    output_path: Path
    shadow_rows: int
    observed_rows: int
    matched_decisions: int
    updated_decisions: int
    unmatched_observed_ids: tuple[str, ...]


@dataclass
class _ObservedFillAggregate:
    rows: int = 0
    total_size: float = 0.0
    price_size: float = 0.0
    realized_pnl: float = 0.0
    has_realized_pnl: bool = False
    has_observation: bool = False

    @property
    def observed_fill_size(self) -> float:
        return self.total_size

    @property
    def observed_fill_price(self) -> float | None:
        if self.total_size <= 0:
            return None
        return self.price_size / self.total_size

    @property
    def observed_realized_pnl(self) -> float | None:
        return self.realized_pnl if self.has_realized_pnl else None


def append_shadow_decision(path: Path | str, decision: ShadowDecision) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHADOW_DECISION_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: asdict(decision).get(column) for column in SHADOW_DECISION_COLUMNS})
    return path


def write_shadow_decisions(path: Path | str, decisions: list[ShadowDecision]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHADOW_DECISION_COLUMNS)
        writer.writeheader()
        for decision in decisions:
            writer.writerow({column: asdict(decision).get(column) for column in SHADOW_DECISION_COLUMNS})
    return path


def merge_observed_fills_into_shadow_decisions(
    *,
    shadow_path: Path | str,
    observed_path: Path | str,
    output_path: Path | str,
) -> ObservedFillMergeReport:
    decisions = read_shadow_decisions(shadow_path)
    observed = _read_observed_fill_aggregates(observed_path)
    matched_ids: set[str] = set()
    updated: list[ShadowDecision] = []

    for decision in decisions:
        aggregate = observed.get(decision.decision_id)
        if aggregate is None:
            updated.append(decision)
            continue
        matched_ids.add(decision.decision_id)
        updated.append(
            replace(
                decision,
                observed_fill_price=aggregate.observed_fill_price,
                observed_fill_size=aggregate.observed_fill_size,
                realized_pnl=aggregate.observed_realized_pnl
                if aggregate.observed_realized_pnl is not None
                else decision.realized_pnl,
            )
        )

    output = write_shadow_decisions(output_path, updated)
    return ObservedFillMergeReport(
        output_path=Path(output),
        shadow_rows=len(decisions),
        observed_rows=sum(aggregate.rows for aggregate in observed.values()),
        matched_decisions=len(matched_ids),
        updated_decisions=len(matched_ids),
        unmatched_observed_ids=tuple(sorted(set(observed).difference(matched_ids))),
    )


def write_observed_fill_template(
    *,
    shadow_path: Path | str,
    output_path: Path | str,
    limit: int = 0,
) -> Path:
    if limit < 0:
        raise ValueError("limit must be non-negative")
    decisions = read_shadow_decisions(shadow_path)
    if limit:
        decisions = decisions[:limit]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OBSERVED_FILL_TEMPLATE_COLUMNS)
        writer.writeheader()
        for decision in decisions:
            writer.writerow(
                {
                    "decision_id": decision.decision_id,
                    "client_order_id": decision.decision_id,
                    "venue": decision.venue,
                    "symbol": decision.symbol,
                    "predicted_side": decision.predicted_side,
                    "intended_price": f"{decision.intended_price:.12g}",
                    "intended_size": f"{decision.intended_size:.12g}",
                    "avgPrice": "",
                    "cumExecQty": "",
                    "realizedPnl": "",
                    "notes": decision.notes,
                }
            )
    return output


def write_simulated_fill_predictions(path: Path | str, predictions: list[SimulatedFillPrediction]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SIMULATED_FILL_COLUMNS)
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(asdict(prediction))
    return path


def write_market_events(path: Path | str, events: list[MarketEvent]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MARKET_EVENT_COLUMNS)
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "timestamp_ms": event.timestamp_ms,
                    "bid": event.bid,
                    "ask": event.ask,
                    "bid_size": event.bid_size,
                    "ask_size": event.ask_size,
                    "trade_side": event.trade_side or "",
                    "trade_size": event.trade_size,
                    "top_imbalance": event.top_imbalance,
                    "volatility_bps": event.volatility_bps,
                    "trade_intensity": event.trade_intensity,
                }
            )
    return path


def read_market_events(path: Path | str) -> list[MarketEvent]:
    with Path(path).open(newline="") as handle:
        return [_market_event_from_row(row) for row in csv.DictReader(handle)]


def market_events_from_feature_csv(path: Path | str) -> list[MarketEvent]:
    with Path(path).open(newline="") as handle:
        return [_market_event_from_feature_row(row) for row in csv.DictReader(handle)]


def write_market_events_from_feature_csv(feature_path: Path | str, output_path: Path | str) -> Path:
    return write_market_events(output_path, market_events_from_feature_csv(feature_path))


def simulate_shadow_fill_predictions(
    *,
    shadow_path: Path | str,
    market_events_path: Path | str,
    mode: str,
    constraints: OrderConstraints,
    latency: LatencyAssumptions | None = None,
    queue: QueueAssumptions | None = None,
) -> list[SimulatedFillPrediction]:
    if mode not in {"taker", "passive"}:
        raise ValueError("mode must be taker or passive")
    decisions = read_shadow_decisions(shadow_path)
    events = sorted(read_market_events(market_events_path), key=lambda event: event.timestamp_ms)
    predictions: list[SimulatedFillPrediction] = []
    for decision in decisions:
        decision_events = [event for event in events if event.timestamp_ms >= decision.timestamp_ms]
        if mode == "taker":
            fill = simulate_taker_latency_order(
                decision_events,
                decision_time_ms=decision.timestamp_ms,
                side=decision.predicted_side,
                quantity=decision.intended_size,
                constraints=constraints,
                latency=latency or LatencyAssumptions(),
            )
            predictions.append(
                SimulatedFillPrediction(
                    decision_id=decision.decision_id,
                    simulated_fill_price=fill.fill_price,
                    simulated_fill_size=fill.quantity if fill.accepted else 0.0,
                )
            )
        else:
            fill = simulate_passive_limit_order(
                decision_events,
                side=decision.predicted_side,
                price=decision.intended_price,
                quantity=decision.intended_size,
                constraints=constraints,
                queue=queue or QueueAssumptions(queue_ahead_size=0.0),
            )
            predictions.append(
                SimulatedFillPrediction(
                    decision_id=decision.decision_id,
                    simulated_fill_price=fill.avg_fill_price,
                    simulated_fill_size=fill.filled_size,
                )
            )
    return predictions


def simulate_shadow_fills_to_file(
    *,
    shadow_path: Path | str,
    market_events_path: Path | str,
    output_path: Path | str,
    mode: str,
    constraints: OrderConstraints,
    latency: LatencyAssumptions | None = None,
    queue: QueueAssumptions | None = None,
) -> Path:
    predictions = simulate_shadow_fill_predictions(
        shadow_path=shadow_path,
        market_events_path=market_events_path,
        mode=mode,
        constraints=constraints,
        latency=latency,
        queue=queue,
    )
    return write_simulated_fill_predictions(output_path, predictions)


def read_simulated_fill_predictions(path: Path | str) -> list[SimulatedFillPrediction]:
    with Path(path).open(newline="") as handle:
        return [_simulated_prediction_from_row(row) for row in csv.DictReader(handle)]


def read_shadow_decisions(path: Path | str) -> list[ShadowDecision]:
    with Path(path).open(newline="") as handle:
        return [_decision_from_row(row) for row in csv.DictReader(handle)]


def validate_shadow_fill_predictions(
    *,
    simulated_path: Path | str,
    shadow_path: Path | str,
    max_price_error: float | None = None,
    max_size_error: float | None = None,
    max_fill_rate_error: float | None = None,
) -> ShadowFillValidationReport:
    predictions = _index_by_decision_id(read_simulated_fill_predictions(simulated_path), label="simulated")
    shadows = _index_by_decision_id(read_shadow_decisions(shadow_path), label="shadow")
    matched_ids = sorted(set(predictions).intersection(shadows))
    if not matched_ids:
        raise ValueError("no matching decision_id values between simulated and shadow fill files")

    simulated: list[tuple[float | None, float]] = []
    live: list[tuple[float | None, float]] = []
    for decision_id in matched_ids:
        prediction = predictions[decision_id]
        decision = shadows[decision_id]
        simulated.append((prediction.simulated_fill_price, prediction.simulated_fill_size))
        live.append((decision.observed_fill_price, decision.observed_fill_size or 0.0))

    validation = validate_simulated_vs_live_fills(simulated, live)
    return ShadowFillValidationReport(
        matched_observations=len(matched_ids),
        missing_simulated_decisions=tuple(sorted(set(shadows).difference(predictions))),
        missing_shadow_decisions=tuple(sorted(set(predictions).difference(shadows))),
        validation=validation,
        max_price_error=max_price_error,
        max_size_error=max_size_error,
        max_fill_rate_error=max_fill_rate_error,
    )


def format_shadow_fill_validation_report(report: ShadowFillValidationReport, *, output_format: str = "text") -> str:
    if output_format == "csv":
        fields = [
            "matched_observations",
            "missing_simulated_decisions",
            "missing_shadow_decisions",
            "mean_abs_price_error",
            "mean_abs_size_error",
            "fill_rate_error",
            "max_price_error",
            "max_size_error",
            "max_fill_rate_error",
            "passed",
        ]
        values = [
            str(report.matched_observations),
            str(len(report.missing_simulated_decisions)),
            str(len(report.missing_shadow_decisions)),
            f"{report.validation.mean_abs_price_error:.12g}",
            f"{report.validation.mean_abs_size_error:.12g}",
            f"{report.validation.fill_rate_error:.12g}",
            _format_optional(report.max_price_error),
            _format_optional(report.max_size_error),
            _format_optional(report.max_fill_rate_error),
            str(int(report.passed)),
        ]
        return ",".join(fields) + "\n" + ",".join(values)
    if output_format != "text":
        raise ValueError("output_format must be text or csv")

    lines = [
        f"matched_observations={report.matched_observations}",
        f"missing_simulated_decisions={len(report.missing_simulated_decisions)}",
        f"missing_shadow_decisions={len(report.missing_shadow_decisions)}",
        f"mean_abs_price_error={report.validation.mean_abs_price_error:.12g}",
        f"mean_abs_size_error={report.validation.mean_abs_size_error:.12g}",
        f"fill_rate_error={report.validation.fill_rate_error:.12g}",
        f"max_price_error={_format_optional(report.max_price_error)}",
        f"max_size_error={_format_optional(report.max_size_error)}",
        f"max_fill_rate_error={_format_optional(report.max_fill_rate_error)}",
        f"passed={int(report.passed)}",
    ]
    if report.missing_simulated_decisions:
        lines.append("missing_simulated=" + ",".join(report.missing_simulated_decisions[:20]))
    if report.missing_shadow_decisions:
        lines.append("missing_shadow=" + ",".join(report.missing_shadow_decisions[:20]))
    return "\n".join(lines)


def format_observed_fill_merge_report(report: ObservedFillMergeReport, *, output_format: str = "text") -> str:
    if output_format == "csv":
        fields = [
            "output_path",
            "shadow_rows",
            "observed_rows",
            "matched_decisions",
            "updated_decisions",
            "unmatched_observed_ids",
        ]
        values = [
            str(report.output_path),
            str(report.shadow_rows),
            str(report.observed_rows),
            str(report.matched_decisions),
            str(report.updated_decisions),
            str(len(report.unmatched_observed_ids)),
        ]
        return ",".join(fields) + "\n" + ",".join(values)
    if output_format != "text":
        raise ValueError("output_format must be text or csv")
    lines = [
        f"output_path={report.output_path}",
        f"shadow_rows={report.shadow_rows}",
        f"observed_rows={report.observed_rows}",
        f"matched_decisions={report.matched_decisions}",
        f"updated_decisions={report.updated_decisions}",
        f"unmatched_observed_ids={len(report.unmatched_observed_ids)}",
    ]
    if report.unmatched_observed_ids:
        lines.append("unmatched_observed=" + ",".join(report.unmatched_observed_ids[:20]))
    return "\n".join(lines)


def _read_observed_fill_aggregates(path: Path | str) -> dict[str, _ObservedFillAggregate]:
    aggregates: dict[str, _ObservedFillAggregate] = {}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("observed fill CSV has no header")
        for row_number, row in enumerate(reader, start=2):
            decision_id = _first_present(row, OBSERVED_DECISION_ID_COLUMNS)
            if not decision_id:
                raise ValueError(f"observed fill row {row_number} missing decision/client order id")
            price = _optional_float_from_aliases(row, OBSERVED_FILL_PRICE_COLUMNS)
            size = _optional_float_from_aliases(row, OBSERVED_FILL_SIZE_COLUMNS)
            realized_pnl = _optional_float_from_aliases(row, OBSERVED_REALIZED_PNL_COLUMNS)
            if price is None and size is None and realized_pnl is None:
                continue
            if size is None:
                size = 0.0
            if size < 0:
                raise ValueError(f"observed fill row {row_number} has negative fill size")
            if size > 0 and price is None:
                raise ValueError(f"observed fill row {row_number} has positive fill size but no fill price")
            aggregate = aggregates.setdefault(decision_id, _ObservedFillAggregate())
            aggregate.rows += 1
            aggregate.has_observation = True
            aggregate.total_size += size
            if price is not None and size > 0:
                aggregate.price_size += price * size
            if realized_pnl is not None:
                aggregate.realized_pnl += realized_pnl
                aggregate.has_realized_pnl = True
    return aggregates


def _decision_from_row(row: dict[str, str]) -> ShadowDecision:
    return ShadowDecision(
        decision_id=row["decision_id"],
        timestamp_ms=int(row["timestamp_ms"]),
        venue=row["venue"],
        symbol=row["symbol"],
        model_name=row["model_name"],
        predicted_side=int(row["predicted_side"]),
        predicted_edge_bps=float(row["predicted_edge_bps"]),
        order_type=row["order_type"],
        intended_price=float(row["intended_price"]),
        intended_size=float(row["intended_size"]),
        observed_fill_price=_optional_float(row["observed_fill_price"]),
        observed_fill_size=_optional_float(row["observed_fill_size"]),
        realized_pnl=_optional_float(row["realized_pnl"]),
        notes=row.get("notes", ""),
    )


def _simulated_prediction_from_row(row: dict[str, str]) -> SimulatedFillPrediction:
    return SimulatedFillPrediction(
        decision_id=row["decision_id"],
        simulated_fill_price=_optional_float(row["simulated_fill_price"]),
        simulated_fill_size=_optional_float(row["simulated_fill_size"]) or 0.0,
    )


def _market_event_from_row(row: dict[str, str]) -> MarketEvent:
    return MarketEvent(
        timestamp_ms=int(float(row["timestamp_ms"])),
        bid=float(row["bid"]),
        ask=float(row["ask"]),
        bid_size=float(row.get("bid_size", "") or 0.0),
        ask_size=float(row.get("ask_size", "") or 0.0),
        trade_side=row.get("trade_side") or None,
        trade_size=float(row.get("trade_size", "") or 0.0),
        top_imbalance=float(row.get("top_imbalance", "") or 0.0),
        volatility_bps=float(row.get("volatility_bps", "") or 0.0),
        trade_intensity=float(row.get("trade_intensity", "") or 0.0),
    )


def _market_event_from_feature_row(row: dict[str, str]) -> MarketEvent:
    trade_qty = float(row.get("trade_qty", "") or 0.0)
    trade_imbalance = float(row.get("trade_imbalance", "") or 0.0)
    trade_side = None
    if trade_qty > 0:
        trade_side = "buy" if trade_imbalance > 0 else "sell" if trade_imbalance < 0 else None
    volatility = float(row.get("realized_volatility_5", "") or 0.0) * 10000.0
    return MarketEvent(
        timestamp_ms=int(float(row.get("entry_event_time") or row["event_time"])),
        bid=float(row.get("entry_bid") or row["bid"]),
        ask=float(row.get("entry_ask") or row["ask"]),
        bid_size=float(row.get("bid_qty", "") or 0.0),
        ask_size=float(row.get("ask_qty", "") or 0.0),
        trade_side=trade_side,
        trade_size=trade_qty,
        top_imbalance=float(row.get("top_imbalance", "") or 0.0),
        volatility_bps=volatility,
        trade_intensity=float(row.get("trade_count", "") or 0.0),
    )


def _index_by_decision_id(rows, *, label: str):
    indexed = {}
    duplicates: list[str] = []
    for row in rows:
        decision_id = row.decision_id
        if decision_id in indexed:
            duplicates.append(decision_id)
            continue
        indexed[decision_id] = row
    if duplicates:
        raise ValueError(f"duplicate {label} decision_id values: {', '.join(sorted(set(duplicates)))}")
    return indexed


def _optional_float(value: str) -> float | None:
    return float(value) if value not in {"", None} else None


def _format_optional(value: float | None) -> str:
    return "" if value is None else f"{value:.12g}"


def _first_present(row: dict[str, str], columns: tuple[str, ...]) -> str | None:
    for column in columns:
        value = row.get(column)
        if value not in {"", None}:
            return str(value)
    return None


def _optional_float_from_aliases(row: dict[str, str], columns: tuple[str, ...]) -> float | None:
    value = _first_present(row, columns)
    return float(value) if value is not None else None
