from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from lob_forge.statistics import newey_west_standard_error, stationary_block_bootstrap_mean_interval


@dataclass(frozen=True)
class HypothesisSpec:
    hypothesis_id: str
    universe: str
    data_source: str
    date_range: str
    feature_set: str
    target: str
    horizon: str
    latency_model: str
    model_class: str
    decision_rule: str
    cost_model: str
    risk_rule: str
    acceptance_criterion: str
    status: str = "candidate"
    notes: str = ""


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    hypothesis_id: str
    command: str
    artifact_path: str
    data_path: str
    candidate_count: int
    git_rev: str
    created_at_utc: str
    working_tree_dirty: bool | None = None
    notes: str = ""
    holdout_manifest_path: str = ""
    holdout_manifest_sha256: str = ""


@dataclass(frozen=True)
class FoldResult:
    fold: int
    test_rows: int
    test_trades: int
    test_gross_pnl: float
    test_net_pnl: float
    test_break_even_fee_bps: float
    test_mean_net_bps: float
    test_median_net_bps: float
    test_profit_factor: float
    test_max_drawdown_pnl: float
    test_sharpe_per_trade: float
    test_win_rate: float


@dataclass(frozen=True)
class ResultAudit:
    artifact_path: str
    inference_grain: str
    fold_count: int
    total_test_rows: int
    total_test_trades: int
    total_test_gross_pnl: float
    total_test_net_pnl: float
    positive_fold_rate: float
    median_fold_net_pnl: float
    min_fold_net_pnl: float
    max_fold_net_pnl: float
    mean_fold_net_pnl: float
    fold_net_pnl_std: float
    max_positive_fold_share_of_total_net: float
    weighted_break_even_fee_bps: float
    median_fold_mean_net_bps: float
    median_fold_median_net_bps: float
    median_fold_profit_factor: float
    max_fold_drawdown_pnl: float
    mean_fold_sharpe_per_trade: float
    bootstrap_mean_net_pnl_lower_5pct: float
    bootstrap_mean_net_pnl_upper_95pct: float
    one_sided_p_value_mean_le_zero: float


@dataclass(frozen=True)
class AcceptanceCriteria:
    min_fold_count: int = 20
    min_positive_fold_rate: float = 0.70
    max_positive_fold_share_of_total_net: float = 0.40
    require_positive_total_net_pnl: bool = True
    require_positive_median_fold: bool = True
    min_break_even_fee_bps: float = 0.0
    require_positive_bootstrap_lower_bound: bool = False


@dataclass(frozen=True)
class AcceptanceVerdict:
    passed: bool
    rejection_reasons: list[str]


@dataclass(frozen=True)
class PValueRecord:
    hypothesis_id: str
    p_value: float
    metric: str = ""


@dataclass(frozen=True)
class PValueCorrection:
    hypothesis_id: str
    metric: str
    p_value: float
    bonferroni_p_value: float
    bh_adjusted_p_value: float
    bh_accept: bool


def hypothesis_to_dict(spec: HypothesisSpec) -> dict[str, str]:
    return asdict(spec)


def append_experiment_record(path: Path | str, record: ExperimentRecord) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as handle:
        handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_fold_results(path: Path | str) -> tuple[list[FoldResult], dict[str, str] | None]:
    input_path = Path(path)
    with input_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("result artifact has no CSV header")
        required = {"fold", "test_rows", "test_trades", "test_net_pnl"}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"result artifact missing required columns: {', '.join(missing)}")

        folds: list[FoldResult] = []
        summary_row: dict[str, str] | None = None
        for row in reader:
            raw_fold = (row.get("fold") or "").strip()
            if raw_fold == "summary":
                summary_row = row
                continue
            if not raw_fold:
                continue
            try:
                fold = int(raw_fold)
            except ValueError:
                continue
            folds.append(
                FoldResult(
                    fold=fold,
                    test_rows=_safe_int(row.get("test_rows")),
                    test_trades=_safe_int(row.get("test_trades")),
                    test_gross_pnl=_safe_float(row.get("test_gross_pnl")),
                    test_net_pnl=_safe_float(row.get("test_net_pnl")),
                    test_break_even_fee_bps=_safe_float(row.get("test_break_even_fee_bps")),
                    test_mean_net_bps=_safe_float(row.get("test_mean_net_bps")),
                    test_median_net_bps=_safe_float(row.get("test_median_net_bps")),
                    test_profit_factor=_safe_float(row.get("test_profit_factor")),
                    test_max_drawdown_pnl=_safe_float(row.get("test_max_drawdown_pnl")),
                    test_sharpe_per_trade=_safe_float(row.get("test_sharpe_per_trade")),
                    test_win_rate=_safe_float(row.get("test_win_rate")),
                )
            )

    if not folds:
        raise ValueError("result artifact contains no fold rows")
    return folds, summary_row


def audit_result_artifact(
    path: Path | str,
    *,
    bootstrap_samples: int = 2000,
    seed: int = 7,
) -> ResultAudit:
    folds, summary_row = read_fold_results(path)
    net_pnls = [fold.test_net_pnl for fold in folds]
    total_test_rows = sum(fold.test_rows for fold in folds)
    total_test_trades = sum(fold.test_trades for fold in folds)
    total_test_gross_pnl = sum(fold.test_gross_pnl for fold in folds)
    total_test_net_pnl = sum(net_pnls)
    positive_fold_count = sum(1 for value in net_pnls if value > 0.0)
    mean_fold_net_pnl = sum(net_pnls) / len(net_pnls)
    fold_net_pnl_std = _sample_std(net_pnls)
    interval = stationary_block_bootstrap_mean_interval(
        net_pnls,
        expected_block_size=max(1, int(math.sqrt(len(net_pnls)))),
        samples=bootstrap_samples,
        confidence=0.90,
        seed=seed,
    )
    lower, upper = interval.lower, interval.upper
    summary_break_even = _safe_float(summary_row.get("test_break_even_fee_bps")) if summary_row else 0.0
    if not summary_break_even:
        summary_break_even = _trade_weighted_break_even(folds)

    return ResultAudit(
        artifact_path=str(path),
        inference_grain="fold_summary",
        fold_count=len(folds),
        total_test_rows=total_test_rows,
        total_test_trades=total_test_trades,
        total_test_gross_pnl=total_test_gross_pnl,
        total_test_net_pnl=total_test_net_pnl,
        positive_fold_rate=positive_fold_count / len(folds),
        median_fold_net_pnl=_median(net_pnls),
        min_fold_net_pnl=min(net_pnls),
        max_fold_net_pnl=max(net_pnls),
        mean_fold_net_pnl=mean_fold_net_pnl,
        fold_net_pnl_std=fold_net_pnl_std,
        max_positive_fold_share_of_total_net=_max_positive_share(net_pnls),
        weighted_break_even_fee_bps=summary_break_even,
        median_fold_mean_net_bps=_median([fold.test_mean_net_bps for fold in folds]),
        median_fold_median_net_bps=_median([fold.test_median_net_bps for fold in folds]),
        median_fold_profit_factor=_median([fold.test_profit_factor for fold in folds]),
        max_fold_drawdown_pnl=max(fold.test_max_drawdown_pnl for fold in folds),
        mean_fold_sharpe_per_trade=sum(fold.test_sharpe_per_trade for fold in folds) / len(folds),
        bootstrap_mean_net_pnl_lower_5pct=lower,
        bootstrap_mean_net_pnl_upper_95pct=upper,
        one_sided_p_value_mean_le_zero=one_sided_hac_p_value_mean_le_zero(net_pnls),
    )


def evaluate_acceptance(audit: ResultAudit, criteria: AcceptanceCriteria) -> AcceptanceVerdict:
    reasons: list[str] = []
    if audit.fold_count < criteria.min_fold_count:
        reasons.append(f"fold count {audit.fold_count} < required {criteria.min_fold_count}")
    if criteria.require_positive_total_net_pnl and audit.total_test_net_pnl <= 0.0:
        reasons.append("total OOS net PnL is not positive")
    if criteria.require_positive_median_fold and audit.median_fold_net_pnl <= 0.0:
        reasons.append("median fold net PnL is not positive")
    if audit.positive_fold_rate < criteria.min_positive_fold_rate:
        reasons.append(f"positive fold rate {audit.positive_fold_rate:.3f} < {criteria.min_positive_fold_rate:.3f}")
    if audit.max_positive_fold_share_of_total_net > criteria.max_positive_fold_share_of_total_net:
        reasons.append(
            "largest positive fold share "
            f"{audit.max_positive_fold_share_of_total_net:.3f} "
            f"> {criteria.max_positive_fold_share_of_total_net:.3f}"
        )
    if audit.weighted_break_even_fee_bps < criteria.min_break_even_fee_bps:
        reasons.append(
            f"break-even fee {audit.weighted_break_even_fee_bps:.6f} bps "
            f"< required {criteria.min_break_even_fee_bps:.6f} bps"
        )
    if criteria.require_positive_bootstrap_lower_bound and audit.bootstrap_mean_net_pnl_lower_5pct <= 0.0:
        reasons.append("bootstrap lower 5% bound for mean fold net PnL is not positive")
    return AcceptanceVerdict(passed=not reasons, rejection_reasons=reasons)


def format_result_audit_csv(audit: ResultAudit, verdict: AcceptanceVerdict) -> str:
    fields = [
        "artifact_path",
        "inference_grain",
        "fold_count",
        "total_test_rows",
        "total_test_trades",
        "total_test_gross_pnl",
        "total_test_net_pnl",
        "positive_fold_rate",
        "median_fold_net_pnl",
        "min_fold_net_pnl",
        "max_fold_net_pnl",
        "mean_fold_net_pnl",
        "fold_net_pnl_std",
        "max_positive_fold_share_of_total_net",
        "weighted_break_even_fee_bps",
        "median_fold_mean_net_bps",
        "median_fold_median_net_bps",
        "median_fold_profit_factor",
        "max_fold_drawdown_pnl",
        "mean_fold_sharpe_per_trade",
        "bootstrap_mean_net_pnl_lower_5pct",
        "bootstrap_mean_net_pnl_upper_95pct",
        "one_sided_p_value_mean_le_zero",
        "acceptance_passed",
        "rejection_reasons",
    ]
    values = [
        audit.artifact_path,
        audit.inference_grain,
        str(audit.fold_count),
        str(audit.total_test_rows),
        str(audit.total_test_trades),
        _fmt(audit.total_test_gross_pnl),
        _fmt(audit.total_test_net_pnl),
        _fmt(audit.positive_fold_rate),
        _fmt(audit.median_fold_net_pnl),
        _fmt(audit.min_fold_net_pnl),
        _fmt(audit.max_fold_net_pnl),
        _fmt(audit.mean_fold_net_pnl),
        _fmt(audit.fold_net_pnl_std),
        _fmt(audit.max_positive_fold_share_of_total_net),
        _fmt(audit.weighted_break_even_fee_bps),
        _fmt(audit.median_fold_mean_net_bps),
        _fmt(audit.median_fold_median_net_bps),
        _fmt(audit.median_fold_profit_factor),
        _fmt(audit.max_fold_drawdown_pnl),
        _fmt(audit.mean_fold_sharpe_per_trade),
        _fmt(audit.bootstrap_mean_net_pnl_lower_5pct),
        _fmt(audit.bootstrap_mean_net_pnl_upper_95pct),
        _fmt(audit.one_sided_p_value_mean_le_zero),
        str(int(verdict.passed)),
        "; ".join(verdict.rejection_reasons),
    ]
    return _csv_line(fields) + "\n" + _csv_line(values)


def format_result_audit_markdown(audit: ResultAudit, verdict: AcceptanceVerdict) -> str:
    reasons = verdict.rejection_reasons or ["passed all configured acceptance gates"]
    lines = [
        f"# Result Audit: {Path(audit.artifact_path).name}",
        "",
        f"- Verdict: {'PASS' if verdict.passed else 'REJECT'}",
        f"- Inference grain: {audit.inference_grain}",
        f"- Folds: {audit.fold_count}",
        f"- Total test rows: {audit.total_test_rows}",
        f"- Total test trades: {audit.total_test_trades}",
        f"- Total test net PnL: {_fmt(audit.total_test_net_pnl)}",
        f"- Positive fold rate: {_fmt(audit.positive_fold_rate)}",
        f"- Median fold net PnL: {_fmt(audit.median_fold_net_pnl)}",
        f"- Max positive fold share of total net PnL: {_fmt(audit.max_positive_fold_share_of_total_net)}",
        f"- Weighted break-even fee: {_fmt(audit.weighted_break_even_fee_bps)} bps",
        f"- Median fold mean net return: {_fmt(audit.median_fold_mean_net_bps)} bps",
        f"- Median fold median net return: {_fmt(audit.median_fold_median_net_bps)} bps",
        f"- Median fold profit factor: {_fmt(audit.median_fold_profit_factor)}",
        f"- Max fold drawdown: {_fmt(audit.max_fold_drawdown_pnl)} raw PnL units",
        f"- Mean fold Sharpe per trade: {_fmt(audit.mean_fold_sharpe_per_trade)}",
        f"- Fold-bootstrap mean net PnL 90% central interval (5/95%): {_fmt(audit.bootstrap_mean_net_pnl_lower_5pct)} / {_fmt(audit.bootstrap_mean_net_pnl_upper_95pct)}",
        f"- One-sided HAC/Newey-West z p-value for mean <= 0: {_fmt(audit.one_sided_p_value_mean_le_zero)}",
        "",
        "## Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in reasons)
    return "\n".join(lines)


def bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int = 2000,
    seed: int = 7,
    lower_pct: float = 5.0,
    upper_pct: float = 95.0,
) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot bootstrap empty values")
    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(draw) / len(draw))
    means.sort()
    return _percentile_sorted(means, lower_pct), _percentile_sorted(means, upper_pct)


def one_sided_normal_p_value_mean_le_zero(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot score empty values")
    mean_value = sum(values) / len(values)
    std_value = _sample_std(values)
    if std_value == 0.0:
        return 0.0 if mean_value > 0.0 else 1.0
    z_score = mean_value / (std_value / math.sqrt(len(values)))
    return 0.5 * math.erfc(z_score / math.sqrt(2.0))


def one_sided_hac_p_value_mean_le_zero(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot score empty values")
    mean_value = sum(values) / len(values)
    se = newey_west_standard_error(values)
    if se == 0.0:
        return 0.0 if mean_value > 0.0 else 1.0
    z_score = mean_value / se
    return 0.5 * math.erfc(z_score / math.sqrt(2.0))


def bonferroni_adjust(p_values: list[float]) -> list[float]:
    k = len(p_values)
    return [min(1.0, max(0.0, value) * k) for value in p_values]


def benjamini_hochberg_adjust(p_values: list[float]) -> list[float]:
    k = len(p_values)
    if k == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0 for _ in p_values]
    running_min = 1.0
    for rank_from_end, (original_idx, p_value) in enumerate(reversed(indexed), start=1):
        rank = k - rank_from_end + 1
        adjusted_value = min(running_min, max(0.0, p_value) * k / rank)
        running_min = adjusted_value
        adjusted[original_idx] = min(1.0, adjusted_value)
    return adjusted


def correct_p_values(records: list[PValueRecord], *, q: float = 0.05) -> list[PValueCorrection]:
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    raw = [record.p_value for record in records]
    bonferroni = bonferroni_adjust(raw)
    bh = benjamini_hochberg_adjust(raw)
    return [
        PValueCorrection(
            hypothesis_id=record.hypothesis_id,
            metric=record.metric,
            p_value=record.p_value,
            bonferroni_p_value=bonferroni[idx],
            bh_adjusted_p_value=bh[idx],
            bh_accept=bh[idx] <= q,
        )
        for idx, record in enumerate(records)
    ]


def read_p_value_records(
    path: Path | str,
    *,
    id_column: str = "hypothesis_id",
    p_value_column: str = "p_value",
    metric_column: str = "metric",
) -> list[PValueRecord]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("p-value CSV has no header")
        missing = [column for column in [id_column, p_value_column] if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"p-value CSV missing required columns: {', '.join(missing)}")
        return [
            PValueRecord(
                hypothesis_id=row[id_column],
                p_value=float(row[p_value_column]),
                metric=row.get(metric_column, ""),
            )
            for row in reader
        ]


def format_p_value_corrections(corrections: list[PValueCorrection]) -> str:
    lines = ["hypothesis_id,metric,p_value,bonferroni_p_value,bh_adjusted_p_value,bh_accept"]
    for correction in corrections:
        lines.append(
            _csv_line(
                [
                    correction.hypothesis_id,
                    correction.metric,
                    _fmt(correction.p_value),
                    _fmt(correction.bonferroni_p_value),
                    _fmt(correction.bh_adjusted_p_value),
                    str(int(correction.bh_accept)),
                ]
            )
        )
    return "\n".join(lines)


def _max_positive_share(values: list[float]) -> float:
    total = sum(values)
    positives = [value for value in values if value > 0.0]
    if total <= 0.0 or not positives:
        return 0.0
    return max(positives) / total


def _trade_weighted_break_even(folds: list[FoldResult]) -> float:
    total_trades = sum(fold.test_trades for fold in folds)
    if not total_trades:
        return 0.0
    return sum(fold.test_break_even_fee_bps * fold.test_trades for fold in folds) / total_trades


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _percentile_sorted(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    if percentile <= 0:
        return values[0]
    if percentile >= 100:
        return values[-1]
    position = (len(values) - 1) * percentile / 100.0
    lower_idx = int(math.floor(position))
    upper_idx = int(math.ceil(position))
    if lower_idx == upper_idx:
        return values[lower_idx]
    lower = values[lower_idx]
    upper = values[upper_idx]
    weight = position - lower_idx
    return lower + (upper - lower) * weight


def _safe_int(raw: str | None) -> int:
    if raw is None or raw == "":
        return 0
    return int(float(raw))


def _safe_float(raw: str | None) -> float:
    if raw is None or raw == "":
        return 0.0
    return float(raw)


def _csv_line(values: list[str]) -> str:
    output = []
    for value in values:
        if any(char in value for char in [",", '"', "\n"]):
            output.append('"' + value.replace('"', '""') + '"')
        else:
            output.append(value)
    return ",".join(output)


def _fmt(value: float) -> str:
    if math.isnan(value):
        return "0"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.6f}"
