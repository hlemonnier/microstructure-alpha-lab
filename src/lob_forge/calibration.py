from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationBin:
    bucket: int
    lower_bound: float
    upper_bound: float
    count: int
    mean_confidence: float
    empirical_accuracy: float
    calibration_gap: float


@dataclass(frozen=True)
class EdgeReliabilityBin:
    decile: int
    lower_edge: float
    upper_edge: float
    count: int
    mean_predicted_edge: float
    mean_realized_pnl: float
    hit_rate: float


@dataclass(frozen=True)
class PosteriorScore:
    n: int
    sample_mean: float
    sample_std: float
    standard_error: float
    p_mean_gt_zero: float
    p_mean_gt_margin: float
    posterior_sharpe: float


def multiclass_brier_score(
    labels: list[int],
    probabilities: list[dict[int, float]],
    *,
    classes: list[int] | None = None,
) -> float:
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have the same length")
    if not labels:
        raise ValueError("need at least one prediction")
    class_values = classes or sorted(
        {label for label in labels} | {klass for probs in probabilities for klass in probs}
    )
    total = 0.0
    for label, probs in zip(labels, probabilities):
        for klass in class_values:
            expected = 1.0 if label == klass else 0.0
            total += (float(probs.get(klass, 0.0)) - expected) ** 2
    return total / len(labels)


def expected_calibration_error(
    labels: list[int],
    probabilities: list[dict[int, float]],
    *,
    bins: int = 10,
) -> tuple[float, list[CalibrationBin]]:
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have the same length")
    if not labels:
        raise ValueError("need at least one prediction")
    if bins <= 0:
        raise ValueError("bins must be positive")

    bucket_rows: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for label, probs in zip(labels, probabilities):
        predicted_label, confidence = _top_probability(probs)
        bucket = min(bins - 1, int(confidence * bins))
        bucket_rows[bucket].append((confidence, predicted_label == label))

    output_bins: list[CalibrationBin] = []
    ece = 0.0
    total = len(labels)
    for bucket, rows in enumerate(bucket_rows):
        lower = bucket / bins
        upper = (bucket + 1) / bins
        if not rows:
            output_bins.append(CalibrationBin(bucket, lower, upper, 0, 0.0, 0.0, 0.0))
            continue
        mean_confidence = sum(confidence for confidence, _ in rows) / len(rows)
        empirical_accuracy = sum(1.0 for _, correct in rows if correct) / len(rows)
        gap = abs(mean_confidence - empirical_accuracy)
        ece += (len(rows) / total) * gap
        output_bins.append(
            CalibrationBin(
                bucket=bucket,
                lower_bound=lower,
                upper_bound=upper,
                count=len(rows),
                mean_confidence=mean_confidence,
                empirical_accuracy=empirical_accuracy,
                calibration_gap=gap,
            )
        )
    return ece, output_bins


def edge_reliability_by_decile(
    predicted_edges: list[float],
    realized_pnls: list[float],
    *,
    deciles: int = 10,
) -> list[EdgeReliabilityBin]:
    if len(predicted_edges) != len(realized_pnls):
        raise ValueError("predicted_edges and realized_pnls must have the same length")
    if not predicted_edges:
        raise ValueError("need at least one edge observation")
    if deciles <= 0:
        raise ValueError("deciles must be positive")

    pairs = sorted(zip(predicted_edges, realized_pnls), key=lambda pair: pair[0])
    output: list[EdgeReliabilityBin] = []
    for bucket in range(deciles):
        start = bucket * len(pairs) // deciles
        end = (bucket + 1) * len(pairs) // deciles
        rows = pairs[start:end]
        if not rows:
            continue
        edges = [edge for edge, _ in rows]
        pnls = [pnl for _, pnl in rows]
        output.append(
            EdgeReliabilityBin(
                decile=bucket + 1,
                lower_edge=edges[0],
                upper_edge=edges[-1],
                count=len(rows),
                mean_predicted_edge=sum(edges) / len(edges),
                mean_realized_pnl=sum(pnls) / len(pnls),
                hit_rate=sum(1.0 for pnl in pnls if pnl > 0.0) / len(pnls),
            )
        )
    return output


def posterior_mean_score(
    observations: list[float],
    *,
    cost_margin: float = 0.0,
    prior_mean: float = 0.0,
    prior_observations: float = 1.0,
) -> PosteriorScore:
    if not observations:
        raise ValueError("need at least one observation")
    if prior_observations < 0:
        raise ValueError("prior_observations must be non-negative")

    n = len(observations)
    sample_mean = sum(observations) / n
    sample_std = _sample_std(observations, sample_mean)
    effective_n = n + prior_observations
    posterior_mean = (sum(observations) + prior_mean * prior_observations) / effective_n
    standard_error = sample_std / math.sqrt(max(1.0, effective_n))
    if standard_error == 0.0:
        p_zero = 1.0 if posterior_mean > 0.0 else 0.0
        p_margin = 1.0 if posterior_mean > cost_margin else 0.0
    else:
        p_zero = 1.0 - _normal_cdf((0.0 - posterior_mean) / standard_error)
        p_margin = 1.0 - _normal_cdf((cost_margin - posterior_mean) / standard_error)

    return PosteriorScore(
        n=n,
        sample_mean=sample_mean,
        sample_std=sample_std,
        standard_error=standard_error,
        p_mean_gt_zero=p_zero,
        p_mean_gt_margin=p_margin,
        posterior_sharpe=posterior_mean / sample_std if sample_std else 0.0,
    )


def format_calibration_bins(bins: list[CalibrationBin]) -> str:
    lines = ["bucket,lower_bound,upper_bound,count,mean_confidence,empirical_accuracy,calibration_gap"]
    for item in bins:
        lines.append(
            ",".join(
                [
                    str(item.bucket),
                    _fmt(item.lower_bound),
                    _fmt(item.upper_bound),
                    str(item.count),
                    _fmt(item.mean_confidence),
                    _fmt(item.empirical_accuracy),
                    _fmt(item.calibration_gap),
                ]
            )
        )
    return "\n".join(lines)


def format_edge_reliability_bins(bins: list[EdgeReliabilityBin]) -> str:
    lines = "decile,lower_edge,upper_edge,count,mean_predicted_edge,mean_realized_pnl,hit_rate"
    output = [lines]
    for item in bins:
        output.append(
            ",".join(
                [
                    str(item.decile),
                    _fmt(item.lower_edge),
                    _fmt(item.upper_edge),
                    str(item.count),
                    _fmt(item.mean_predicted_edge),
                    _fmt(item.mean_realized_pnl),
                    _fmt(item.hit_rate),
                ]
            )
        )
    return "\n".join(output)


def _top_probability(probabilities: dict[int, float]) -> tuple[int, float]:
    if not probabilities:
        raise ValueError("probability dictionary cannot be empty")
    label, probability = max(probabilities.items(), key=lambda item: item[1])
    return label, float(probability)


def _sample_std(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(0.0, variance))


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _fmt(value: float) -> str:
    return f"{value:.12g}"
