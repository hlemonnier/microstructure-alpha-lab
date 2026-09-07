from __future__ import annotations

import math
from dataclasses import dataclass

from .statistics import student_t_cdf


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
    posterior_mean: float = 0.0
    posterior_degrees_of_freedom: float = 0.0
    method: str = "normal_inverse_gamma_iid"


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
    if len(set(class_values)) != len(class_values) or not set(labels) <= set(class_values):
        raise ValueError("classes must be unique and include every label")
    total = 0.0
    for label, probs in zip(labels, probabilities):
        _validate_probabilities(probs)
        if not set(probs) <= set(class_values):
            raise ValueError("probability class missing from classes")
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

    if not all(math.isfinite(x) for x in predicted_edges + realized_pnls):
        raise ValueError("edge reliability values must be finite")
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
    prior_shape: float = 2.0,
    prior_scale: float = 1.0,
) -> PosteriorScore:
    """NIG posterior for independent Gaussian observations in common units.

    Variance prior is InverseGamma(prior_shape, prior_scale); scale has
    squared observation units. Prior parameters must be chosen before looking
    at results. Group overlapping/dependent returns before using this iid model.
    standard_error is posterior SD of the mean, not a plug-in sample SE.
    posterior_sharpe is posterior mean / sqrt(E[variance]), not E[Sharpe].
    A zero prior_observations value uses the limiting improper NIG prior;
    it is not an independently specified flat-mean times IG-variance prior.
    """
    if not observations:
        raise ValueError("need at least one observation")
    if not all(math.isfinite(x) for x in [*observations, cost_margin, prior_mean,
                                        prior_observations, prior_shape, prior_scale]):
        raise ValueError("posterior inputs must be finite")
    if prior_observations < 0 or prior_shape <= 1 or prior_scale <= 0:
        raise ValueError("need prior_observations >= 0, prior_shape > 1, prior_scale > 0")
    n = len(observations)
    sample_mean = sum(observations) / n
    sample_std = _sample_std(observations, sample_mean)
    kappa = n + prior_observations
    posterior_mean = (sum(observations) + prior_mean * prior_observations) / kappa
    alpha = prior_shape + n / 2
    beta = (prior_scale + sum((x - sample_mean) ** 2 for x in observations) / 2
            + prior_observations * n * (sample_mean - prior_mean) ** 2 / (2 * kappa))
    degrees = 2 * alpha
    scale = math.sqrt(beta / (alpha * kappa))
    return PosteriorScore(
        n=n, sample_mean=sample_mean, sample_std=sample_std,
        standard_error=math.sqrt(beta / ((alpha - 1) * kappa)),
        p_mean_gt_zero=student_t_cdf(posterior_mean / scale, degrees),
        p_mean_gt_margin=student_t_cdf((posterior_mean - cost_margin) / scale, degrees),
        posterior_sharpe=posterior_mean / math.sqrt(beta / (alpha - 1)),
        posterior_mean=posterior_mean, posterior_degrees_of_freedom=degrees,
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
    _validate_probabilities(probabilities)
    label, probability = max(probabilities.items(), key=lambda item: item[1])
    return label, float(probability)


def _sample_std(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(0.0, variance))


def _validate_probabilities(probabilities: dict[int, float]) -> None:
    if not probabilities or any(not math.isfinite(p) or p < 0 or p > 1 for p in probabilities.values()):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-9):
        raise ValueError("probabilities must sum to one")


def _fmt(value: float) -> str:
    return f"{value:.12g}"
