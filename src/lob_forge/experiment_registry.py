from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentAttempt:
    run_id: str
    family: str
    model_class: str
    feature: str
    threshold: float
    selection_metric: str
    config_sha256: str = ""
    config_json: str = ""
    data_path: str = ""
    holdout_manifest_path: str = ""
    horizon_ms: int | None = None
    latency_ms: int | None = None
    taker_fee_bps: float | None = None
    maker_fee_bps: float | None = None
    random_seed: int | None = None
    status: str = "planned"
    selected: bool = False
    artifact_path: str = ""
    validation_net_pnl: float | None = None
    test_net_pnl: float | None = None
    failure_reason: str = ""
    notes: str = ""


def write_threshold_experiment_registry(
    path: Path | str,
    *,
    run_id: str,
    model_class: str,
    features: list[str],
    thresholds: list[float],
    selection_metric: str,
    selected: tuple[str, float] | None = None,
    selected_metrics: dict[str, float] | None = None,
    artifact_path: str = "",
    data_path: str = "",
    holdout_manifest_path: str = "",
    horizon_ms: int | None = None,
    latency_ms: int | None = None,
    taker_fee_bps: float | None = None,
    maker_fee_bps: float | None = None,
    random_seed: int | None = None,
    extra_config: dict[str, object] | None = None,
    notes: str = "",
) -> list[ExperimentAttempt]:
    if not run_id:
        raise ValueError("run_id cannot be empty")
    if not model_class:
        raise ValueError("model_class cannot be empty")
    if not features:
        raise ValueError("features cannot be empty")
    if not thresholds:
        raise ValueError("thresholds cannot be empty")
    attempts: list[ExperimentAttempt] = []
    for feature in features:
        for threshold in thresholds:
            is_selected = selected == (feature, threshold)
            config = {
                "family": "threshold_grid",
                "model_class": model_class,
                "feature": feature,
                "threshold": threshold,
                "selection_metric": selection_metric,
                "data_path": data_path,
                "holdout_manifest_path": holdout_manifest_path,
                "horizon_ms": horizon_ms,
                "latency_ms": latency_ms,
                "taker_fee_bps": taker_fee_bps,
                "maker_fee_bps": maker_fee_bps,
                "random_seed": random_seed,
            }
            if extra_config:
                config.update(extra_config)
            config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
            attempts.append(
                ExperimentAttempt(
                    run_id=run_id,
                    family="threshold_grid",
                    model_class=model_class,
                    feature=feature,
                    threshold=threshold,
                    selection_metric=selection_metric,
                    config_sha256=hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
                    config_json=config_json,
                    data_path=data_path,
                    holdout_manifest_path=holdout_manifest_path,
                    horizon_ms=horizon_ms,
                    latency_ms=latency_ms,
                    taker_fee_bps=taker_fee_bps,
                    maker_fee_bps=maker_fee_bps,
                    random_seed=random_seed,
                    status="selected" if is_selected else "evaluated_unselected",
                    selected=is_selected,
                    artifact_path=artifact_path,
                    validation_net_pnl=selected_metrics.get("validation_net_pnl")
                    if is_selected and selected_metrics
                    else None,
                    test_net_pnl=selected_metrics.get("test_net_pnl") if is_selected and selected_metrics else None,
                    notes=notes,
                )
            )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for attempt in attempts:
            handle.write(json.dumps(asdict(attempt), sort_keys=True) + "\n")
    return attempts


def read_experiment_registry(path: Path | str) -> list[ExperimentAttempt]:
    attempts: list[ExperimentAttempt] = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                payload.setdefault("selected", False)
                payload.setdefault("config_sha256", "")
                payload.setdefault("config_json", "")
                payload.setdefault("data_path", "")
                payload.setdefault("holdout_manifest_path", "")
                payload.setdefault("horizon_ms", None)
                payload.setdefault("latency_ms", None)
                payload.setdefault("taker_fee_bps", None)
                payload.setdefault("maker_fee_bps", None)
                payload.setdefault("random_seed", None)
                payload.setdefault("artifact_path", "")
                payload.setdefault("validation_net_pnl", None)
                payload.setdefault("test_net_pnl", None)
                payload.setdefault("failure_reason", "")
                attempts.append(ExperimentAttempt(**payload))
    return attempts
