from __future__ import annotations

import csv
import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lob_forge.execution_sim import MarketEvent, SignalEvent, StatefulExecutionConfig, simulate_stateful_execution
from lob_forge.holdout import read_holdout_manifest, sha256_file, verify_holdout_manifest_file, write_development_csv


SKLEARN_MODELS = {
    "gradient_boosting": ("sklearn.ensemble", "GradientBoostingClassifier"),
    "random_forest": ("sklearn.ensemble", "RandomForestClassifier"),
    "hist_gradient_boosting": ("sklearn.ensemble", "HistGradientBoostingClassifier"),
}

SKLEARN_REGRESSORS = {
    "gradient_boosting_regressor": ("sklearn.ensemble", "GradientBoostingRegressor"),
    "random_forest_regressor": ("sklearn.ensemble", "RandomForestRegressor"),
    "hist_gradient_boosting_regressor": ("sklearn.ensemble", "HistGradientBoostingRegressor"),
}

OPTIONAL_MODEL_DEPENDENCIES = {
    "sklearn": ("gradient_boosting", "random_forest", "hist_gradient_boosting"),
    "xgboost": ("xgboost_classifier",),
    "torch": ("sequence_mlp", "sequence_tcn", "sequence_transformer", "lob_cnn"),
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    dependency: str
    available: bool
    purpose: str


@dataclass(frozen=True)
class SequenceDataset:
    features: list[str]
    window: int
    sequences: list[list[list[float]]]
    labels: list[int]


@dataclass(frozen=True)
class BaselineReadiness:
    audit_path: Path
    fold_count: int
    acceptance_passed: bool
    rejection_reasons: str
    min_fold_count: int

    @property
    def passed(self) -> bool:
        return self.acceptance_passed and self.fold_count >= self.min_fold_count and not self.rejection_reasons.strip()


@dataclass(frozen=True)
class L2TensorReadiness:
    path: Path
    rows_checked: int
    has_snapshot: bool
    has_delta: bool
    has_sequence: bool
    has_bid: bool
    has_ask: bool
    crossed_updates: int
    sequence_gaps: int
    min_rows: int
    require_delta: bool
    allow_fi2010: bool
    venue_values: tuple[str, ...]

    @property
    def is_fi2010_only(self) -> bool:
        return bool(self.venue_values) and set(self.venue_values) == {"fi2010"}

    @property
    def data_scope(self) -> str:
        if self.is_fi2010_only:
            return "fi2010_benchmark"
        return "true_l2_replay_candidate"

    @property
    def passed(self) -> bool:
        if self.rows_checked < self.min_rows:
            return False
        if not self.has_snapshot or not self.has_bid or not self.has_ask:
            return False
        if not self.has_sequence:
            return False
        if self.crossed_updates or self.sequence_gaps:
            return False
        if self.is_fi2010_only:
            return self.allow_fi2010
        if self.require_delta and not self.has_delta:
            return False
        return True


@dataclass(frozen=True)
class L2MaskedPretrainingReport:
    l2_path: Path
    output_path: Path
    depth: int
    window: int
    rows_checked: int
    snapshots: int
    sequence_count: int
    feature_count: int
    mask_probability: float
    masked_values: int
    mean_reconstruction_mse: float
    zero_reconstruction_mse: float
    mean_abs_error: float
    passed: bool


@dataclass(frozen=True)
class L2SequenceExperimentReport:
    model_name: str
    l2_path: Path
    baseline_audit_path: Path
    output_path: Path
    holdout_manifest_path: str
    holdout_manifest_sha256: str
    holdout_manifest_verified: bool
    development_l2_path: str
    source_rows_before_holdout_filter: int
    development_rows_after_holdout_filter: int
    holdout_rows_excluded: int
    readiness_passed: bool
    dependency_available: bool
    depth: int
    window: int
    label_horizon: int
    flat_threshold_bps: float
    rows_checked: int
    snapshots: int
    sequence_count: int
    feature_count: int
    purge_gap: int
    train_rows: int
    validation_rows: int
    test_rows: int
    epochs: int
    learning_rate: float
    validation_accuracy: float
    validation_macro_f1: float
    validation_balanced_accuracy: float
    test_accuracy: float
    test_macro_f1: float
    test_balanced_accuracy: float
    validation_brier_score: float
    validation_expected_calibration_error: float
    test_brier_score: float
    test_expected_calibration_error: float
    validation_confusion_matrix_json: str
    test_confusion_matrix_json: str
    batch_size: int
    early_stopping_patience: int
    best_epoch: int
    seed: int
    selected_device: str
    class_weighting: str
    lr_scheduler_gamma: float
    final_learning_rate: float
    checkpoint_path: str
    resumed_from_checkpoint: bool
    prediction_output_path: str
    test_stateful_trades: int
    test_stateful_turnover: float
    test_stateful_net_pnl: float
    test_stateful_break_even_fee_bps: float
    pipeline_completed: bool
    acceptance_passed: bool

    @property
    def passed(self) -> bool:
        return self.pipeline_completed


@dataclass(frozen=True)
class SequenceStandardizer:
    means: tuple[float, ...]
    stds: tuple[float, ...]


@dataclass(frozen=True)
class SequenceHoldoutMetadata:
    holdout_manifest_path: str = ""
    holdout_manifest_sha256: str = ""
    holdout_manifest_verified: bool = False
    development_l2_path: str = ""
    source_rows_before_holdout_filter: int = 0
    development_rows_after_holdout_filter: int = 0
    holdout_rows_excluded: int = 0


@dataclass(frozen=True)
class ModelReadinessReport:
    model_name: str
    dependency: str
    dependency_available: bool
    baseline: BaselineReadiness
    l2: L2TensorReadiness
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.dependency_available and self.baseline.passed and self.l2.passed and not self.reasons


def available_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="gradient_boosting",
            family="tree_ensemble",
            dependency="sklearn",
            available=_module_available("sklearn"),
            purpose="nonlinear tabular expected-net-PnL or direction baseline",
        ),
        ModelSpec(
            name="random_forest",
            family="tree_ensemble",
            dependency="sklearn",
            available=_module_available("sklearn"),
            purpose="robust nonlinear tabular baseline and feature sanity check",
        ),
        ModelSpec(
            name="xgboost_classifier",
            family="boosting",
            dependency="xgboost",
            available=_module_available("xgboost"),
            purpose="XGBoost-style baseline when the optional dependency is installed",
        ),
        ModelSpec(
            name="sequence_mlp",
            family="sequence",
            dependency="torch",
            available=_module_available("torch"),
            purpose="top-of-book sequence baseline before true L2 tensors",
        ),
        ModelSpec(
            name="sequence_tcn",
            family="sequence",
            dependency="torch",
            available=_module_available("torch"),
            purpose="TCN extension after top-of-book sequence pipeline is verified",
        ),
        ModelSpec(
            name="sequence_transformer",
            family="sequence",
            dependency="torch",
            available=_module_available("torch"),
            purpose="Transformer extension after baseline and L2 tensor paths are verified",
        ),
        ModelSpec(
            name="lob_cnn",
            family="lob_tensor",
            dependency="torch",
            available=_module_available("torch"),
            purpose="compact LOB tensor CNN smoke model; not a named literature replication",
        ),
    ]


def build_sequence_dataset(
    rows: list[dict[str, str]],
    features: list[str],
    *,
    window: int,
    label_column: str = "label",
) -> SequenceDataset:
    if window <= 0:
        raise ValueError("window must be positive")
    if len(rows) < window:
        raise ValueError("not enough rows for one sequence")
    for feature in features:
        if feature not in rows[0]:
            raise ValueError(f"feature {feature!r} is missing")
    if label_column not in rows[0]:
        raise ValueError(f"label column {label_column!r} is missing")

    sequences: list[list[list[float]]] = []
    labels: list[int] = []
    for end in range(window - 1, len(rows)):
        start = end - window + 1
        sequences.append(
            [[float(rows[index].get(feature, 0.0) or 0.0) for feature in features] for index in range(start, end + 1)]
        )
        labels.append(int(float(rows[end][label_column])))
    return SequenceDataset(features=features, window=window, sequences=sequences, labels=labels)


def fit_sklearn_classifier(
    rows: list[dict[str, str]],
    features: list[str],
    *,
    model_name: str = "gradient_boosting",
    label_column: str = "label",
    random_state: int = 7,
) -> Any:
    if model_name not in SKLEARN_MODELS:
        valid = ", ".join(sorted(SKLEARN_MODELS))
        raise ValueError(f"unknown sklearn model {model_name!r}; expected one of: {valid}")
    if not _module_available("sklearn"):
        raise RuntimeError("scikit-learn is required for sklearn model baselines")
    if not rows:
        raise ValueError("cannot fit a model on empty rows")

    module_name, class_name = SKLEARN_MODELS[model_name]
    module = __import__(module_name, fromlist=[class_name])
    model_class = getattr(module, class_name)
    kwargs = {"random_state": random_state}
    if model_name == "hist_gradient_boosting":
        kwargs = {"random_state": random_state, "max_iter": 100}
    model = model_class(**kwargs)
    x_rows = [[float(row.get(feature, 0.0) or 0.0) for feature in features] for row in rows]
    y_rows = [int(float(row[label_column])) for row in rows]
    return model.fit(x_rows, y_rows)


def fit_sklearn_regressor(
    rows: list[dict[str, str]],
    features: list[str],
    *,
    target_column: str,
    model_name: str = "gradient_boosting_regressor",
    random_state: int = 7,
) -> Any:
    if model_name not in SKLEARN_REGRESSORS:
        valid = ", ".join(sorted(SKLEARN_REGRESSORS))
        raise ValueError(f"unknown sklearn regressor {model_name!r}; expected one of: {valid}")
    if not _module_available("sklearn"):
        raise RuntimeError("scikit-learn is required for sklearn nonlinear expected-PnL regressors")
    if not rows:
        raise ValueError("cannot fit a model on empty rows")
    if target_column not in rows[0]:
        raise ValueError(f"target column {target_column!r} is missing")

    module_name, class_name = SKLEARN_REGRESSORS[model_name]
    module = __import__(module_name, fromlist=[class_name])
    model_class = getattr(module, class_name)
    kwargs = {"random_state": random_state}
    if model_name == "hist_gradient_boosting_regressor":
        kwargs = {"random_state": random_state, "max_iter": 100}
    model = model_class(**kwargs)
    x_rows = [[float(row.get(feature, 0.0) or 0.0) for feature in features] for row in rows]
    y_rows = [float(row[target_column]) for row in rows]
    return model.fit(x_rows, y_rows)


def fit_xgboost_classifier(
    rows: list[dict[str, str]],
    features: list[str],
    *,
    label_column: str = "label",
    random_state: int = 7,
) -> Any:
    if not _module_available("xgboost"):
        raise RuntimeError("xgboost is required for the XGBoost-style baseline")
    if not rows:
        raise ValueError("cannot fit a model on empty rows")
    from xgboost import XGBClassifier

    x_rows = [[float(row.get(feature, 0.0) or 0.0) for feature in features] for row in rows]
    raw_labels = [int(float(row[label_column])) for row in rows]
    class_map = {-1: 0, 0: 1, 1: 2}
    y_rows = [class_map[label] for label in raw_labels]
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=random_state,
    )
    return model.fit(x_rows, y_rows)


def predict_sklearn_probabilities(
    model: Any, rows: list[dict[str, str]], features: list[str]
) -> list[dict[int, float]]:
    if not hasattr(model, "predict_proba"):
        raise ValueError("model does not expose predict_proba")
    x_rows = [[float(row.get(feature, 0.0) or 0.0) for feature in features] for row in rows]
    class_values = [int(value) for value in model.classes_]
    return [
        {klass: float(probability) for klass, probability in zip(class_values, row_probabilities)}
        for row_probabilities in model.predict_proba(x_rows)
    ]


def predict_regression_values(model: Any, rows: list[dict[str, str]], features: list[str]) -> list[float]:
    if not hasattr(model, "predict"):
        raise ValueError("model does not expose predict")
    x_rows = [[float(row.get(feature, 0.0) or 0.0) for feature in features] for row in rows]
    return [float(value) for value in model.predict(x_rows)]


def build_torch_sequence_classifier(
    *,
    window: int,
    feature_count: int,
    model_name: str = "sequence_mlp",
    hidden_size: int = 64,
    class_count: int = 3,
) -> Any:
    if not _module_available("torch"):
        raise RuntimeError("torch is required for sequence MLP/TCN/Transformer/LOB CNN models")
    if window <= 0 or feature_count <= 0:
        raise ValueError("window and feature_count must be positive")
    if model_name not in {"sequence_mlp", "sequence_tcn", "sequence_transformer", "lob_cnn"}:
        raise ValueError("model_name must be one of: sequence_mlp, sequence_tcn, sequence_transformer, lob_cnn")

    import math
    import torch
    import torch.nn as nn

    class TransposeForConv1d(nn.Module):
        def forward(self, tensor: Any) -> Any:
            return tensor.transpose(1, 2)

    class UnsqueezeChannel(nn.Module):
        def forward(self, tensor: Any) -> Any:
            return tensor.unsqueeze(1)

    class LastToken(nn.Module):
        def forward(self, tensor: Any) -> Any:
            return tensor[:, -1, :]

    class Chomp1d(nn.Module):
        def __init__(self, chomp_size: int) -> None:
            super().__init__()
            self.chomp_size = chomp_size

        def forward(self, tensor: Any) -> Any:
            if self.chomp_size == 0:
                return tensor
            return tensor[:, :, : -self.chomp_size]

    class CausalBlock(nn.Module):
        def __init__(self, channels_in: int, channels_out: int, *, dilation: int) -> None:
            super().__init__()
            padding = (3 - 1) * dilation
            self.net = nn.Sequential(
                nn.Conv1d(channels_in, channels_out, kernel_size=3, padding=padding, dilation=dilation),
                Chomp1d(padding),
                nn.ReLU(),
                nn.Conv1d(channels_out, channels_out, kernel_size=3, padding=padding, dilation=dilation),
                Chomp1d(padding),
            )
            self.residual = (
                nn.Conv1d(channels_in, channels_out, kernel_size=1) if channels_in != channels_out else nn.Identity()
            )
            self.activation = nn.ReLU()
            self.receptive_field = 1 + 2 * (3 - 1) * dilation

        def forward(self, tensor: Any) -> Any:
            return self.activation(self.net(tensor) + self.residual(tensor))

    class CausalTCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.Sequential(
                CausalBlock(feature_count, hidden_size, dilation=1),
                CausalBlock(hidden_size, hidden_size, dilation=2),
            )
            self.receptive_field = 1 + 2 * (3 - 1) * (1 + 2)
            self.head = nn.Linear(hidden_size, class_count)

        def forward(self, tensor: Any) -> Any:
            encoded = self.blocks(tensor.transpose(1, 2))
            return self.head(encoded[:, :, -1])

    class PositionalTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Linear(feature_count, hidden_size)
            positions = torch.arange(window, dtype=torch.float32).unsqueeze(1)
            dimensions = torch.arange(hidden_size, dtype=torch.float32).unsqueeze(0)
            div_term = torch.exp(2 * (dimensions // 2) * (-math.log(10000.0) / hidden_size))
            encoding = torch.zeros(window, hidden_size)
            encoding[:, 0::2] = torch.sin(positions * div_term[:, 0::2])
            encoding[:, 1::2] = torch.cos(positions * div_term[:, 1::2])
            self.register_buffer("positional_encoding", encoding.unsqueeze(0), persistent=False)
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=4,
                    batch_first=True,
                    dim_feedforward=hidden_size * 2,
                ),
                num_layers=2,
            )
            self.head = nn.Linear(hidden_size, class_count)

        def forward(self, tensor: Any) -> Any:
            encoded = self.input(tensor) + self.positional_encoding[:, : tensor.shape[1], :]
            return self.head(self.encoder(encoded)[:, -1, :])

    if model_name == "sequence_mlp":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(window * feature_count, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, class_count),
        )
    if model_name == "sequence_tcn":
        return CausalTCN()
    if model_name == "sequence_transformer":
        return PositionalTransformer()
    return nn.Sequential(
        UnsqueezeChannel(),
        nn.Conv2d(1, hidden_size, kernel_size=(3, min(4, feature_count)), padding=(1, 0)),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(hidden_size, class_count),
    )


def require_model_dependency(dependency: str) -> None:
    if dependency not in OPTIONAL_MODEL_DEPENDENCIES:
        valid = ", ".join(sorted(OPTIONAL_MODEL_DEPENDENCIES))
        raise ValueError(f"unknown dependency {dependency!r}; expected one of: {valid}")
    if not _module_available(dependency):
        models = ", ".join(OPTIONAL_MODEL_DEPENDENCIES[dependency])
        raise RuntimeError(f"{dependency} is required for: {models}")


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def build_masked_pretraining_batch(
    sequences: list[list[list[float]]],
    *,
    mask_probability: float = 0.15,
    mask_value: float = 0.0,
) -> tuple[list[list[list[float]]], list[list[list[float]]], list[list[list[int]]]]:
    if not 0.0 <= mask_probability <= 1.0:
        raise ValueError("mask_probability must be between 0 and 1")
    masked: list[list[list[float]]] = []
    targets: list[list[list[float]]] = []
    mask: list[list[list[int]]] = []
    state = 23
    for sequence in sequences:
        masked_sequence: list[list[float]] = []
        target_sequence: list[list[float]] = []
        mask_sequence: list[list[int]] = []
        for row in sequence:
            masked_row: list[float] = []
            target_row: list[float] = []
            mask_row: list[int] = []
            for value in row:
                state = (1103515245 * state + 12345) % (2**31)
                should_mask = (state / (2**31)) < mask_probability
                masked_row.append(mask_value if should_mask else value)
                target_row.append(value)
                mask_row.append(1 if should_mask else 0)
            masked_sequence.append(masked_row)
            target_sequence.append(target_row)
            mask_sequence.append(mask_row)
        masked.append(masked_sequence)
        targets.append(target_sequence)
        mask.append(mask_sequence)
    return masked, targets, mask


def run_l2_masked_pretraining_smoke(
    l2_path: Path | str,
    output_path: Path | str,
    *,
    depth: int = 5,
    window: int = 4,
    mask_probability: float = 0.15,
    max_rows: int = 100000,
    max_snapshots: int = 2000,
) -> L2MaskedPretrainingReport:
    """Run a dependency-free masked reconstruction baseline on true L2 tensors.

    This is a self-supervised pretraining smoke artifact: it verifies that the
    L2 tensor path can produce masked sequence targets and records a deterministic
    mean-imputation reconstruction baseline. It does not claim neural pretraining.
    """
    if depth <= 0:
        raise ValueError("depth must be positive")
    if window <= 0:
        raise ValueError("window must be positive")
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if max_snapshots <= 0:
        raise ValueError("max_snapshots must be positive")

    l2_path = Path(l2_path)
    output_path = Path(output_path)
    snapshots, rows_checked = _load_l2_top_n_vectors(
        l2_path,
        depth=depth,
        max_rows=max_rows,
        max_snapshots=max_snapshots,
        include_time_delta=False,
    )
    if len(snapshots) < window:
        raise ValueError(
            f"not enough L2 snapshots for one pretraining sequence: snapshots={len(snapshots)} window={window}"
        )

    sequences = [snapshots[index : index + window] for index in range(0, len(snapshots) - window + 1)]
    masked, targets, mask = build_masked_pretraining_batch(sequences, mask_probability=mask_probability)
    feature_count = len(sequences[0][0])
    feature_means = _feature_means(targets, feature_count=feature_count)

    masked_values = 0
    squared_error = 0.0
    zero_squared_error = 0.0
    absolute_error = 0.0
    for sequence_index, sequence in enumerate(targets):
        for row_index, row in enumerate(sequence):
            for feature_index, target in enumerate(row):
                if mask[sequence_index][row_index][feature_index] != 1:
                    continue
                masked_values += 1
                prediction = feature_means[feature_index]
                squared_error += (prediction - target) ** 2
                zero_squared_error += target**2
                absolute_error += abs(prediction - target)

    if masked_values == 0:
        raise ValueError("mask_probability produced zero masked values; increase mask_probability or input rows")

    report = L2MaskedPretrainingReport(
        l2_path=l2_path,
        output_path=output_path,
        depth=depth,
        window=window,
        rows_checked=rows_checked,
        snapshots=len(snapshots),
        sequence_count=len(sequences),
        feature_count=feature_count,
        mask_probability=mask_probability,
        masked_values=masked_values,
        mean_reconstruction_mse=squared_error / masked_values,
        zero_reconstruction_mse=zero_squared_error / masked_values,
        mean_abs_error=absolute_error / masked_values,
        passed=True,
    )
    write_l2_masked_pretraining_report(report, output_path)
    return report


def _prepare_sequence_l2_training_source(
    *,
    source_l2_path: Path,
    output_path: Path,
    holdout_manifest_path: Path | str | None,
    development_l2_output_path: Path | str | None,
) -> tuple[Path, SequenceHoldoutMetadata]:
    if holdout_manifest_path is None:
        return source_l2_path, SequenceHoldoutMetadata()

    manifest_path = Path(holdout_manifest_path)
    if not verify_holdout_manifest_file(manifest_path):
        raise ValueError(f"holdout manifest verification failed: {manifest_path}")
    manifest = read_holdout_manifest(manifest_path)
    development_path = (
        Path(development_l2_output_path)
        if development_l2_output_path is not None
        else output_path.with_name(output_path.stem + "_development_l2.csv")
    )
    development = write_development_csv(source_l2_path, manifest, development_path)
    return development.path, SequenceHoldoutMetadata(
        holdout_manifest_path=str(manifest_path),
        holdout_manifest_sha256=sha256_file(manifest_path),
        holdout_manifest_verified=True,
        development_l2_path=str(development.path),
        source_rows_before_holdout_filter=development.source_rows,
        development_rows_after_holdout_filter=development.development_rows,
        holdout_rows_excluded=development.excluded_holdout_rows,
    )


def run_l2_torch_sequence_experiment(
    *,
    model_name: str,
    l2_path: Path | str,
    baseline_audit_path: Path | str,
    output_path: Path | str,
    depth: int = 5,
    window: int = 16,
    label_horizon: int = 1,
    flat_threshold_bps: float = 0.0,
    epochs: int = 3,
    learning_rate: float = 0.001,
    max_rows: int = 100000,
    max_snapshots: int = 2000,
    min_fold_count: int = 20,
    min_l2_rows: int = 1000,
    seed: int = 7,
    batch_size: int = 32,
    early_stopping_patience: int = 3,
    device: str = "auto",
    class_weighting: str = "none",
    lr_scheduler_gamma: float = 1.0,
    checkpoint_path: Path | str | None = None,
    resume_from_checkpoint: bool = False,
    prediction_output_path: Path | str | None = None,
    holdout_manifest_path: Path | str | None = None,
    development_l2_output_path: Path | str | None = None,
    economic_target_notional: float = 100.0,
    economic_taker_fee_bps: float = 1.0,
    economic_slippage_bps: float = 0.0,
) -> L2SequenceExperimentReport:
    if model_name not in {"sequence_tcn", "sequence_transformer"}:
        raise ValueError("model_name must be sequence_tcn or sequence_transformer")
    if depth <= 0:
        raise ValueError("depth must be positive")
    if window <= 1:
        raise ValueError("window must be greater than 1")
    if label_horizon <= 0:
        raise ValueError("label_horizon must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if class_weighting not in {"none", "balanced"}:
        raise ValueError("class_weighting must be none or balanced")
    if not 0.0 < lr_scheduler_gamma <= 1.0:
        raise ValueError("lr_scheduler_gamma must be in (0, 1]")
    if economic_target_notional <= 0.0:
        raise ValueError("economic_target_notional must be positive")

    source_l2_path = Path(l2_path)
    baseline_audit_path = Path(baseline_audit_path)
    output_path = Path(output_path)
    training_l2_path, holdout_metadata = _prepare_sequence_l2_training_source(
        source_l2_path=source_l2_path,
        output_path=output_path,
        holdout_manifest_path=holdout_manifest_path,
        development_l2_output_path=development_l2_output_path,
    )
    readiness = evaluate_model_readiness(
        model_name=model_name,
        baseline_audit_path=baseline_audit_path,
        l2_path=training_l2_path,
        min_fold_count=min_fold_count,
        min_l2_rows=min_l2_rows,
        require_delta=True,
        allow_fi2010=False,
        max_l2_rows=max_rows,
    )
    if not readiness.passed:
        reasons = "; ".join(readiness.reasons) or "readiness prerequisites failed"
        raise RuntimeError(f"model readiness gate failed for {model_name}: {reasons}")

    snapshots, rows_checked = _load_l2_top_n_vectors(
        training_l2_path,
        depth=depth,
        max_rows=max_rows,
        max_snapshots=max_snapshots,
        include_time_delta=True,
    )
    sequences, labels, sequence_end_indices = _l2_direction_sequences(
        snapshots,
        window=window,
        label_horizon=label_horizon,
        flat_threshold_bps=flat_threshold_bps,
    )
    if len(sequences) < 5:
        raise ValueError(f"not enough labeled L2 sequences for train/validation/test split: {len(sequences)}")
    sequences = _stationarize_l2_sequences(sequences)
    feature_count = len(sequences[0][0])
    train_count, validation_count, test_count, purge_gap = _purged_sequential_split_counts(
        len(sequences),
        purge_gap=window + label_horizon - 1,
    )
    validation_start = train_count + purge_gap
    validation_end = validation_start + validation_count
    test_start = validation_end + purge_gap
    train_sequences = sequences[:train_count]
    validation_sequences = sequences[validation_start:validation_end]
    test_sequences = sequences[test_start:]
    train_labels = labels[:train_count]
    validation_labels = labels[validation_start:validation_end]
    test_labels = labels[test_start:]
    validation_end_indices = sequence_end_indices[validation_start:validation_end]
    test_end_indices = sequence_end_indices[test_start:]
    standardizer = _fit_sequence_standardizer(train_sequences)
    train_sequences = _apply_sequence_standardizer(train_sequences, standardizer)
    validation_sequences = _apply_sequence_standardizer(validation_sequences, standardizer)
    test_sequences = _apply_sequence_standardizer(test_sequences, standardizer)

    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("torch is required for sequence model experiments") from exc

    selected_device = _select_torch_device(torch, device)
    torch_device = torch.device(selected_device)
    torch.manual_seed(seed)
    if selected_device == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = build_torch_sequence_classifier(
        window=window,
        feature_count=feature_count,
        model_name=model_name,
        class_count=3,
    ).to(torch_device)
    class_weights = _torch_class_weights(train_labels, torch, torch_device) if class_weighting == "balanced" else None
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, epochs // 3),
        gamma=lr_scheduler_gamma,
    )

    x_train = torch.tensor(train_sequences, dtype=torch.float32, device=torch_device)
    y_train = torch.tensor(train_labels, dtype=torch.long, device=torch_device)
    x_validation = torch.tensor(validation_sequences, dtype=torch.float32, device=torch_device)
    y_validation = torch.tensor(validation_labels, dtype=torch.long, device=torch_device)
    x_test = torch.tensor(test_sequences, dtype=torch.float32, device=torch_device)
    y_test = torch.tensor(test_labels, dtype=torch.long, device=torch_device)

    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_validation_loss = math.inf
    stale_epochs = 0
    start_epoch = 1
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    resumed_from_checkpoint = False
    if resume_from_checkpoint and checkpoint is not None and checkpoint.exists():
        payload = torch.load(checkpoint, map_location=torch_device)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        if "scheduler_state" in payload:
            scheduler.load_state_dict(payload["scheduler_state"])
        best_state = payload.get("best_model_state")
        best_epoch = int(payload.get("best_epoch", 0))
        best_validation_loss = float(payload.get("best_validation_loss", math.inf))
        stale_epochs = int(payload.get("stale_epochs", 0))
        start_epoch = int(payload.get("epoch", 0)) + 1
        resumed_from_checkpoint = True
    effective_batch_size = max(1, min(batch_size, len(train_sequences)))
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        generator = torch.Generator()
        generator.manual_seed(seed + epoch)
        indices = torch.randperm(x_train.shape[0], generator=generator).to(torch_device)
        for start in range(0, x_train.shape[0], effective_batch_size):
            batch_indices = indices[start : start + effective_batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[batch_indices]), y_train[batch_indices])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(x_validation), y_validation).item())
        if validation_loss < best_validation_loss - 1e-12:
            best_validation_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= early_stopping_patience:
                break
        scheduler.step()
        if checkpoint is not None:
            _write_torch_sequence_checkpoint(
                torch,
                checkpoint,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_state=best_state,
                best_epoch=best_epoch,
                best_validation_loss=best_validation_loss,
                stale_epochs=stale_epochs,
                seed=seed,
            )
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        validation_logits = model(x_validation)
        test_logits = model(x_test)
        validation_probabilities = torch.softmax(validation_logits, dim=1).detach().cpu().tolist()
        test_probabilities = torch.softmax(test_logits, dim=1).detach().cpu().tolist()
        validation_predictions = validation_logits.argmax(dim=1).detach().cpu().tolist()
        test_predictions = test_logits.argmax(dim=1).detach().cpu().tolist()

    validation_accuracy, validation_macro_f1, validation_balanced_accuracy, validation_confusion = (
        _classification_report(
            y_validation.detach().cpu().tolist(),
            validation_predictions,
        )
    )
    test_accuracy, test_macro_f1, test_balanced_accuracy, test_confusion = _classification_report(
        y_test.detach().cpu().tolist(),
        test_predictions,
    )
    validation_brier_score, validation_ece = _classification_probability_report(
        y_validation.detach().cpu().tolist(),
        validation_predictions,
        validation_probabilities,
    )
    test_brier_score, test_ece = _classification_probability_report(
        y_test.detach().cpu().tolist(),
        test_predictions,
        test_probabilities,
    )
    prediction_path = Path(prediction_output_path) if prediction_output_path is not None else None
    if prediction_path is not None:
        _write_sequence_predictions(
            prediction_path,
            validation_labels=y_validation.detach().cpu().tolist(),
            validation_predictions=validation_predictions,
            validation_probabilities=validation_probabilities,
            validation_end_indices=validation_end_indices,
            test_labels=y_test.detach().cpu().tolist(),
            test_predictions=test_predictions,
            test_probabilities=test_probabilities,
            test_end_indices=test_end_indices,
        )
    economic = _sequence_stateful_economics(
        snapshots,
        test_end_indices,
        test_predictions,
        label_horizon=label_horizon,
        target_notional=economic_target_notional,
        taker_fee_bps=economic_taker_fee_bps,
        slippage_bps=economic_slippage_bps,
    )
    report = L2SequenceExperimentReport(
        model_name=model_name,
        l2_path=source_l2_path,
        baseline_audit_path=baseline_audit_path,
        output_path=output_path,
        holdout_manifest_path=holdout_metadata.holdout_manifest_path,
        holdout_manifest_sha256=holdout_metadata.holdout_manifest_sha256,
        holdout_manifest_verified=holdout_metadata.holdout_manifest_verified,
        development_l2_path=holdout_metadata.development_l2_path,
        source_rows_before_holdout_filter=holdout_metadata.source_rows_before_holdout_filter,
        development_rows_after_holdout_filter=holdout_metadata.development_rows_after_holdout_filter,
        holdout_rows_excluded=holdout_metadata.holdout_rows_excluded,
        readiness_passed=readiness.passed,
        dependency_available=readiness.dependency_available,
        depth=depth,
        window=window,
        label_horizon=label_horizon,
        flat_threshold_bps=flat_threshold_bps,
        rows_checked=rows_checked,
        snapshots=len(snapshots),
        sequence_count=len(sequences),
        feature_count=feature_count,
        purge_gap=purge_gap,
        train_rows=train_count,
        validation_rows=validation_count,
        test_rows=test_count,
        epochs=epochs,
        learning_rate=learning_rate,
        validation_accuracy=validation_accuracy,
        validation_macro_f1=validation_macro_f1,
        validation_balanced_accuracy=validation_balanced_accuracy,
        test_accuracy=test_accuracy,
        test_macro_f1=test_macro_f1,
        test_balanced_accuracy=test_balanced_accuracy,
        validation_brier_score=validation_brier_score,
        validation_expected_calibration_error=validation_ece,
        test_brier_score=test_brier_score,
        test_expected_calibration_error=test_ece,
        validation_confusion_matrix_json=json.dumps(validation_confusion, sort_keys=True),
        test_confusion_matrix_json=json.dumps(test_confusion, sort_keys=True),
        batch_size=effective_batch_size,
        early_stopping_patience=early_stopping_patience,
        best_epoch=best_epoch,
        seed=seed,
        selected_device=selected_device,
        class_weighting=class_weighting,
        lr_scheduler_gamma=lr_scheduler_gamma,
        final_learning_rate=float(optimizer.param_groups[0]["lr"]),
        checkpoint_path=str(checkpoint) if checkpoint is not None else "",
        resumed_from_checkpoint=resumed_from_checkpoint,
        prediction_output_path=str(prediction_path) if prediction_path is not None else "",
        test_stateful_trades=int(economic["trades"]),
        test_stateful_turnover=float(economic["turnover"]),
        test_stateful_net_pnl=float(economic["net_pnl"]),
        test_stateful_break_even_fee_bps=float(economic["break_even_fee_bps"]),
        pipeline_completed=readiness.passed and test_count > 0 and math.isfinite(test_macro_f1),
        acceptance_passed=False,
    )
    write_l2_sequence_experiment_report(report, output_path)
    return report


def write_l2_sequence_experiment_report(report: L2SequenceExperimentReport, path: Path | str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_name",
        "l2_path",
        "baseline_audit_path",
        "holdout_manifest_path",
        "holdout_manifest_sha256",
        "holdout_manifest_verified",
        "development_l2_path",
        "source_rows_before_holdout_filter",
        "development_rows_after_holdout_filter",
        "holdout_rows_excluded",
        "readiness_passed",
        "dependency_available",
        "depth",
        "window",
        "label_horizon",
        "flat_threshold_bps",
        "rows_checked",
        "snapshots",
        "sequence_count",
        "feature_count",
        "purge_gap",
        "train_rows",
        "validation_rows",
        "test_rows",
        "epochs",
        "learning_rate",
        "validation_accuracy",
        "validation_macro_f1",
        "validation_balanced_accuracy",
        "test_accuracy",
        "test_macro_f1",
        "test_balanced_accuracy",
        "validation_brier_score",
        "validation_expected_calibration_error",
        "test_brier_score",
        "test_expected_calibration_error",
        "validation_confusion_matrix_json",
        "test_confusion_matrix_json",
        "batch_size",
        "early_stopping_patience",
        "best_epoch",
        "seed",
        "selected_device",
        "class_weighting",
        "lr_scheduler_gamma",
        "final_learning_rate",
        "checkpoint_path",
        "resumed_from_checkpoint",
        "prediction_output_path",
        "test_stateful_trades",
        "test_stateful_turnover",
        "test_stateful_net_pnl",
        "test_stateful_break_even_fee_bps",
        "pipeline_completed",
        "acceptance_passed",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "model_name": report.model_name,
                "l2_path": _display_path(report.l2_path),
                "baseline_audit_path": _display_path(report.baseline_audit_path),
                "holdout_manifest_path": report.holdout_manifest_path,
                "holdout_manifest_sha256": report.holdout_manifest_sha256,
                "holdout_manifest_verified": int(report.holdout_manifest_verified),
                "development_l2_path": report.development_l2_path,
                "source_rows_before_holdout_filter": report.source_rows_before_holdout_filter,
                "development_rows_after_holdout_filter": report.development_rows_after_holdout_filter,
                "holdout_rows_excluded": report.holdout_rows_excluded,
                "readiness_passed": int(report.readiness_passed),
                "dependency_available": int(report.dependency_available),
                "depth": report.depth,
                "window": report.window,
                "label_horizon": report.label_horizon,
                "flat_threshold_bps": f"{report.flat_threshold_bps:.12g}",
                "rows_checked": report.rows_checked,
                "snapshots": report.snapshots,
                "sequence_count": report.sequence_count,
                "feature_count": report.feature_count,
                "purge_gap": report.purge_gap,
                "train_rows": report.train_rows,
                "validation_rows": report.validation_rows,
                "test_rows": report.test_rows,
                "epochs": report.epochs,
                "learning_rate": f"{report.learning_rate:.12g}",
                "validation_accuracy": f"{report.validation_accuracy:.12g}",
                "validation_macro_f1": f"{report.validation_macro_f1:.12g}",
                "validation_balanced_accuracy": f"{report.validation_balanced_accuracy:.12g}",
                "test_accuracy": f"{report.test_accuracy:.12g}",
                "test_macro_f1": f"{report.test_macro_f1:.12g}",
                "test_balanced_accuracy": f"{report.test_balanced_accuracy:.12g}",
                "validation_brier_score": f"{report.validation_brier_score:.12g}",
                "validation_expected_calibration_error": f"{report.validation_expected_calibration_error:.12g}",
                "test_brier_score": f"{report.test_brier_score:.12g}",
                "test_expected_calibration_error": f"{report.test_expected_calibration_error:.12g}",
                "validation_confusion_matrix_json": report.validation_confusion_matrix_json,
                "test_confusion_matrix_json": report.test_confusion_matrix_json,
                "batch_size": report.batch_size,
                "early_stopping_patience": report.early_stopping_patience,
                "best_epoch": report.best_epoch,
                "seed": report.seed,
                "selected_device": report.selected_device,
                "class_weighting": report.class_weighting,
                "lr_scheduler_gamma": f"{report.lr_scheduler_gamma:.12g}",
                "final_learning_rate": f"{report.final_learning_rate:.12g}",
                "checkpoint_path": report.checkpoint_path,
                "resumed_from_checkpoint": int(report.resumed_from_checkpoint),
                "prediction_output_path": report.prediction_output_path,
                "test_stateful_trades": report.test_stateful_trades,
                "test_stateful_turnover": f"{report.test_stateful_turnover:.12g}",
                "test_stateful_net_pnl": f"{report.test_stateful_net_pnl:.12g}",
                "test_stateful_break_even_fee_bps": f"{report.test_stateful_break_even_fee_bps:.12g}",
                "pipeline_completed": int(report.pipeline_completed),
                "acceptance_passed": int(report.acceptance_passed),
            }
        )
    return output_path


def format_l2_sequence_experiment_report(report: L2SequenceExperimentReport, *, output_format: str = "text") -> str:
    if output_format == "csv":
        fields = [
            "model_name",
            "l2_path",
            "output_path",
            "holdout_manifest_path",
            "holdout_manifest_sha256",
            "holdout_manifest_verified",
            "development_l2_path",
            "source_rows_before_holdout_filter",
            "development_rows_after_holdout_filter",
            "holdout_rows_excluded",
            "sequence_count",
            "purge_gap",
            "train_rows",
            "validation_rows",
            "test_rows",
            "validation_macro_f1",
            "validation_balanced_accuracy",
            "test_macro_f1",
            "test_balanced_accuracy",
            "test_brier_score",
            "test_expected_calibration_error",
            "test_stateful_net_pnl",
            "test_stateful_break_even_fee_bps",
            "pipeline_completed",
            "acceptance_passed",
        ]
        values = [
            report.model_name,
            str(report.l2_path),
            str(report.output_path),
            report.holdout_manifest_path,
            report.holdout_manifest_sha256,
            str(int(report.holdout_manifest_verified)),
            report.development_l2_path,
            str(report.source_rows_before_holdout_filter),
            str(report.development_rows_after_holdout_filter),
            str(report.holdout_rows_excluded),
            str(report.sequence_count),
            str(report.purge_gap),
            str(report.train_rows),
            str(report.validation_rows),
            str(report.test_rows),
            f"{report.validation_macro_f1:.12g}",
            f"{report.validation_balanced_accuracy:.12g}",
            f"{report.test_macro_f1:.12g}",
            f"{report.test_balanced_accuracy:.12g}",
            f"{report.test_brier_score:.12g}",
            f"{report.test_expected_calibration_error:.12g}",
            f"{report.test_stateful_net_pnl:.12g}",
            f"{report.test_stateful_break_even_fee_bps:.12g}",
            str(int(report.pipeline_completed)),
            str(int(report.acceptance_passed)),
        ]
        return ",".join(fields) + "\n" + ",".join(values)
    if output_format != "text":
        raise ValueError("output_format must be text or csv")
    return "\n".join(
        [
            f"model_name={report.model_name}",
            f"l2_path={report.l2_path}",
            f"baseline_audit_path={report.baseline_audit_path}",
            f"output_path={report.output_path}",
            f"holdout_manifest_path={report.holdout_manifest_path}",
            f"holdout_manifest_sha256={report.holdout_manifest_sha256}",
            f"holdout_manifest_verified={int(report.holdout_manifest_verified)}",
            f"development_l2_path={report.development_l2_path}",
            f"source_rows_before_holdout_filter={report.source_rows_before_holdout_filter}",
            f"development_rows_after_holdout_filter={report.development_rows_after_holdout_filter}",
            f"holdout_rows_excluded={report.holdout_rows_excluded}",
            f"readiness_passed={int(report.readiness_passed)}",
            f"dependency_available={int(report.dependency_available)}",
            f"depth={report.depth}",
            f"window={report.window}",
            f"label_horizon={report.label_horizon}",
            f"flat_threshold_bps={report.flat_threshold_bps:.12g}",
            f"rows_checked={report.rows_checked}",
            f"snapshots={report.snapshots}",
            f"sequence_count={report.sequence_count}",
            f"feature_count={report.feature_count}",
            f"purge_gap={report.purge_gap}",
            f"train_rows={report.train_rows}",
            f"validation_rows={report.validation_rows}",
            f"test_rows={report.test_rows}",
            f"epochs={report.epochs}",
            f"learning_rate={report.learning_rate:.12g}",
            f"validation_accuracy={report.validation_accuracy:.12g}",
            f"validation_macro_f1={report.validation_macro_f1:.12g}",
            f"validation_balanced_accuracy={report.validation_balanced_accuracy:.12g}",
            f"test_accuracy={report.test_accuracy:.12g}",
            f"test_macro_f1={report.test_macro_f1:.12g}",
            f"test_balanced_accuracy={report.test_balanced_accuracy:.12g}",
            f"validation_brier_score={report.validation_brier_score:.12g}",
            f"validation_expected_calibration_error={report.validation_expected_calibration_error:.12g}",
            f"test_brier_score={report.test_brier_score:.12g}",
            f"test_expected_calibration_error={report.test_expected_calibration_error:.12g}",
            f"validation_confusion_matrix_json={report.validation_confusion_matrix_json}",
            f"test_confusion_matrix_json={report.test_confusion_matrix_json}",
            f"batch_size={report.batch_size}",
            f"early_stopping_patience={report.early_stopping_patience}",
            f"best_epoch={report.best_epoch}",
            f"seed={report.seed}",
            f"selected_device={report.selected_device}",
            f"class_weighting={report.class_weighting}",
            f"lr_scheduler_gamma={report.lr_scheduler_gamma:.12g}",
            f"final_learning_rate={report.final_learning_rate:.12g}",
            f"checkpoint_path={report.checkpoint_path}",
            f"resumed_from_checkpoint={int(report.resumed_from_checkpoint)}",
            f"prediction_output_path={report.prediction_output_path}",
            f"test_stateful_trades={report.test_stateful_trades}",
            f"test_stateful_turnover={report.test_stateful_turnover:.12g}",
            f"test_stateful_net_pnl={report.test_stateful_net_pnl:.12g}",
            f"test_stateful_break_even_fee_bps={report.test_stateful_break_even_fee_bps:.12g}",
            f"pipeline_completed={int(report.pipeline_completed)}",
            f"acceptance_passed={int(report.acceptance_passed)}",
        ]
    )


def write_l2_masked_pretraining_report(report: L2MaskedPretrainingReport, path: Path | str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "l2_path",
        "depth",
        "window",
        "rows_checked",
        "snapshots",
        "sequence_count",
        "feature_count",
        "mask_probability",
        "masked_values",
        "mean_reconstruction_mse",
        "zero_reconstruction_mse",
        "mean_abs_error",
        "passed",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "l2_path": str(report.l2_path),
                "depth": report.depth,
                "window": report.window,
                "rows_checked": report.rows_checked,
                "snapshots": report.snapshots,
                "sequence_count": report.sequence_count,
                "feature_count": report.feature_count,
                "mask_probability": f"{report.mask_probability:.12g}",
                "masked_values": report.masked_values,
                "mean_reconstruction_mse": f"{report.mean_reconstruction_mse:.12g}",
                "zero_reconstruction_mse": f"{report.zero_reconstruction_mse:.12g}",
                "mean_abs_error": f"{report.mean_abs_error:.12g}",
                "passed": int(report.passed),
            }
        )
    return output_path


def format_l2_masked_pretraining_report(report: L2MaskedPretrainingReport, *, output_format: str = "text") -> str:
    if output_format == "csv":
        fields = [
            "l2_path",
            "output_path",
            "depth",
            "window",
            "rows_checked",
            "snapshots",
            "sequence_count",
            "feature_count",
            "mask_probability",
            "masked_values",
            "mean_reconstruction_mse",
            "zero_reconstruction_mse",
            "mean_abs_error",
            "passed",
        ]
        values = [
            str(report.l2_path),
            str(report.output_path),
            str(report.depth),
            str(report.window),
            str(report.rows_checked),
            str(report.snapshots),
            str(report.sequence_count),
            str(report.feature_count),
            f"{report.mask_probability:.12g}",
            str(report.masked_values),
            f"{report.mean_reconstruction_mse:.12g}",
            f"{report.zero_reconstruction_mse:.12g}",
            f"{report.mean_abs_error:.12g}",
            str(int(report.passed)),
        ]
        return ",".join(fields) + "\n" + ",".join(values)
    if output_format != "text":
        raise ValueError("output_format must be text or csv")
    return "\n".join(
        [
            f"l2_path={report.l2_path}",
            f"output_path={report.output_path}",
            f"depth={report.depth}",
            f"window={report.window}",
            f"rows_checked={report.rows_checked}",
            f"snapshots={report.snapshots}",
            f"sequence_count={report.sequence_count}",
            f"feature_count={report.feature_count}",
            f"mask_probability={report.mask_probability:.12g}",
            f"masked_values={report.masked_values}",
            f"mean_reconstruction_mse={report.mean_reconstruction_mse:.12g}",
            f"zero_reconstruction_mse={report.zero_reconstruction_mse:.12g}",
            f"mean_abs_error={report.mean_abs_error:.12g}",
            f"passed={int(report.passed)}",
        ]
    )


def evaluate_model_readiness(
    *,
    model_name: str,
    baseline_audit_path: Path | str,
    l2_path: Path | str,
    min_fold_count: int = 20,
    min_l2_rows: int = 1000,
    require_delta: bool = True,
    allow_fi2010: bool = False,
    max_l2_rows: int = 100000,
) -> ModelReadinessReport:
    spec = _model_spec_by_name(model_name)
    baseline = evaluate_baseline_readiness(baseline_audit_path, min_fold_count=min_fold_count)
    l2 = evaluate_l2_tensor_readiness(
        l2_path,
        min_rows=min_l2_rows,
        require_delta=require_delta,
        allow_fi2010=allow_fi2010,
        max_rows=max_l2_rows,
    )
    reasons: list[str] = []
    if not spec.available:
        reasons.append(f"dependency {spec.dependency} is not available")
    if not baseline.passed:
        reasons.append("baseline audit gate failed")
    if not l2.passed:
        reasons.append("L2 tensor data gate failed")
    return ModelReadinessReport(
        model_name=model_name,
        dependency=spec.dependency,
        dependency_available=spec.available,
        baseline=baseline,
        l2=l2,
        reasons=tuple(reasons),
    )


def evaluate_baseline_readiness(path: Path | str, *, min_fold_count: int = 20) -> BaselineReadiness:
    path = Path(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"baseline audit must contain exactly one row: {path}")
    row = rows[0]
    return BaselineReadiness(
        audit_path=path,
        fold_count=int(float(row.get("fold_count", "0") or 0)),
        acceptance_passed=bool(int(float(row.get("acceptance_passed", "0") or 0))),
        rejection_reasons=row.get("rejection_reasons", ""),
        min_fold_count=min_fold_count,
    )


def evaluate_l2_tensor_readiness(
    path: Path | str,
    *,
    min_rows: int = 1000,
    require_delta: bool = True,
    allow_fi2010: bool = False,
    max_rows: int = 100000,
) -> L2TensorReadiness:
    if min_rows <= 0:
        raise ValueError("min_rows must be positive")
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    path = Path(path)
    rows_checked = 0
    has_snapshot = False
    has_delta = False
    has_sequence = False
    has_bid = False
    has_ask = False
    crossed_updates = 0
    sequence_gaps = 0
    venues: set[str] = set()
    last_sequence: int | None = None
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    current_event_key: tuple[str, int | None, str, str] | None = None
    current_event_type = ""
    current_sequence: int | None = None
    pending_levels: list[tuple[str, float, float]] = []

    def flush_event() -> None:
        nonlocal crossed_updates, sequence_gaps, last_sequence
        nonlocal current_event_type, current_sequence, pending_levels
        if not pending_levels:
            return
        if current_sequence is not None and last_sequence is not None and current_sequence > last_sequence + 1:
            sequence_gaps += 1
        if current_sequence is not None:
            last_sequence = max(last_sequence, current_sequence) if last_sequence is not None else current_sequence
        if current_event_type == "snapshot":
            bids.clear()
            asks.clear()
        for side, price, size in pending_levels:
            levels = bids if side == "bid" else asks if side == "ask" else None
            if levels is None:
                continue
            if size <= 0.0:
                levels.pop(price, None)
            else:
                levels[price] = size
        if bids and asks and max(bids) >= min(asks):
            crossed_updates += 1
        pending_levels = []

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"event_type", "side", "price", "size"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"normalized L2 file missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            if rows_checked >= max_rows:
                break
            rows_checked += 1
            event_type = row.get("event_type", "")
            side = row.get("side", "")
            price = float(row.get("price", "0") or 0.0)
            size = float(row.get("size", "0") or 0.0)
            sequence = _optional_int(row.get("update_id") or row.get("sequence"))
            venue = row.get("venue", "")
            if venue:
                venues.add(venue)
            has_snapshot = has_snapshot or event_type == "snapshot"
            has_delta = has_delta or event_type == "delta"
            has_bid = has_bid or side == "bid"
            has_ask = has_ask or side == "ask"
            has_sequence = has_sequence or sequence is not None
            event_key = (
                event_type,
                sequence,
                row.get("exchange_timestamp", ""),
                row.get("local_timestamp", ""),
            )
            if current_event_key is not None and event_key != current_event_key:
                flush_event()
            current_event_key = event_key
            current_event_type = event_type
            current_sequence = sequence
            pending_levels.append((side, price, size))
    flush_event()
    return L2TensorReadiness(
        path=path,
        rows_checked=rows_checked,
        has_snapshot=has_snapshot,
        has_delta=has_delta,
        has_sequence=has_sequence,
        has_bid=has_bid,
        has_ask=has_ask,
        crossed_updates=crossed_updates,
        sequence_gaps=sequence_gaps,
        min_rows=min_rows,
        require_delta=require_delta,
        allow_fi2010=allow_fi2010,
        venue_values=tuple(sorted(venues)),
    )


def format_model_readiness_report(report: ModelReadinessReport, *, output_format: str = "text") -> str:
    if output_format == "csv":
        fields = [
            "model_name",
            "dependency",
            "dependency_available",
            "baseline_fold_count",
            "baseline_acceptance_passed",
            "l2_rows_checked",
            "l2_has_snapshot",
            "l2_has_delta",
            "l2_has_sequence",
            "l2_crossed_updates",
            "l2_sequence_gaps",
            "l2_venues",
            "l2_data_scope",
            "passed",
            "reasons",
        ]
        values = [
            report.model_name,
            report.dependency,
            str(int(report.dependency_available)),
            str(report.baseline.fold_count),
            str(int(report.baseline.acceptance_passed)),
            str(report.l2.rows_checked),
            str(int(report.l2.has_snapshot)),
            str(int(report.l2.has_delta)),
            str(int(report.l2.has_sequence)),
            str(report.l2.crossed_updates),
            str(report.l2.sequence_gaps),
            " ".join(report.l2.venue_values),
            report.l2.data_scope,
            str(int(report.passed)),
            "; ".join(report.reasons),
        ]
        return ",".join(fields) + "\n" + ",".join(values)
    if output_format != "text":
        raise ValueError("output_format must be text or csv")
    lines = [
        f"model_name={report.model_name}",
        f"dependency={report.dependency}",
        f"dependency_available={int(report.dependency_available)}",
        f"baseline_audit={report.baseline.audit_path}",
        f"baseline_fold_count={report.baseline.fold_count}",
        f"baseline_acceptance_passed={int(report.baseline.acceptance_passed)}",
        f"baseline_rejection_reasons={report.baseline.rejection_reasons}",
        f"l2_path={report.l2.path}",
        f"l2_rows_checked={report.l2.rows_checked}",
        f"l2_has_snapshot={int(report.l2.has_snapshot)}",
        f"l2_has_delta={int(report.l2.has_delta)}",
        f"l2_has_sequence={int(report.l2.has_sequence)}",
        f"l2_has_bid={int(report.l2.has_bid)}",
        f"l2_has_ask={int(report.l2.has_ask)}",
        f"l2_crossed_updates={report.l2.crossed_updates}",
        f"l2_sequence_gaps={report.l2.sequence_gaps}",
        f"l2_venues={' '.join(report.l2.venue_values)}",
        f"l2_data_scope={report.l2.data_scope}",
        f"passed={int(report.passed)}",
    ]
    for reason in report.reasons:
        lines.append(f"reason={reason}")
    return "\n".join(lines)


def _model_spec_by_name(model_name: str) -> ModelSpec:
    specs = {spec.name: spec for spec in available_model_specs()}
    try:
        return specs[model_name]
    except KeyError as exc:
        valid = ", ".join(sorted(specs))
        raise ValueError(f"unknown model {model_name!r}; expected one of: {valid}") from exc


def _load_l2_top_n_vectors(
    path: Path,
    *,
    depth: int,
    max_rows: int,
    max_snapshots: int,
    include_time_delta: bool = False,
) -> tuple[list[list[float]], int]:
    snapshots: list[list[float]] = []
    rows_checked = 0
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    current_event_key: tuple[str, int | None, str, str] | None = None
    current_event_type = ""
    current_exchange_timestamp = ""
    previous_snapshot_timestamp: int | None = None
    pending_levels: list[tuple[str, float, float]] = []

    def flush_event() -> None:
        nonlocal pending_levels, previous_snapshot_timestamp
        if not pending_levels:
            return
        if current_event_type == "snapshot":
            bids.clear()
            asks.clear()
        for side, price, size in pending_levels:
            levels = bids if side == "bid" else asks if side == "ask" else None
            if levels is None:
                continue
            if size <= 0.0:
                levels.pop(price, None)
            else:
                levels[price] = size
        pending_levels = []
        if not bids or not asks:
            return
        if max(bids) >= min(asks):
            return
        vector = _book_vector(bids, asks, depth=depth)
        if include_time_delta:
            timestamp = _optional_int(current_exchange_timestamp) or previous_snapshot_timestamp or 0
            delta_ms = 0 if previous_snapshot_timestamp is None else max(0, timestamp - previous_snapshot_timestamp)
            vector.append(math.log1p(delta_ms))
            previous_snapshot_timestamp = timestamp
        snapshots.append(vector)

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"event_type", "side", "price", "size"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"normalized L2 file missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            if rows_checked >= max_rows or len(snapshots) >= max_snapshots:
                break
            rows_checked += 1
            event_type = row.get("event_type", "")
            sequence = _optional_int(row.get("update_id") or row.get("sequence"))
            event_key = (
                event_type,
                sequence,
                row.get("exchange_timestamp", ""),
                row.get("local_timestamp", ""),
            )
            if current_event_key is not None and event_key != current_event_key:
                flush_event()
            current_event_key = event_key
            current_event_type = event_type
            current_exchange_timestamp = row.get("exchange_timestamp", "")
            pending_levels.append(
                (
                    row.get("side", ""),
                    float(row.get("price", "0") or 0.0),
                    float(row.get("size", "0") or 0.0),
                )
            )
    if len(snapshots) < max_snapshots:
        flush_event()
    return snapshots, rows_checked


def _book_vector(bids: dict[float, float], asks: dict[float, float], *, depth: int) -> list[float]:
    bid_levels = sorted(bids.items(), key=lambda item: item[0], reverse=True)[:depth]
    ask_levels = sorted(asks.items(), key=lambda item: item[0])[:depth]
    vector: list[float] = []
    for index in range(depth):
        ask = ask_levels[index] if index < len(ask_levels) else (0.0, 0.0)
        bid = bid_levels[index] if index < len(bid_levels) else (0.0, 0.0)
        vector.extend([ask[0], ask[1], bid[0], bid[1]])
    return vector


def _feature_means(sequences: list[list[list[float]]], *, feature_count: int) -> list[float]:
    totals = [0.0] * feature_count
    count = 0
    for sequence in sequences:
        for row in sequence:
            count += 1
            for index, value in enumerate(row):
                totals[index] += value
    if count == 0:
        raise ValueError("cannot compute feature means on empty sequences")
    return [total / count for total in totals]


def _l2_direction_sequences(
    snapshots: list[list[float]],
    *,
    window: int,
    label_horizon: int,
    flat_threshold_bps: float,
) -> tuple[list[list[list[float]]], list[int], list[int]]:
    if len(snapshots) < window + label_horizon:
        raise ValueError(
            "not enough L2 snapshots for labeled sequence experiment: "
            f"snapshots={len(snapshots)} window={window} label_horizon={label_horizon}"
        )
    sequences: list[list[list[float]]] = []
    labels: list[int] = []
    end_indices: list[int] = []
    for end_index in range(window - 1, len(snapshots) - label_horizon):
        current_mid = _top_mid_from_vector(snapshots[end_index])
        future_mid = _top_mid_from_vector(snapshots[end_index + label_horizon])
        if current_mid <= 0.0:
            continue
        move_bps = 10000.0 * (future_mid - current_mid) / current_mid
        if move_bps > flat_threshold_bps:
            label = 2
        elif move_bps < -flat_threshold_bps:
            label = 0
        else:
            label = 1
        start_index = end_index - window + 1
        sequences.append(snapshots[start_index : end_index + 1])
        labels.append(label)
        end_indices.append(end_index)
    if not sequences:
        raise ValueError("L2 snapshots produced no labeled sequences")
    return sequences, labels, end_indices


def _top_mid_from_vector(vector: list[float]) -> float:
    if len(vector) < 3:
        return 0.0
    ask_price = vector[0]
    bid_price = vector[2]
    return (ask_price + bid_price) / 2.0 if ask_price > 0.0 and bid_price > 0.0 else 0.0


def _stationarize_l2_sequences(sequences: list[list[list[float]]]) -> list[list[list[float]]]:
    return [[_stationary_l2_vector(row) for row in sequence] for sequence in sequences]


def _stationary_l2_vector(vector: list[float]) -> list[float]:
    if len(vector) < 4:
        return vector[:]
    ask0 = vector[0]
    bid0 = vector[2]
    mid = (ask0 + bid0) / 2.0 if ask0 > 0.0 and bid0 > 0.0 else 0.0
    book_feature_count = len(vector) - (1 if len(vector) % 4 == 1 else 0)
    transformed: list[float] = []
    for index in range(0, book_feature_count, 4):
        ask_price, ask_size, bid_price, bid_size = vector[index : index + 4]
        ask_distance = 10000.0 * (ask_price - mid) / mid if mid > 0.0 and ask_price > 0.0 else 0.0
        bid_distance = 10000.0 * (bid_price - mid) / mid if mid > 0.0 and bid_price > 0.0 else 0.0
        transformed.extend(
            [
                ask_distance,
                math.log1p(max(0.0, ask_size)),
                bid_distance,
                math.log1p(max(0.0, bid_size)),
            ]
        )
    if book_feature_count < len(vector):
        transformed.append(vector[-1])
    return transformed


def _fit_sequence_standardizer(sequences: list[list[list[float]]]) -> SequenceStandardizer:
    if not sequences:
        raise ValueError("cannot fit sequence standardizer on empty data")
    feature_count = len(sequences[0][0])
    totals = [0.0] * feature_count
    squared_totals = [0.0] * feature_count
    count = 0
    for sequence in sequences:
        for row in sequence:
            count += 1
            for index, value in enumerate(row):
                totals[index] += value
                squared_totals[index] += value * value
    if count == 0:
        raise ValueError("cannot fit sequence standardizer on empty data")
    means = [total / count for total in totals]
    variances = [max(squared_totals[index] / count - means[index] ** 2, 0.0) for index in range(feature_count)]
    stds = [math.sqrt(variance) if variance > 1e-12 else 1.0 for variance in variances]
    return SequenceStandardizer(means=tuple(means), stds=tuple(stds))


def _apply_sequence_standardizer(
    sequences: list[list[list[float]]],
    standardizer: SequenceStandardizer,
) -> list[list[list[float]]]:
    return [
        [
            [(value - standardizer.means[index]) / standardizer.stds[index] for index, value in enumerate(row)]
            for row in sequence
        ]
        for sequence in sequences
    ]


def _standardize_sequences(sequences: list[list[list[float]]]) -> list[list[list[float]]]:
    return _apply_sequence_standardizer(sequences, _fit_sequence_standardizer(sequences))


def _purged_sequential_split_counts(total_rows: int, *, purge_gap: int) -> tuple[int, int, int, int]:
    if purge_gap < 0:
        raise ValueError("purge_gap must be non-negative")
    if total_rows < 5:
        raise ValueError("need at least 5 sequences for train/validation/test split")
    available_after_purge = total_rows - 2 * purge_gap
    if available_after_purge < 5:
        raise ValueError("Insufficient samples to satisfy the requested sequence purge and embargo.")
    train_rows, validation_rows, test_rows = _sequential_split_counts(available_after_purge)
    return train_rows, validation_rows, test_rows, purge_gap


def _sequential_split_counts(total_rows: int) -> tuple[int, int, int]:
    if total_rows < 5:
        raise ValueError("need at least 5 sequences for train/validation/test split")
    train_rows = max(1, int(total_rows * 0.6))
    validation_rows = max(1, int(total_rows * 0.2))
    test_rows = total_rows - train_rows - validation_rows
    if test_rows <= 0:
        test_rows = 1
        if validation_rows > 1:
            validation_rows -= 1
        else:
            train_rows -= 1
    if train_rows <= 0 or validation_rows <= 0 or test_rows <= 0:
        raise ValueError("invalid train/validation/test split")
    return train_rows, validation_rows, test_rows


def _select_torch_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return requested


def _torch_class_weights(labels: list[int], torch: Any, device: Any) -> Any:
    counts = [labels.count(klass) for klass in (0, 1, 2)]
    total = len(labels)
    weights = [total / (3.0 * count) if count else 0.0 for count in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _write_torch_sequence_checkpoint(
    torch: Any,
    path: Path,
    *,
    epoch: int,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    best_state: dict[str, Any] | None,
    best_epoch: int,
    best_validation_loss: float,
    stale_epochs: int,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_model_state": best_state,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "stale_epochs": stale_epochs,
            "seed": seed,
        },
        path,
    )


def _write_sequence_predictions(
    path: Path,
    *,
    validation_labels: list[int],
    validation_predictions: list[int],
    validation_probabilities: list[list[float]],
    validation_end_indices: list[int],
    test_labels: list[int],
    test_predictions: list[int],
    test_probabilities: list[list[float]],
    test_end_indices: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "row",
        "sequence_end_index",
        "true_label",
        "predicted_label",
        "prob_down",
        "prob_flat",
        "prob_up",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        rows = [
            ("validation", validation_labels, validation_predictions, validation_probabilities, validation_end_indices),
            ("test", test_labels, test_predictions, test_probabilities, test_end_indices),
        ]
        for split, labels, predictions, probabilities, end_indices in rows:
            for row_number, (label, prediction, probability, end_index) in enumerate(
                zip(labels, predictions, probabilities, end_indices),
                start=1,
            ):
                padded = (probability + [0.0, 0.0, 0.0])[:3]
                writer.writerow(
                    {
                        "split": split,
                        "row": row_number,
                        "sequence_end_index": end_index,
                        "true_label": label,
                        "predicted_label": prediction,
                        "prob_down": f"{padded[0]:.12g}",
                        "prob_flat": f"{padded[1]:.12g}",
                        "prob_up": f"{padded[2]:.12g}",
                    }
                )


def _sequence_stateful_economics(
    snapshots: list[list[float]],
    end_indices: list[int],
    predictions: list[int],
    *,
    label_horizon: int,
    target_notional: float,
    taker_fee_bps: float,
    slippage_bps: float,
) -> dict[str, float | int]:
    events: list[MarketEvent] = []
    signals: list[SignalEvent] = []
    for row_number, (end_index, prediction) in enumerate(zip(end_indices, predictions), start=1):
        if end_index + label_horizon >= len(snapshots):
            continue
        current = snapshots[end_index]
        future = snapshots[end_index + label_horizon]
        current_bid, current_ask = _top_bid_ask_from_vector(current)
        future_bid, future_ask = _top_bid_ask_from_vector(future)
        if min(current_bid, current_ask, future_bid, future_ask) <= 0.0:
            continue
        event_time = row_number * 2000
        future_time = event_time + 1000
        events.append(MarketEvent(event_time, bid=current_bid, ask=current_ask, bid_size=10.0, ask_size=10.0))
        events.append(MarketEvent(future_time, bid=future_bid, ask=future_ask, bid_size=10.0, ask_size=10.0))
        side = -1 if prediction == 0 else 1 if prediction == 2 else 0
        signals.append(
            SignalEvent(
                event_time,
                target_side=side,
                target_notional=target_notional if side else 0.0,
                signal_id=f"neural-test-{row_number}",
            )
        )
    if not events:
        return {"trades": 0, "turnover": 0.0, "net_pnl": 0.0, "break_even_fee_bps": 0.0}
    initial_cash = 1000.0
    result = simulate_stateful_execution(
        events,
        signals,
        config=StatefulExecutionConfig(
            initial_cash=initial_cash,
            max_position_notional=target_notional * 2.0,
            max_leverage=1.0,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
            latency_ms=0,
        ),
    )
    fees = sum(fill.fee for fill in result.fills)
    net_pnl = result.final_equity - initial_cash
    gross_pnl = net_pnl + fees
    break_even = gross_pnl / result.turnover * 10_000.0 if result.turnover else 0.0
    return {
        "trades": len(result.fills),
        "turnover": result.turnover,
        "net_pnl": net_pnl,
        "break_even_fee_bps": break_even,
    }


def _top_bid_ask_from_vector(vector: list[float]) -> tuple[float, float]:
    if len(vector) < 3:
        return 0.0, 0.0
    return vector[2], vector[0]


def _classification_report(labels: list[int], predictions: list[int]) -> tuple[float, float, float, list[list[int]]]:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length")
    confusion = [[0, 0, 0] for _ in range(3)]
    for label, prediction in zip(labels, predictions):
        if 0 <= label <= 2 and 0 <= prediction <= 2:
            confusion[label][prediction] += 1
    if not labels:
        return 0.0, 0.0, 0.0, confusion
    accuracy = sum(1 for label, prediction in zip(labels, predictions) if label == prediction) / len(labels)
    f1_scores: list[float] = []
    recalls: list[float] = []
    for klass in (0, 1, 2):
        tp = sum(1 for label, prediction in zip(labels, predictions) if label == klass and prediction == klass)
        fp = sum(1 for label, prediction in zip(labels, predictions) if label != klass and prediction == klass)
        fn = sum(1 for label, prediction in zip(labels, predictions) if label == klass and prediction != klass)
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)
        if tp == 0 and (fp > 0 or fn > 0):
            f1_scores.append(0.0)
            continue
        if tp == 0:
            f1_scores.append(0.0)
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_scores.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return accuracy, sum(f1_scores) / len(f1_scores), sum(recalls) / len(recalls), confusion


def _classification_scores(labels: list[int], predictions: list[int]) -> tuple[float, float]:
    accuracy, macro_f1, _, _ = _classification_report(labels, predictions)
    return accuracy, macro_f1


def _classification_probability_report(
    labels: list[int],
    predictions: list[int],
    probabilities: list[list[float]],
    *,
    bins: int = 10,
) -> tuple[float, float]:
    if len(labels) != len(predictions) or len(labels) != len(probabilities):
        raise ValueError("labels, predictions and probabilities must have the same length")
    if not labels:
        return 0.0, 0.0
    brier = 0.0
    confidence_buckets: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for label, prediction, probability in zip(labels, predictions, probabilities):
        padded = (probability + [0.0, 0.0, 0.0])[:3]
        for klass in (0, 1, 2):
            target = 1.0 if label == klass else 0.0
            brier += (padded[klass] - target) ** 2
        confidence = max(padded)
        correct = 1.0 if label == prediction else 0.0
        bucket = min(bins - 1, int(confidence * bins))
        confidence_buckets[bucket].append((confidence, correct))
    ece = 0.0
    for bucket_values in confidence_buckets:
        if not bucket_values:
            continue
        weight = len(bucket_values) / len(labels)
        avg_confidence = sum(value[0] for value in bucket_values) / len(bucket_values)
        avg_accuracy = sum(value[1] for value in bucket_values) / len(bucket_values)
        ece += weight * abs(avg_accuracy - avg_confidence)
    return brier / len(labels), ece


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path)


def _optional_int(value: object) -> int | None:
    if value in {"", None}:
        return None
    try:
        return int(str(value))
    except ValueError:
        try:
            return int(float(str(value)))
        except ValueError:
            return None
