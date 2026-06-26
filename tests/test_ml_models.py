import csv
from pathlib import Path

from lob_forge.holdout import build_holdout_manifest, sha256_file, write_holdout_manifest
from lob_forge.ml_models import (
    _apply_sequence_standardizer,
    _fit_sequence_standardizer,
    _purged_sequential_split_counts,
    _stationary_l2_vector,
    available_model_specs,
    build_sequence_dataset,
    build_torch_sequence_classifier,
    build_masked_pretraining_batch,
    evaluate_l2_tensor_readiness,
    evaluate_model_readiness,
    format_l2_masked_pretraining_report,
    format_l2_sequence_experiment_report,
    format_model_readiness_report,
    fit_sklearn_regressor,
    fit_xgboost_classifier,
    require_model_dependency,
    run_l2_masked_pretraining_smoke,
    run_l2_torch_sequence_experiment,
)


def test_model_specs_include_optional_research_stack() -> None:
    specs = {spec.name: spec for spec in available_model_specs()}

    assert "gradient_boosting" in specs
    assert "random_forest" in specs
    assert "xgboost_classifier" in specs
    assert "sequence_mlp" in specs
    assert "sequence_transformer" in specs
    assert "lob_cnn" in specs
    assert "not a named literature replication" in specs["lob_cnn"].purpose


def test_build_sequence_dataset_creates_rolling_windows() -> None:
    rows = [
        {"microprice_deviation": "0.1", "top_imbalance": "0.2", "label": "1"},
        {"microprice_deviation": "0.2", "top_imbalance": "0.3", "label": "0"},
        {"microprice_deviation": "0.3", "top_imbalance": "0.4", "label": "-1"},
    ]

    dataset = build_sequence_dataset(rows, ["microprice_deviation", "top_imbalance"], window=2)

    assert dataset.window == 2
    assert len(dataset.sequences) == 2
    assert dataset.sequences[0][0] == [0.1, 0.2]
    assert dataset.labels == [0, -1]


def test_require_unknown_dependency_is_rejected() -> None:
    try:
        require_model_dependency("missing")
    except ValueError as exc:
        assert "unknown dependency" in str(exc)
    else:
        raise AssertionError("expected unknown dependency to be rejected")


def test_optional_model_trainers_are_dependency_gated() -> None:
    specs = {spec.name: spec for spec in available_model_specs()}

    if not specs["xgboost_classifier"].available:
        try:
            fit_xgboost_classifier([{"feature": "1.0", "label": "1"}], ["feature"])
        except RuntimeError as exc:
            assert "xgboost" in str(exc)
        else:
            raise AssertionError("expected missing xgboost dependency to be rejected")

    if not specs["gradient_boosting"].available:
        try:
            fit_sklearn_regressor([{"feature": "1.0", "target": "1.0"}], ["feature"], target_column="target")
        except RuntimeError as exc:
            assert "scikit-learn" in str(exc)
        else:
            raise AssertionError("expected missing scikit-learn dependency to be rejected")


def test_torch_sequence_builder_is_dependency_gated_or_builds() -> None:
    try:
        model = build_torch_sequence_classifier(window=4, feature_count=3, model_name="sequence_mlp")
    except RuntimeError as exc:
        assert "torch" in str(exc)
    else:
        assert model is not None


def test_torch_tcn_and_transformer_architecture_contracts() -> None:
    try:
        tcn = build_torch_sequence_classifier(window=8, feature_count=5, model_name="sequence_tcn")
        transformer = build_torch_sequence_classifier(window=8, feature_count=5, model_name="sequence_transformer")
    except RuntimeError as exc:
        assert "torch" in str(exc)
    else:
        assert getattr(tcn, "receptive_field") >= 8
        assert hasattr(transformer, "positional_encoding")


def test_masked_pretraining_batch_returns_targets_and_mask() -> None:
    sequences = [[[1.0, 2.0], [3.0, 4.0]]]

    masked, targets, mask = build_masked_pretraining_batch(sequences, mask_probability=1.0)

    assert masked == [[[0.0, 0.0], [0.0, 0.0]]]
    assert targets == sequences
    assert mask == [[[1, 1], [1, 1]]]


def test_sequence_standardizer_is_fit_on_training_sequences_only() -> None:
    train = [
        [[1.0, 10.0], [3.0, 14.0]],
        [[5.0, 18.0], [7.0, 22.0]],
    ]
    validation = [[[1000.0, 2000.0], [1200.0, 2400.0]]]

    standardizer = _fit_sequence_standardizer(train)
    transformed = _apply_sequence_standardizer(train + validation, standardizer)

    train_values = [row[0] for sequence in transformed[:2] for row in sequence]
    validation_values = [row[0] for sequence in transformed[2:] for row in sequence]
    assert abs(sum(train_values)) < 1e-12
    assert min(validation_values) > 100.0


def test_purged_sequence_split_leaves_embargo_between_folds() -> None:
    train, validation, test, gap = _purged_sequential_split_counts(12, purge_gap=3)

    validation_start = train + gap
    test_start = validation_start + validation + gap
    assert gap == 3
    assert validation_start > train
    assert test_start + test == 12


def test_purged_sequence_split_rejects_insufficient_samples() -> None:
    try:
        _purged_sequential_split_counts(6, purge_gap=3)
    except ValueError as exc:
        assert "Insufficient samples" in str(exc)
    else:
        raise AssertionError("expected insufficient purged split to be rejected")


def test_l2_tensor_readiness_requires_true_l2_when_fi2010_not_allowed(tmp_path: Path) -> None:
    l2_path = tmp_path / "fi2010.csv"
    _write_normalized_l2(
        l2_path,
        [
            ("snapshot", "bid", 100.0, 1.0, 1, "fi2010"),
            ("snapshot", "ask", 101.0, 1.0, 1, "fi2010"),
            ("snapshot", "bid", 99.0, 0.5, 2, "fi2010"),
        ],
    )

    blocked = evaluate_l2_tensor_readiness(l2_path, min_rows=3, allow_fi2010=False)
    allowed = evaluate_l2_tensor_readiness(l2_path, min_rows=3, allow_fi2010=True)

    assert blocked.is_fi2010_only
    assert blocked.data_scope == "fi2010_benchmark"
    assert not blocked.passed
    assert allowed.passed


def test_l2_tensor_readiness_requires_deltas_for_crypto_l2_by_default(tmp_path: Path) -> None:
    l2_path = tmp_path / "okx_snapshots.csv"
    _write_normalized_l2(
        l2_path,
        [
            ("snapshot", "bid", 100.0, 1.0, 1, "okx"),
            ("snapshot", "ask", 101.0, 1.0, 1, "okx"),
            ("snapshot", "bid", 99.0, 0.5, 2, "okx"),
        ],
    )

    report = evaluate_l2_tensor_readiness(l2_path, min_rows=3)

    assert report.data_scope == "true_l2_replay_candidate"
    assert not report.has_delta
    assert not report.passed


def test_l2_tensor_readiness_groups_row_expanded_bybit_events(tmp_path: Path) -> None:
    l2_path = tmp_path / "bybit_l2.csv"
    _write_bybit_row_expanded_l2(l2_path)

    report = evaluate_l2_tensor_readiness(l2_path, min_rows=6)

    assert report.has_snapshot
    assert report.has_delta
    assert report.has_sequence
    assert report.crossed_updates == 0
    assert report.sequence_gaps == 0
    assert report.passed


def test_l2_stationary_vector_uses_mid_distance_and_log_depth() -> None:
    vector = _stationary_l2_vector([101.0, 3.0, 99.0, 1.0, 7.0])

    assert round(vector[0], 6) == 100.0
    assert round(vector[2], 6) == -100.0
    assert vector[1] > vector[3]
    assert vector[-1] == 7.0


def test_sequence_standardizer_fits_train_only() -> None:
    train_sequences = [[[1.0], [1.0]], [[1.0], [1.0]]]
    validation_test = [[[1000.0], [1000.0]]]

    standardizer = _fit_sequence_standardizer(train_sequences)
    transformed = _apply_sequence_standardizer(train_sequences + validation_test, standardizer)

    assert transformed[0][0][0] == 0.0
    assert transformed[-1][0][0] == 999.0


def test_l2_masked_pretraining_smoke_writes_artifact(tmp_path: Path) -> None:
    l2_path = tmp_path / "bybit_l2.csv"
    output_path = tmp_path / "self_supervised_pretraining.csv"
    _write_bybit_row_expanded_l2(l2_path)

    report = run_l2_masked_pretraining_smoke(
        l2_path,
        output_path,
        depth=1,
        window=2,
        mask_probability=1.0,
    )
    text = format_l2_masked_pretraining_report(report)

    assert report.passed
    assert report.sequence_count == 1
    assert report.feature_count == 4
    assert report.masked_values == 8
    assert output_path.exists()
    assert "pipeline_completed=1" in text

    with output_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["l2_path"] == str(l2_path)
    assert rows[0]["pipeline_completed"] == "1"


def test_model_readiness_gate_checks_baseline_l2_and_dependency(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.csv"
    l2_path = tmp_path / "okx_l2.csv"
    _write_audit(audit_path, fold_count=20, acceptance_passed=1, rejection_reasons="")
    _write_normalized_l2(
        l2_path,
        [
            ("snapshot", "bid", 100.0, 1.0, 1, "okx"),
            ("snapshot", "ask", 101.0, 1.0, 1, "okx"),
            ("delta", "bid", 100.0, 0.5, 2, "okx"),
            ("delta", "ask", 101.0, 0.5, 3, "okx"),
        ],
    )

    report = evaluate_model_readiness(
        model_name="sequence_tcn",
        baseline_audit_path=audit_path,
        l2_path=l2_path,
        min_fold_count=20,
        min_l2_rows=4,
    )
    text = format_model_readiness_report(report)
    csv_text = format_model_readiness_report(report, output_format="csv")

    assert report.baseline.passed
    assert report.l2.passed
    assert report.passed == report.dependency_available
    assert "model_name=sequence_tcn" in text
    assert "l2_data_scope=true_l2_replay_candidate" in text
    assert csv_text.splitlines()[0].startswith("model_name,dependency")


def test_model_readiness_gate_fails_rejected_baseline(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.csv"
    l2_path = tmp_path / "okx_l2.csv"
    _write_audit(audit_path, fold_count=4, acceptance_passed=0, rejection_reasons="fold count 4 < required 20")
    _write_normalized_l2(
        l2_path,
        [
            ("snapshot", "bid", 100.0, 1.0, 1, "okx"),
            ("snapshot", "ask", 101.0, 1.0, 1, "okx"),
            ("delta", "bid", 100.0, 0.5, 2, "okx"),
        ],
    )

    report = evaluate_model_readiness(
        model_name="sequence_transformer",
        baseline_audit_path=audit_path,
        l2_path=l2_path,
        min_fold_count=20,
        min_l2_rows=3,
    )

    assert not report.baseline.passed
    assert not report.passed
    assert "baseline audit gate failed" in report.reasons


def test_l2_sequence_experiment_is_readiness_and_dependency_gated(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.csv"
    l2_path = tmp_path / "bybit_l2.csv"
    output_path = tmp_path / "sequence_tcn_results.csv"
    checkpoint_path = tmp_path / "sequence_tcn.pt"
    prediction_path = tmp_path / "sequence_tcn_predictions.csv"
    holdout_manifest_path = tmp_path / "l2_holdout_manifest.json"
    development_l2_path = tmp_path / "sequence_tcn_development_l2.csv"
    _write_audit(audit_path, fold_count=20, acceptance_passed=1, rejection_reasons="")
    _write_many_l2_snapshots_and_deltas(l2_path)
    holdout_manifest = build_holdout_manifest(
        l2_path,
        split_column="exchange_timestamp",
        holdout_values=["1684195200000"],
        created_at_utc="2026-06-26T00:00:00Z",
        git_commit="a" * 40,
    )
    write_holdout_manifest(holdout_manifest, holdout_manifest_path)

    try:
        report = run_l2_torch_sequence_experiment(
            model_name="sequence_tcn",
            l2_path=l2_path,
            baseline_audit_path=audit_path,
            output_path=output_path,
            depth=1,
            window=3,
            label_horizon=1,
            epochs=1,
            min_l2_rows=20,
            max_rows=200,
            max_snapshots=50,
            batch_size=2,
            class_weighting="balanced",
            lr_scheduler_gamma=0.9,
            checkpoint_path=checkpoint_path,
            prediction_output_path=prediction_path,
            holdout_manifest_path=holdout_manifest_path,
            development_l2_output_path=development_l2_path,
        )
    except RuntimeError as exc:
        assert "torch" in str(exc) or "model readiness gate failed" in str(exc)
        assert not output_path.exists()
    else:
        text = format_l2_sequence_experiment_report(report)
        assert report.passed
        assert report.model_name == "sequence_tcn"
        assert report.test_rows > 0
        assert output_path.exists()
        assert development_l2_path.exists()
        assert checkpoint_path.exists()
        assert prediction_path.exists()
        result_rows = list(csv.DictReader(output_path.open()))
        assert result_rows[0]["holdout_manifest_path"] == str(holdout_manifest_path)
        assert result_rows[0]["holdout_manifest_sha256"] == sha256_file(holdout_manifest_path)
        assert result_rows[0]["holdout_manifest_verified"] == "1"
        assert result_rows[0]["development_l2_path"] == str(development_l2_path)
        assert result_rows[0]["source_rows_before_holdout_filter"] == "24"
        assert result_rows[0]["development_rows_after_holdout_filter"] == "22"
        assert result_rows[0]["holdout_rows_excluded"] == "2"
        development_rows = list(csv.DictReader(development_l2_path.open()))
        assert {row["exchange_timestamp"] for row in development_rows} == {
            str(1684195200000 + index) for index in range(1, 12)
        }
        assert prediction_path.read_text().splitlines()[0].startswith("split,row,sequence_end_index")
        assert report.selected_device in {"cpu", "cuda"}
        assert report.class_weighting == "balanced"
        assert report.holdout_manifest_verified
        assert report.holdout_manifest_sha256 == sha256_file(holdout_manifest_path)
        assert report.development_l2_path == str(development_l2_path)
        assert report.checkpoint_path == str(checkpoint_path)
        assert report.prediction_output_path == str(prediction_path)
        assert report.test_stateful_trades >= 0
        assert "pipeline_completed=1" in text
        assert f"holdout_manifest_sha256={sha256_file(holdout_manifest_path)}" in text
        assert "holdout_rows_excluded=2" in text
        assert "acceptance_passed=0" in text
        assert "test_brier_score=" in text
        assert "test_stateful_break_even_fee_bps=" in text
        resumed = run_l2_torch_sequence_experiment(
            model_name="sequence_tcn",
            l2_path=l2_path,
            baseline_audit_path=audit_path,
            output_path=tmp_path / "sequence_tcn_resumed_results.csv",
            depth=1,
            window=3,
            label_horizon=1,
            epochs=2,
            min_l2_rows=20,
            max_rows=200,
            max_snapshots=50,
            batch_size=2,
            class_weighting="balanced",
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=True,
            prediction_output_path=tmp_path / "sequence_tcn_resumed_predictions.csv",
        )
        assert resumed.resumed_from_checkpoint
        assert resumed.pipeline_completed


def _write_audit(path: Path, *, fold_count: int, acceptance_passed: int, rejection_reasons: str) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold_count", "acceptance_passed", "rejection_reasons"])
        writer.writeheader()
        writer.writerow(
            {
                "fold_count": fold_count,
                "acceptance_passed": acceptance_passed,
                "rejection_reasons": rejection_reasons,
            }
        )


def _write_normalized_l2(path: Path, rows: list[tuple[str, str, float, float, int, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "event_type",
                "exchange_timestamp",
                "local_timestamp",
                "side",
                "price",
                "size",
                "sequence",
                "update_id",
                "venue",
                "symbol",
            ],
        )
        writer.writeheader()
        for index, (event_type, side, price, size, sequence, venue) in enumerate(rows, start=1):
            writer.writerow(
                {
                    "event_type": event_type,
                    "exchange_timestamp": index,
                    "local_timestamp": index,
                    "side": side,
                    "price": price,
                    "size": size,
                    "sequence": sequence,
                    "update_id": "",
                    "venue": venue,
                    "symbol": "BTC-USDT-SWAP",
                }
            )


def _write_bybit_row_expanded_l2(path: Path) -> None:
    rows = [
        ("snapshot", 1684195201451, "bid", 100.0, 1.0, 100, 1),
        ("snapshot", 1684195201451, "bid", 99.0, 2.0, 100, 1),
        ("snapshot", 1684195201451, "ask", 101.0, 1.0, 100, 1),
        ("snapshot", 1684195201451, "ask", 102.0, 2.0, 100, 1),
        ("delta", 1684195201547, "bid", 100.0, 0.5, 350, 2),
        ("delta", 1684195201547, "ask", 101.0, 0.5, 350, 2),
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "event_type",
                "exchange_timestamp",
                "local_timestamp",
                "side",
                "price",
                "size",
                "sequence",
                "update_id",
                "venue",
                "symbol",
            ],
        )
        writer.writeheader()
        for event_type, timestamp, side, price, size, sequence, update_id in rows:
            writer.writerow(
                {
                    "event_type": event_type,
                    "exchange_timestamp": timestamp,
                    "local_timestamp": timestamp,
                    "side": side,
                    "price": price,
                    "size": size,
                    "sequence": sequence,
                    "update_id": update_id,
                    "venue": "bybit",
                    "symbol": "BTCUSDT",
                }
            )


def _write_many_l2_snapshots_and_deltas(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "event_type",
                "exchange_timestamp",
                "local_timestamp",
                "side",
                "price",
                "size",
                "sequence",
                "update_id",
                "venue",
                "symbol",
            ],
        )
        writer.writeheader()
        sequence = 1
        for index in range(12):
            bid = 100.0 + index * 0.1
            ask = 101.0 + index * 0.1
            event_type = "snapshot" if index == 0 else "delta"
            for side, price in (("bid", bid), ("ask", ask)):
                writer.writerow(
                    {
                        "event_type": event_type,
                        "exchange_timestamp": 1684195200000 + index,
                        "local_timestamp": 1684195200000 + index,
                        "side": side,
                        "price": price,
                        "size": 1.0,
                        "sequence": sequence,
                        "update_id": sequence,
                        "venue": "bybit",
                        "symbol": "BTCUSDT",
                    }
                )
            sequence += 1
