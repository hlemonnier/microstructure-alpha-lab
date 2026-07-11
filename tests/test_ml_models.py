import csv
from pathlib import Path

from lob_forge.holdout import (
    build_holdout_manifest,
    canonical_json_sha256,
    sha256_file,
    write_development_csv,
    write_holdout_manifest,
)
from lob_forge.ml_models import (
    _apply_sequence_standardizer,
    _fit_sequence_standardizer,
    _purged_sequential_split_counts,
    _sequence_stateful_economics,
    _stationary_l2_vector,
    SEQUENCE_ECONOMICS_VERSION,
    available_model_specs,
    build_sequence_dataset,
    build_torch_sequence_classifier,
    evaluate_l2_sequence_final_holdout,
    build_masked_pretraining_batch,
    evaluate_l2_tensor_readiness,
    evaluate_model_readiness,
    format_l2_masked_pretraining_report,
    format_l2_sequence_experiment_report,
    format_model_readiness_report,
    fit_sklearn_regressor,
    fit_xgboost_classifier,
    freeze_l2_sequence_candidate,
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


def test_sequence_stateful_economics_flattens_at_the_label_horizon_with_exact_costs() -> None:
    economics = _sequence_stateful_economics(
        [
            [101.0, 1.0, 99.0, 1.0],
            [105.0, 1.0, 103.0, 1.0],
        ],
        [0],
        [2],
        label_horizon=1,
        target_notional=100.0,
        taker_fee_bps=10.0,
        slippage_bps=5.0,
    )

    expected_turnover = 101.0 + 103.0
    expected_net_pnl = (103.0 - 101.0) - expected_turnover * 15.0 / 10_000.0
    expected_break_even_fee_bps = (103.0 - 101.0) / expected_turnover * 10_000.0 - 5.0
    assert economics["trades"] == 1
    assert abs(float(economics["turnover"]) - expected_turnover) < 1e-12
    assert abs(float(economics["net_pnl"]) - expected_net_pnl) < 1e-12
    assert abs(float(economics["break_even_fee_bps"]) - expected_break_even_fee_bps) < 1e-12
    assert abs(float(economics["final_inventory"])) < 1e-12


def test_sequence_stateful_economics_skips_overlapping_prediction_windows() -> None:
    economics = _sequence_stateful_economics(
        [
            [101.0, 1.0, 99.0, 1.0],
            [101.0, 1.0, 99.0, 1.0],
            [101.0, 1.0, 99.0, 1.0],
            [101.0, 1.0, 99.0, 1.0],
            [101.0, 1.0, 99.0, 1.0],
        ],
        [0, 1, 2],
        [2, 2, 2],
        label_horizon=2,
        target_notional=100.0,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
    )

    assert economics["trades"] == 2
    assert abs(float(economics["turnover"]) - 400.0) < 1e-12
    assert abs(float(economics["net_pnl"]) + 4.0) < 1e-12
    assert abs(float(economics["final_inventory"])) < 1e-12


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
            epochs=2,
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
        assert report.l2_sha256 == sha256_file(l2_path)
        assert report.baseline_audit_sha256 == sha256_file(audit_path)
        assert report.holdout_manifest_sha256 == sha256_file(holdout_manifest_path)
        assert report.development_l2_path == str(development_l2_path)
        assert report.development_l2_sha256 == sha256_file(development_l2_path)
        assert report.checkpoint_path == str(checkpoint_path)
        assert report.checkpoint_sha256 == sha256_file(checkpoint_path)
        assert report.prediction_output_path == str(prediction_path)
        assert report.prediction_output_sha256 == sha256_file(prediction_path)
        assert report.economic_simulation_version == SEQUENCE_ECONOMICS_VERSION
        assert abs(report.test_stateful_final_inventory) < 1e-12
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
            lr_scheduler_gamma=0.9,
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=True,
            prediction_output_path=tmp_path / "sequence_tcn_resumed_predictions.csv",
            holdout_manifest_path=holdout_manifest_path,
            development_l2_output_path=development_l2_path,
        )
        assert resumed.resumed_from_checkpoint
        assert resumed.pipeline_completed

        incompatible = run_l2_torch_sequence_experiment(
            model_name="sequence_tcn",
            l2_path=l2_path,
            baseline_audit_path=audit_path,
            output_path=tmp_path / "sequence_tcn_incompatible_results.csv",
            depth=1,
            window=3,
            label_horizon=1,
            epochs=2,
            min_l2_rows=20,
            max_rows=200,
            max_snapshots=50,
            batch_size=2,
            class_weighting="balanced",
            lr_scheduler_gamma=0.9,
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=True,
            prediction_output_path=tmp_path / "sequence_tcn_incompatible_predictions.csv",
        )
        assert not incompatible.resumed_from_checkpoint


def test_l2_sequence_candidate_freeze_and_final_holdout_use_frozen_preprocessing(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.csv"
    l2_path = tmp_path / "bybit_l2.csv"
    development_manifest_path = tmp_path / "development_l2_holdout_manifest.json"
    final_manifest_path = tmp_path / "final_l2_holdout_manifest.json"
    output_path = tmp_path / "sequence_tcn_results.csv"
    checkpoint_path = tmp_path / "sequence_tcn.pt"
    prediction_path = tmp_path / "sequence_tcn_predictions.csv"
    final_prediction_path = tmp_path / "sequence_tcn_final_predictions.csv"
    _write_audit(audit_path, fold_count=20, acceptance_passed=1, rejection_reasons="")
    _write_many_l2_snapshots_and_deltas(l2_path, snapshot_count=30)
    holdout_values = [str(1684195200000 + index) for index in range(20, 30)]
    development_manifest = build_holdout_manifest(
        l2_path,
        split_column="exchange_timestamp",
        holdout_values=holdout_values,
        created_at_utc="2026-06-26T00:00:00Z",
        git_commit="a" * 40,
    )
    write_holdout_manifest(development_manifest, development_manifest_path)

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
            max_rows=100,
            max_snapshots=30,
            batch_size=2,
            checkpoint_path=checkpoint_path,
            prediction_output_path=prediction_path,
            holdout_manifest_path=development_manifest_path,
            development_l2_output_path=tmp_path / "development_l2.csv",
            economic_target_notional=125.0,
            economic_taker_fee_bps=0.25,
            economic_slippage_bps=0.05,
        )
    except RuntimeError as exc:
        assert "torch" in str(exc) or "model readiness gate failed" in str(exc)
        return
    assert report.pipeline_completed

    candidate = freeze_l2_sequence_candidate(output_path)
    assert candidate["candidate_type"] == "l2_sequence_torch_v2"
    assert candidate["checkpoint_sha256"] == sha256_file(checkpoint_path)
    assert candidate["development_l2_sha256"] == sha256_file(tmp_path / "development_l2.csv")
    assert candidate["standardizer_means"]
    assert candidate["economic_target_notional"] == 125.0
    assert candidate["economic_taker_fee_bps"] == 0.25
    assert candidate["economic_slippage_bps"] == 0.05
    development_text = (tmp_path / "development_l2.csv").read_text()
    (tmp_path / "development_l2.csv").write_text(development_text + "\n")
    try:
        freeze_l2_sequence_candidate(output_path)
    except ValueError as exc:
        assert "development_l2_sha256" in str(exc)
    else:
        raise AssertionError("expected tampered development L2 to be rejected")
    (tmp_path / "development_l2.csv").write_text(development_text)
    final_manifest = build_holdout_manifest(
        l2_path,
        split_column="exchange_timestamp",
        holdout_values=holdout_values,
        created_at_utc="2026-06-26T00:00:00Z",
        git_commit="a" * 40,
        candidate_sha256=canonical_json_sha256(candidate),
    )
    write_holdout_manifest(final_manifest, final_manifest_path)

    final_report = evaluate_l2_sequence_final_holdout(
        l2_path=l2_path,
        manifest=final_manifest,
        candidate=candidate,
        predictions_output_path=final_prediction_path,
        device="cpu",
        max_rows=100,
        max_snapshots=30,
    )

    assert final_report.sequence_count > 0
    assert final_report.predictions_output_path == str(final_prediction_path)
    assert final_prediction_path.read_text().splitlines()[1].startswith("holdout,")
    with final_prediction_path.open(newline="") as handle:
        final_predictions = list(csv.DictReader(handle))
    expected_stateful_trades = 0
    prior_exit_index: int | None = None
    for row in sorted(final_predictions, key=lambda value: int(value["sequence_end_index"])):
        if row["predicted_label"] == "1":
            continue
        entry_index = int(row["sequence_end_index"])
        if prior_exit_index is not None and entry_index < prior_exit_index:
            continue
        expected_stateful_trades += 1
        prior_exit_index = entry_index + int(candidate["label_horizon"])
    assert final_report.stateful_trades == expected_stateful_trades

    wrong_source = dict(candidate)
    wrong_source["l2_sha256"] = "0" * 64
    try:
        evaluate_l2_sequence_final_holdout(
            l2_path=l2_path,
            manifest=final_manifest,
            candidate=wrong_source,
            device="cpu",
        )
    except ValueError as exc:
        assert "source L2" in str(exc)
    else:
        raise AssertionError("expected source-L2 candidate mismatch to be rejected")

    tampered = dict(candidate)
    tampered["checkpoint_sha256"] = "0" * 64
    try:
        evaluate_l2_sequence_final_holdout(
            l2_path=l2_path,
            manifest=final_manifest,
            candidate=tampered,
            device="cpu",
        )
    except ValueError as exc:
        assert "checkpoint hash" in str(exc)
    else:
        raise AssertionError("expected tampered checkpoint hash to be rejected")


def test_l2_sequence_candidate_freeze_locks_economic_config_for_final_holdout(tmp_path: Path) -> None:
    l2_path = tmp_path / "source_l2.csv"
    development_l2_path = tmp_path / "development_l2.csv"
    development_manifest_path = tmp_path / "development_holdout.json"
    artifact_path = tmp_path / "sequence_tcn_results.csv"
    checkpoint_path = tmp_path / "sequence_tcn.pt"
    baseline_audit_path = tmp_path / "baseline_audit.csv"
    with l2_path.open("w", newline="") as handle:
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
        for index in range(40):
            timestamp = 1684195200000 + index
            bid = 100.0 + index * 0.1
            ask = bid + 1.0
            for side, price in (("bid", bid), ("ask", ask)):
                writer.writerow(
                    {
                        "event_type": "snapshot",
                        "exchange_timestamp": timestamp,
                        "local_timestamp": timestamp,
                        "side": side,
                        "price": price,
                        "size": 1.0,
                        "sequence": index + 1,
                        "update_id": index + 1,
                        "venue": "bybit",
                        "symbol": "BTCUSDT",
                    }
                )
    checkpoint_path.write_text("not-a-real-checkpoint")
    baseline_audit_path.write_text("fold_count,acceptance_passed,rejection_reasons\n20,1,\n")
    development_manifest = build_holdout_manifest(
        l2_path,
        split_column="exchange_timestamp",
        holdout_values=[str(1684195200000 + index) for index in range(30, 40)],
        created_at_utc="2026-06-26T00:00:00Z",
        git_commit="a" * 40,
    )
    write_holdout_manifest(development_manifest, development_manifest_path)
    development = write_development_csv(l2_path, development_manifest, development_l2_path)
    with artifact_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_name",
                "pipeline_completed",
                "l2_path",
                "l2_sha256",
                "baseline_audit_path",
                "baseline_audit_sha256",
                "checkpoint_path",
                "checkpoint_sha256",
                "holdout_manifest_path",
                "holdout_manifest_sha256",
                "development_l2_path",
                "development_l2_sha256",
                "holdout_manifest_verified",
                "source_rows_before_holdout_filter",
                "development_rows_after_holdout_filter",
                "holdout_rows_excluded",
                "depth",
                "window",
                "label_horizon",
                "flat_threshold_bps",
                "rows_checked",
                "snapshots",
                "economic_target_notional",
                "economic_taker_fee_bps",
                "economic_slippage_bps",
                "economic_simulation_version",
                "test_stateful_final_inventory",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model_name": "sequence_tcn",
                "pipeline_completed": "1",
                "l2_path": str(l2_path),
                "l2_sha256": sha256_file(l2_path),
                "baseline_audit_path": str(baseline_audit_path),
                "baseline_audit_sha256": sha256_file(baseline_audit_path),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "holdout_manifest_path": str(development_manifest_path),
                "holdout_manifest_sha256": sha256_file(development_manifest_path),
                "development_l2_path": str(development_l2_path),
                "development_l2_sha256": sha256_file(development_l2_path),
                "holdout_manifest_verified": "1",
                "source_rows_before_holdout_filter": str(development.source_rows),
                "development_rows_after_holdout_filter": str(development.development_rows),
                "holdout_rows_excluded": str(development.excluded_holdout_rows),
                "depth": "1",
                "window": "3",
                "label_horizon": "1",
                "flat_threshold_bps": "0",
                "rows_checked": "200",
                "snapshots": "80",
                "economic_target_notional": "125",
                "economic_taker_fee_bps": "0.25",
                "economic_slippage_bps": "0.05",
                "economic_simulation_version": SEQUENCE_ECONOMICS_VERSION,
                "test_stateful_final_inventory": "0",
            }
        )

    candidate = freeze_l2_sequence_candidate(artifact_path)
    assert candidate["economic_target_notional"] == 125.0
    assert candidate["economic_taker_fee_bps"] == 0.25
    assert candidate["economic_slippage_bps"] == 0.05
    manifest = build_holdout_manifest(
        l2_path,
        split_column="exchange_timestamp",
        holdout_values=[str(1684195200000 + index) for index in range(20, 30)],
        created_at_utc="2026-06-26T00:00:00Z",
        git_commit="a" * 40,
        candidate_sha256=canonical_json_sha256(candidate),
    )

    try:
        evaluate_l2_sequence_final_holdout(
            l2_path=l2_path,
            manifest=manifest,
            candidate=candidate,
            economic_target_notional=200.0,
        )
    except ValueError as exc:
        assert "economic_target_notional must match frozen candidate" in str(exc)
    else:
        raise AssertionError("expected final sequence holdout to reject economic override drift")


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


def _write_many_l2_snapshots_and_deltas(path: Path, *, snapshot_count: int = 12) -> None:
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
        for index in range(snapshot_count):
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
