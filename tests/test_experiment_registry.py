from pathlib import Path

from lob_forge.experiment_registry import read_experiment_registry, write_threshold_experiment_registry


def test_threshold_experiment_registry_records_full_search_family(tmp_path: Path) -> None:
    path = tmp_path / "experiment_registry.jsonl"

    attempts = write_threshold_experiment_registry(
        path,
        run_id="fixture-run",
        model_class="threshold_rule",
        features=["microprice_deviation", "trade_imbalance"],
        thresholds=[0.05, 0.15],
        selection_metric="validation_net_pnl",
        selected=("microprice_deviation", 0.15),
        selected_metrics={"validation_net_pnl": 1.25, "test_net_pnl": 0.5},
        attempt_metrics={
            ("microprice_deviation", 0.05): {"validation_net_pnl": 0.75, "test_net_pnl": -0.25},
            ("microprice_deviation", 0.15): {"validation_net_pnl": 1.25, "test_net_pnl": 0.5},
            ("trade_imbalance", 0.05): {"validation_net_pnl": -0.1, "test_net_pnl": 0.0},
            ("trade_imbalance", 0.15): {"validation_net_pnl": -0.2, "test_net_pnl": -0.3},
        },
        artifact_path="results/walk_forward.csv",
        data_path="data/features.csv",
        holdout_manifest_path="data/holdout.json",
        horizon_ms=5000,
        latency_ms=1000,
        taker_fee_bps=0.1,
        random_seed=7,
        notes="fixture only",
    )
    loaded = read_experiment_registry(path)

    assert len(attempts) == 4
    assert loaded == attempts
    assert {attempt.feature for attempt in loaded} == {"microprice_deviation", "trade_imbalance"}
    assert {attempt.threshold for attempt in loaded} == {0.05, 0.15}
    selected = [attempt for attempt in loaded if attempt.selected]
    assert len(selected) == 1
    assert selected[0].status == "selected"
    assert selected[0].validation_net_pnl == 1.25
    assert selected[0].data_path == "data/features.csv"
    assert selected[0].holdout_manifest_path == "data/holdout.json"
    assert selected[0].config_sha256
    assert all(attempt.status != "planned" for attempt in loaded)
    assert all(attempt.validation_net_pnl is not None for attempt in loaded)
    assert all(attempt.test_net_pnl is not None for attempt in loaded)


def test_threshold_experiment_registry_records_failed_attempt_reason(tmp_path: Path) -> None:
    path = tmp_path / "experiment_registry.jsonl"

    attempts = write_threshold_experiment_registry(
        path,
        run_id="fixture-run",
        model_class="threshold_rule",
        features=["microprice_deviation"],
        thresholds=[0.05],
        selection_metric="validation_net_pnl",
        attempt_failures={("microprice_deviation", 0.05): "no validation rows"},
    )

    assert len(attempts) == 1
    assert attempts[0].status == "failed"
    assert attempts[0].failure_reason == "no validation rows"
