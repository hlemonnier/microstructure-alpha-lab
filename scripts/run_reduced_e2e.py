from __future__ import annotations

import csv
import json
import os
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from lob_forge.baselines import (
    WalkForwardFoldResult,
    evaluate_threshold_grid_on_splits,
    format_walk_forward_results,
    predict_feature_threshold,
    run_walk_forward_thresholds,
)
from lob_forge.experiment_registry import write_threshold_experiment_registry
from lob_forge.execution_sim import (
    FillLedgerRow,
    MarketEvent,
    PositionLedgerRow,
    SignalEvent,
    StatefulExecutionConfig,
    simulate_stateful_execution,
)
from lob_forge.holdout import GIT_COMMIT_RE, build_holdout_manifest, write_development_csv, write_holdout_manifest
from lob_forge.ml_models import run_l2_torch_sequence_experiment
from lob_forge.statistics import (
    break_even_cost_interval,
    day_level_mean_interval,
    newey_west_standard_error,
    sharpe_like_interval,
)


ROOT = Path(__file__).resolve().parents[1]
FEATURE_FIXTURE = ROOT / "examples" / "fixtures" / "feature_fixture.csv"
L2_SEQUENCE_FIXTURE = ROOT / "examples" / "fixtures" / "l2_sequence_fixture.csv"
OUT = ROOT / "artifacts" / "reduced_e2e"
TRAIN_SIZE = 4
VALIDATION_SIZE = 2
TEST_SIZE = 2
STEP_SIZE = 2
INITIAL_CASH = 1000.0
SOURCE_GIT_COMMIT_ENV = "LOB_FORGE_SOURCE_GIT_COMMIT"
SOURCE_ARCHIVE_COMMIT_FILE = ".source-git-commit"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    source_git_commit = _git_commit()
    source_working_tree_dirty = _working_tree_dirty()
    classical_path = OUT / "classical_walk_forward.csv"
    holdout_path = OUT / "holdout_manifest.json"
    development_fixture_path = OUT / "development_feature_fixture.csv"
    experiment_registry_path = OUT / "experiment_registry.jsonl"
    orders_path = OUT / "stateful_orders.csv"
    fills_path = OUT / "stateful_fills.csv"
    positions_path = OUT / "stateful_positions.csv"
    manifest_path = OUT / "result_manifest.json"
    baseline_audit_path = OUT / "baseline_audit_fixture.csv"
    sequence_tcn_path = OUT / "sequence_tcn_smoke.csv"
    sequence_transformer_path = OUT / "sequence_transformer_smoke.csv"
    sequence_checkpoint_dir = OUT / "sequence_checkpoints"
    sequence_prediction_dir = OUT / "sequence_predictions"
    sequence_l2_holdout_path = OUT / "l2_sequence_holdout_manifest.json"
    sequence_development_l2_path = OUT / "development_l2_sequence_fixture.csv"
    research_manifest_path = ROOT / "artifacts" / "research_manifest.json"
    _write_dataclass_csv(
        baseline_audit_path,
        [{"fold_count": 1, "acceptance_passed": 1, "rejection_reasons": ""}],
    )

    holdout_manifest = build_holdout_manifest(
        FEATURE_FIXTURE,
        split_column="source_date",
        holdout_values=["2026-01-03"],
        created_at_utc="2026-06-24T00:00:00Z",
        feature_version="synthetic_fixture_v1",
        target_version="fixture_mid_move_v1",
        git_commit=source_git_commit,
        notes="Synthetic fixture for CI/reduced pipeline only; not empirical evidence.",
        source_root=ROOT,
    )
    holdout_path.unlink(missing_ok=True)
    write_holdout_manifest(holdout_manifest, holdout_path)
    development_csv = write_development_csv(FEATURE_FIXTURE, holdout_manifest, development_fixture_path)
    sequence_l2_holdout_manifest = build_holdout_manifest(
        L2_SEQUENCE_FIXTURE,
        split_column="exchange_timestamp",
        holdout_values=["1140"],
        created_at_utc="2026-06-24T00:00:00Z",
        feature_version="l2_sequence_fixture_v1",
        target_version="fixture_l2_delta_v1",
        git_commit=source_git_commit,
        notes="Synthetic L2 fixture holdout for CI/reduced neural pipeline only; not empirical evidence.",
        source_root=ROOT,
    )
    sequence_l2_holdout_path.unlink(missing_ok=True)
    write_holdout_manifest(sequence_l2_holdout_manifest, sequence_l2_holdout_path)
    sequence_development_l2 = write_development_csv(
        L2_SEQUENCE_FIXTURE,
        sequence_l2_holdout_manifest,
        sequence_development_l2_path,
    )
    selected_features = ["microprice_deviation", "trade_imbalance"]
    selected_thresholds = [0.05, 0.15, 0.25]
    folds = run_walk_forward_thresholds(
        development_fixture_path,
        train_size=TRAIN_SIZE,
        validation_size=VALIDATION_SIZE,
        test_size=TEST_SIZE,
        step_size=STEP_SIZE,
        features=selected_features,
        thresholds=selected_thresholds,
        taker_fee_bps=1.0,
        slippage_bps=0.1,
        sort_by="validation_net_pnl",
    )
    classical_path.write_text(format_walk_forward_results(folds) + "\n")
    selected_result = folds[0].result
    selected = (
        (selected_result.feature, selected_result.threshold) if selected_result.feature in selected_features else None
    )
    development_rows = _read_csv_rows(development_fixture_path)
    train_rows = development_rows[:TRAIN_SIZE]
    validation_rows = development_rows[TRAIN_SIZE : TRAIN_SIZE + VALIDATION_SIZE]
    test_rows = development_rows[TRAIN_SIZE + VALIDATION_SIZE : TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE]
    evaluated_grid = evaluate_threshold_grid_on_splits(
        train_rows=train_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
        features=selected_features,
        thresholds=selected_thresholds,
        taker_fee_bps=1.0,
        slippage_bps=0.1,
        sort_by="validation_net_pnl",
    )
    attempt_metrics = {
        (result.feature, result.threshold): {
            "validation_net_pnl": result.validation_economics.net_pnl,
            "test_net_pnl": result.test_economics.net_pnl,
        }
        for result in evaluated_grid
    }
    registry_attempts = write_threshold_experiment_registry(
        experiment_registry_path,
        run_id="reduced_e2e_fixture",
        model_class="threshold_rule",
        features=selected_features,
        thresholds=selected_thresholds,
        selection_metric="validation_net_pnl",
        selected=selected,
        selected_metrics={
            "validation_net_pnl": selected_result.validation_economics.net_pnl,
            "test_net_pnl": selected_result.test_economics.net_pnl,
        },
        attempt_metrics=attempt_metrics,
        artifact_path=str(classical_path.relative_to(ROOT)),
        data_path=str(development_fixture_path.relative_to(ROOT)),
        holdout_manifest_path=str(holdout_path.relative_to(ROOT)),
        horizon_ms=600,
        latency_ms=100,
        taker_fee_bps=1.0,
        random_seed=7,
        extra_config={"features": selected_features, "thresholds": selected_thresholds},
        notes="Synthetic fixture registry of the evaluated threshold grid and selected validation candidate.",
    )

    simulation = simulate_stateful_execution(
        _market_events_from_rows(development_rows),
        _signals_from_walk_forward(development_rows, folds),
        config=StatefulExecutionConfig(
            initial_cash=INITIAL_CASH,
            max_position_notional=200.0,
            max_leverage=1.0,
            taker_fee_bps=1.0,
            slippage_bps=0.1,
            latency_ms=100,
            kill_switch_loss=100.0,
        ),
    )
    _write_dataclass_csv(orders_path, [asdict(row) for row in simulation.orders])
    _write_dataclass_csv(fills_path, [asdict(row) for row in simulation.fills])
    _write_dataclass_csv(positions_path, [asdict(row) for row in simulation.positions])

    timestamp_days = _timestamp_days(development_rows)
    net_pnls, net_returns, days, gross_pnls, turnovers = _ledger_statistics_inputs(
        simulation.positions,
        simulation.fills,
        timestamp_days=timestamp_days,
        initial_cash=INITIAL_CASH,
    )
    statistics = {
        "newey_west_net_pnl_se": newey_west_standard_error(net_pnls),
        "day_level_mean": asdict(day_level_mean_interval(net_pnls, days, samples=50)),
        "sharpe_like": asdict(sharpe_like_interval(net_returns, block_size=1, samples=50)),
        "break_even_cost": asdict(break_even_cost_interval(gross_pnls, turnovers, block_size=1, samples=50))
        if turnovers
        else None,
    }
    sequence_smokes = _run_sequence_smokes(
        baseline_audit_path=baseline_audit_path,
        sequence_tcn_path=sequence_tcn_path,
        sequence_transformer_path=sequence_transformer_path,
        checkpoint_dir=sequence_checkpoint_dir,
        prediction_dir=sequence_prediction_dir,
        holdout_manifest_path=sequence_l2_holdout_path,
        development_l2_path=sequence_development_l2_path,
    )

    manifest = {
        "artifact_version": 1,
        "fixture": str(FEATURE_FIXTURE.relative_to(ROOT)),
        "development_fixture": str(development_fixture_path.relative_to(ROOT)),
        "experiment_registry": str(experiment_registry_path.relative_to(ROOT)),
        "registered_candidate_count": len(registry_attempts),
        "classical_walk_forward": str(classical_path.relative_to(ROOT)),
        "holdout_manifest": str(holdout_path.relative_to(ROOT)),
        "stateful_orders": str(orders_path.relative_to(ROOT)),
        "stateful_fills": str(fills_path.relative_to(ROOT)),
        "stateful_positions": str(positions_path.relative_to(ROOT)),
        "baseline_audit_fixture": str(baseline_audit_path.relative_to(ROOT)),
        "l2_replay_fixture": "examples/fixtures/l2_replay_fixture.csv",
        "l2_sequence_fixture": str(L2_SEQUENCE_FIXTURE.relative_to(ROOT)),
        "l2_sequence_holdout_manifest": str(sequence_l2_holdout_path.relative_to(ROOT)),
        "l2_sequence_development_fixture": str(sequence_development_l2.path.relative_to(ROOT)),
        "l2_sequence_source_rows_before_holdout_filter": sequence_development_l2.source_rows,
        "l2_sequence_development_rows_after_holdout_filter": sequence_development_l2.development_rows,
        "l2_sequence_holdout_rows_excluded": sequence_development_l2.excluded_holdout_rows,
        "sequence_smokes": sequence_smokes,
        "source_rows_before_holdout_filter": development_csv.source_rows,
        "development_rows_after_holdout_filter": development_csv.development_rows,
        "holdout_rows_excluded": development_csv.excluded_holdout_rows,
        "stateful_final_equity": simulation.final_equity,
        "stateful_turnover": simulation.turnover,
        "statistics": statistics,
        "claim_scope": "Synthetic reduced E2E fixture only; no live or historical profitability claim.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    research_manifest = {
        "artifact_version": 1,
        "claim_scope": "Repository verification and synthetic fixture evidence only; no final empirical profitability claim.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit": source_git_commit,
        "working_tree_dirty": source_working_tree_dirty,
        "spec_traceability": "IMPLEMENTATION_TRACEABILITY.md",
        "research_note": "docs/research_note.md",
        "reproducibility": "docs/reproducibility.md",
        "reduced_e2e_manifest": str(manifest_path.relative_to(ROOT)),
        "reduced_e2e_artifacts": {
            "classical_walk_forward": str(classical_path.relative_to(ROOT)),
            "holdout_manifest": str(holdout_path.relative_to(ROOT)),
            "experiment_registry": str(experiment_registry_path.relative_to(ROOT)),
            "stateful_orders": str(orders_path.relative_to(ROOT)),
            "stateful_fills": str(fills_path.relative_to(ROOT)),
            "stateful_positions": str(positions_path.relative_to(ROOT)),
            "sequence_tcn_smoke": sequence_smokes.get("sequence_tcn", {}).get("path", ""),
            "sequence_transformer_smoke": sequence_smokes.get("sequence_transformer", {}).get("path", ""),
            "sequence_tcn_predictions": sequence_smokes.get("sequence_tcn", {}).get("predictions", ""),
            "sequence_transformer_predictions": sequence_smokes.get("sequence_transformer", {}).get("predictions", ""),
            "l2_sequence_holdout_manifest": str(sequence_l2_holdout_path.relative_to(ROOT)),
            "l2_sequence_development_fixture": str(sequence_development_l2.path.relative_to(ROOT)),
        },
        "final_holdout": {
            "status": "not_run",
            "reason": "No predeclared full-data final holdout evaluation was executed in this local pass.",
        },
        "empirical_blockers": [
            "Full multi-month, multi-asset confirmatory run requires external data/cloud compute.",
            "Genuine crypto multi-level L2 neural experiments require sufficient replay-grade L2 coverage.",
            "Paper/live fill validation requires observed fills from a shadow or paper trading session.",
        ],
    }
    research_manifest_path.write_text(json.dumps(research_manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)
    return 0


def _write_dataclass_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("\n")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _market_events_from_rows(rows: list[dict[str, str]]) -> list[MarketEvent]:
    events: list[MarketEvent] = []
    for row in rows:
        events.append(
            MarketEvent(
                int(float(row["event_time"])),
                bid=float(row["bid"]),
                ask=float(row["ask"]),
                bid_size=2.0,
                ask_size=2.0,
            )
        )
        events.append(
            MarketEvent(
                int(float(row["future_event_time"])),
                bid=float(row["future_bid"]),
                ask=float(row["future_ask"]),
                bid_size=2.0,
                ask_size=2.0,
            )
        )
    return events


def _signals_from_walk_forward(rows: list[dict[str, str]], folds: list[WalkForwardFoldResult]) -> list[SignalEvent]:
    signals: list[SignalEvent] = []
    total_window = TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE
    for fold in folds:
        start = (fold.fold - 1) * STEP_SIZE
        test_rows = rows[start + TRAIN_SIZE + VALIDATION_SIZE : start + total_window]
        signals.extend(_signals_from_selected_rule(test_rows, fold, offset=len(signals)))
    return signals


def _signals_from_selected_rule(
    rows: list[dict[str, str]],
    fold: WalkForwardFoldResult,
    *,
    offset: int = 0,
) -> list[SignalEvent]:
    result = fold.result
    signals: list[SignalEvent] = []
    for index, row in enumerate(rows, start=1 + offset):
        if result.feature == "constant":
            side = 0
        else:
            side = predict_feature_threshold(row, result.feature, result.threshold)
        signals.append(
            SignalEvent(
                int(float(row["event_time"])),
                target_side=side,
                target_notional=100.0 if side else 0.0,
                signal_id=f"selected-rule-oos-{index}",
                predicted_edge_bps=abs(float(row[result.feature])) if result.feature in row else 0.0,
            )
        )
    return signals


def _timestamp_days(rows: list[dict[str, str]]) -> dict[int, str]:
    days: dict[int, str] = {}
    for row in rows:
        day = row["source_date"]
        days[int(float(row["event_time"]))] = day
        days[int(float(row["future_event_time"]))] = day
    return days


def _ledger_statistics_inputs(
    positions: list[PositionLedgerRow],
    fills: list[FillLedgerRow],
    *,
    timestamp_days: dict[int, str],
    initial_cash: float,
) -> tuple[list[float], list[float], list[str], list[float], list[float]]:
    net_pnls: list[float] = []
    net_returns: list[float] = []
    days: list[str] = []
    gross_pnls: list[float] = []
    turnovers: list[float] = []
    fees_by_time: dict[int, float] = {}
    for fill in fills:
        fees_by_time[fill.fill_time_ms] = fees_by_time.get(fill.fill_time_ms, 0.0) + fill.fee

    previous_equity = initial_cash
    previous_turnover = 0.0
    for position in positions:
        net_pnl = position.equity - previous_equity
        turnover_delta = position.turnover - previous_turnover
        if abs(net_pnl) <= 1e-12 and turnover_delta <= 1e-12:
            previous_equity = position.equity
            previous_turnover = position.turnover
            continue
        fee_delta = fees_by_time.get(position.timestamp_ms, 0.0)
        net_pnls.append(net_pnl)
        net_returns.append(net_pnl / initial_cash)
        days.append(timestamp_days.get(position.timestamp_ms, "unknown"))
        gross_pnls.append(net_pnl + fee_delta)
        turnovers.append(turnover_delta)
        previous_equity = position.equity
        previous_turnover = position.turnover
    if not net_pnls:
        net_pnls = [0.0]
        net_returns = [0.0]
        days = ["unknown"]
    return net_pnls, net_returns, days, gross_pnls, turnovers


def _run_sequence_smokes(
    *,
    baseline_audit_path: Path,
    sequence_tcn_path: Path,
    sequence_transformer_path: Path,
    checkpoint_dir: Path,
    prediction_dir: Path,
    holdout_manifest_path: Path,
    development_l2_path: Path,
) -> dict[str, dict[str, object]]:
    outputs: dict[str, dict[str, object]] = {}
    for model_name, output_path in (
        ("sequence_tcn", sequence_tcn_path),
        ("sequence_transformer", sequence_transformer_path),
    ):
        checkpoint_path = checkpoint_dir / f"{model_name}.pt"
        prediction_path = prediction_dir / f"{model_name}_predictions.csv"
        try:
            report = run_l2_torch_sequence_experiment(
                model_name=model_name,
                l2_path=L2_SEQUENCE_FIXTURE,
                baseline_audit_path=baseline_audit_path,
                output_path=output_path,
                depth=1,
                window=3,
                label_horizon=1,
                epochs=1,
                min_fold_count=1,
                min_l2_rows=10,
                max_rows=100,
                max_snapshots=50,
                batch_size=2,
                class_weighting="balanced",
                checkpoint_path=checkpoint_path,
                prediction_output_path=prediction_path,
                holdout_manifest_path=holdout_manifest_path,
                development_l2_output_path=development_l2_path,
            )
        except RuntimeError as exc:
            output_path.unlink(missing_ok=True)
            checkpoint_path.unlink(missing_ok=True)
            prediction_path.unlink(missing_ok=True)
            outputs[model_name] = {
                "status": "skipped",
                "reason": str(exc),
                "path": str(output_path.relative_to(ROOT)),
                "checkpoint": str(checkpoint_path.relative_to(ROOT)),
                "predictions": str(prediction_path.relative_to(ROOT)),
            }
        else:
            outputs[model_name] = {
                "status": "pipeline_completed" if report.passed else "failed",
                "path": str(output_path.relative_to(ROOT)),
                "checkpoint": str(checkpoint_path.relative_to(ROOT)),
                "predictions": str(prediction_path.relative_to(ROOT)),
                "validation_macro_f1": report.validation_macro_f1,
                "test_macro_f1": report.test_macro_f1,
                "test_stateful_net_pnl": report.test_stateful_net_pnl,
                "purge_gap": report.purge_gap,
                "holdout_manifest": str(holdout_manifest_path.relative_to(ROOT)),
                "holdout_manifest_sha256": report.holdout_manifest_sha256,
                "development_l2": str(Path(report.development_l2_path).relative_to(ROOT)),
                "holdout_rows_excluded": report.holdout_rows_excluded,
            }
    return outputs


def _git_commit() -> str:
    env_commit = os.environ.get(SOURCE_GIT_COMMIT_ENV, "").strip()
    if env_commit:
        if not GIT_COMMIT_RE.fullmatch(env_commit):
            raise RuntimeError(f"{SOURCE_GIT_COMMIT_ENV} must be a 40- or 64-character Git commit hash")
        return env_commit
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        archive_commit = _source_archive_git_commit()
        if archive_commit:
            return archive_commit
        raise RuntimeError(
            "run_reduced_e2e requires Git metadata, a Git archive with expanded "
            f"{SOURCE_ARCHIVE_COMMIT_FILE}, or {SOURCE_GIT_COMMIT_ENV}=<commit-hash>"
        ) from None
    if not GIT_COMMIT_RE.fullmatch(commit):
        raise RuntimeError("git rev-parse HEAD did not return a valid commit hash")
    return commit


def _source_archive_git_commit() -> str | None:
    path = ROOT / SOURCE_ARCHIVE_COMMIT_FILE
    if not path.exists():
        return None
    value = path.read_text().strip()
    if not value or "$Format" in value:
        return None
    if not GIT_COMMIT_RE.fullmatch(value):
        raise RuntimeError(f"{SOURCE_ARCHIVE_COMMIT_FILE} must contain a 40- or 64-character Git commit hash")
    return value


def _working_tree_dirty() -> bool | None:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--short"],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
