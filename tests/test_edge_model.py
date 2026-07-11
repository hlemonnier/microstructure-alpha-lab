import io
from pathlib import Path
from contextlib import redirect_stdout

from lob_forge.cli import main as cli_main
from lob_forge.edge_model import (
    fit_edge_model,
    format_edge_walk_forward_results,
    predict_gross_edges_bps,
    run_edge_shadow_decisions_streaming,
    run_edge_walk_forward,
    run_edge_walk_forward_streaming,
    write_edge_shadow_decisions_streaming,
)
from lob_forge.holdout import build_holdout_manifest, write_holdout_manifest


def test_edge_model_predicts_long_and_short_gross_edges() -> None:
    rows = _synthetic_rows(30)
    model = fit_edge_model(rows, ["microprice_deviation", "top_imbalance"], l2=0.1)

    long_edge, short_edge = predict_gross_edges_bps(model, rows[0])
    down_long_edge, down_short_edge = predict_gross_edges_bps(model, rows[1])

    assert long_edge > short_edge
    assert down_short_edge > down_long_edge


def test_edge_walk_forward_runs_and_formats(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    _write_feature_csv(path, _synthetic_rows(36))

    folds = run_edge_walk_forward(
        path,
        features=["microprice_deviation", "top_imbalance"],
        edge_thresholds_bps=[0.0, 0.1],
        train_size=12,
        validation_size=8,
        test_size=8,
        step_size=8,
        purge_label_overlap=False,
        l2=0.1,
        taker_fee_bps=0.0,
    )
    output = format_edge_walk_forward_results(folds)
    lines = output.splitlines()

    assert len(folds) == 2
    assert folds[0].result.name == "ridge_expected_edge"
    assert folds[0].result.validation_economics.net_pnl > 0
    assert lines[0].startswith("fold,train_rows")
    assert lines[-1].startswith("summary,")
    assert len(lines[0].split(",")) == len(lines[-1].split(","))


def test_edge_walk_forward_can_select_flat_when_costs_dominate(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    _write_feature_csv(path, _synthetic_rows(36))

    folds = run_edge_walk_forward(
        path,
        features=["microprice_deviation", "top_imbalance"],
        edge_thresholds_bps=[0.0, 0.1],
        train_size=12,
        validation_size=8,
        test_size=8,
        step_size=8,
        purge_label_overlap=False,
        l2=0.1,
        taker_fee_bps=1000.0,
    )

    assert folds[0].result.name == "always_flat"


def test_edge_walk_forward_cli_streams_by_default(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    _write_feature_csv(path, _synthetic_rows(36))
    holdout_manifest = _write_holdout_manifest(tmp_path, path, holdout_event_time="35000")

    output = io.StringIO()
    with redirect_stdout(output):
        exit_code = cli_main(
            [
                "edge-walk-forward",
                str(path),
                "--holdout-manifest",
                str(holdout_manifest),
                "--features",
                "microprice_deviation,top_imbalance",
                "--edge-thresholds-bps",
                "0,0.1",
                "--train-size",
                "12",
                "--validation-size",
                "8",
                "--test-size",
                "8",
                "--step-size",
                "8",
                "--no-purge",
                "--l2",
                "0.1",
                "--taker-fee-bps",
                "0",
                "--max-load-memory-gb",
                "0.000000001",
            ]
        )

    assert exit_code == 0
    assert output.getvalue().startswith("fold,train_rows")


def test_edge_walk_forward_cli_no_stream_uses_memory_guard(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    _write_feature_csv(path, _synthetic_rows(36))
    holdout_manifest = _write_holdout_manifest(tmp_path, path, holdout_event_time="35000")

    try:
        cli_main(
            [
                "edge-walk-forward",
                str(path),
                "--holdout-manifest",
                str(holdout_manifest),
                "--features",
                "microprice_deviation,top_imbalance",
                "--edge-thresholds-bps",
                "0,0.1",
                "--train-size",
                "12",
                "--validation-size",
                "8",
                "--test-size",
                "8",
                "--step-size",
                "8",
                "--no-purge",
                "--l2",
                "0.1",
                "--taker-fee-bps",
                "0",
                "--max-load-memory-gb",
                "0.000000001",
                "--no-stream",
            ]
        )
    except ValueError as exc:
        assert "estimated CSV load memory exceeds budget" in str(exc)
    else:
        raise AssertionError("expected non-streaming CLI path to enforce the CSV memory guard")


def test_streaming_edge_walk_forward_matches_full_read(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    _write_feature_csv(path, _synthetic_rows(52))

    kwargs = {
        "features": ["microprice_deviation", "top_imbalance"],
        "edge_thresholds_bps": [0.0, 0.1, 0.2],
        "train_size": 12,
        "validation_size": 8,
        "test_size": 8,
        "step_size": 8,
        "purge_label_overlap": False,
        "l2": 0.1,
        "taker_fee_bps": 0.0,
    }

    full_output = format_edge_walk_forward_results(run_edge_walk_forward(path, **kwargs))
    streaming_output = format_edge_walk_forward_results(run_edge_walk_forward_streaming(path, **kwargs))

    assert streaming_output == full_output


def test_streaming_edge_walk_forward_handles_step_larger_than_window(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    _write_feature_csv(path, _synthetic_rows(80))

    kwargs = {
        "features": ["microprice_deviation", "top_imbalance"],
        "edge_thresholds_bps": [0.0, 0.1],
        "train_size": 10,
        "validation_size": 5,
        "test_size": 5,
        "step_size": 25,
        "purge_label_overlap": False,
        "l2": 0.1,
        "taker_fee_bps": 0.0,
    }

    full_output = format_edge_walk_forward_results(run_edge_walk_forward(path, **kwargs))
    streaming_output = format_edge_walk_forward_results(run_edge_walk_forward_streaming(path, **kwargs))

    assert streaming_output == full_output


def test_edge_walk_forward_can_cap_fold_count(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    _write_feature_csv(path, _synthetic_rows(52))

    folds = run_edge_walk_forward_streaming(
        path,
        features=["microprice_deviation", "top_imbalance"],
        edge_thresholds_bps=[0.0, 0.1],
        train_size=12,
        validation_size=8,
        test_size=8,
        step_size=8,
        purge_label_overlap=False,
        l2=0.1,
        taker_fee_bps=0.0,
        max_folds=2,
    )

    assert [fold.fold for fold in folds] == [1, 2]


def test_edge_walk_forward_rejects_overlapping_oos_test_windows(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    _write_feature_csv(path, _synthetic_rows(52))

    try:
        run_edge_walk_forward_streaming(
            path,
            features=["microprice_deviation", "top_imbalance"],
            train_size=12,
            validation_size=8,
            test_size=8,
            step_size=4,
            purge_label_overlap=False,
        )
    except ValueError as exc:
        assert "OOS test windows do not overlap" in str(exc)
    else:
        raise AssertionError("expected overlapping OOS windows to be rejected")


def test_edge_shadow_decisions_export_oos_rows(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    output = tmp_path / "shadow.csv"
    _write_feature_csv(path, _synthetic_rows(36))

    decisions = run_edge_shadow_decisions_streaming(
        path,
        venue="binance",
        symbol="BTCUSDT",
        intended_notional=100.0,
        order_type="paper_taker",
        include_flat=True,
        features=["microprice_deviation", "top_imbalance"],
        edge_thresholds_bps=[0.0, 0.1],
        train_size=12,
        validation_size=8,
        test_size=8,
        step_size=8,
        purge_label_overlap=False,
        l2=0.1,
        taker_fee_bps=0.0,
    )
    write_edge_shadow_decisions_streaming(
        output,
        path,
        venue="binance",
        symbol="BTCUSDT",
        intended_notional=100.0,
        order_type="paper_taker",
        include_flat=True,
        features=["microprice_deviation", "top_imbalance"],
        edge_thresholds_bps=[0.0, 0.1],
        train_size=12,
        validation_size=8,
        test_size=8,
        step_size=8,
        purge_label_overlap=False,
        l2=0.1,
        taker_fee_bps=0.0,
    )

    assert len(decisions) == 16
    assert decisions[0].decision_id.startswith("BTCUSDT-f1-")
    assert decisions[0].venue == "binance"
    assert decisions[0].symbol == "BTCUSDT"
    assert decisions[0].order_type == "paper_taker"
    assert decisions[0].intended_size > 0
    assert "threshold_bps=" in decisions[0].notes
    assert output.read_text().splitlines()[0].startswith("decision_id,timestamp_ms")


def _write_feature_csv(path: Path, rows: list[dict[str, str]]) -> None:
    columns = [
        "event_time",
        "future_event_time",
        "label",
        "microprice_deviation",
        "top_imbalance",
        "bid",
        "ask",
        "future_bid",
        "future_ask",
    ]
    with path.open("w") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join(row[column] for column in columns) + "\n")


def _write_holdout_manifest(tmp_path: Path, feature_csv: Path, *, holdout_event_time: str) -> Path:
    path = tmp_path / "holdout.json"
    manifest = build_holdout_manifest(
        feature_csv,
        split_column="event_time",
        holdout_values=[holdout_event_time],
        created_at_utc="2026-06-25T00:00:00Z",
        git_commit="a" * 40,
    )
    write_holdout_manifest(manifest, path)
    return path


def _synthetic_rows(n: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for idx in range(n):
        klass = idx % 3
        if klass == 0:
            label = 1
            value = 2.0
            future_bid = 101.0
            future_ask = 101.1
        elif klass == 1:
            label = -1
            value = -2.0
            future_bid = 99.0
            future_ask = 99.1
        else:
            label = 0
            value = 0.0
            future_bid = 100.0
            future_ask = 100.1
        rows.append(
            {
                "event_time": str(idx * 1000),
                "future_event_time": str(idx * 1000 + 500),
                "label": str(label),
                "microprice_deviation": str(value),
                "top_imbalance": str(value / 2.0),
                "bid": "100.0",
                "ask": "100.1",
                "future_bid": str(future_bid),
                "future_ask": str(future_ask),
            }
        )
    return rows
