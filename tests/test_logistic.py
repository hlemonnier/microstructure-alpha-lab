from pathlib import Path

from lob_forge.logistic import (
    fit_softmax_model,
    format_logistic_walk_forward_results,
    predict_probabilities,
    run_logistic_walk_forward,
)


def test_softmax_model_outputs_probabilities() -> None:
    rows = _synthetic_rows(24)
    model = fit_softmax_model(
        rows,
        ["microprice_deviation", "top_imbalance"],
        epochs=60,
        learning_rate=0.1,
        l2=0.001,
    )

    probabilities = predict_probabilities(model, rows[0])

    assert len(probabilities) == 3
    assert abs(sum(probabilities) - 1.0) < 1e-9
    assert max(probabilities) > 0.4


def test_logistic_walk_forward_runs_and_formats(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    with path.open("w") as handle:
        handle.write(
            "event_time,future_event_time,label,microprice_deviation,top_imbalance,bid,ask,future_bid,future_ask\n"
        )
        for row in _synthetic_rows(36):
            handle.write(
                ",".join(
                    [
                        row["event_time"],
                        row["future_event_time"],
                        row["label"],
                        row["microprice_deviation"],
                        row["top_imbalance"],
                        row["bid"],
                        row["ask"],
                        row["future_bid"],
                        row["future_ask"],
                    ]
                )
                + "\n"
            )

    folds = run_logistic_walk_forward(
        path,
        features=["microprice_deviation", "top_imbalance"],
        alpha_thresholds=[0.0, 0.1],
        train_size=12,
        validation_size=8,
        test_size=8,
        step_size=8,
        purge_label_overlap=False,
        epochs=60,
        learning_rate=0.1,
        taker_fee_bps=0.0,
    )
    output = format_logistic_walk_forward_results(folds)
    lines = output.splitlines()

    assert len(folds) == 2
    assert folds[0].result.name == "softmax_logistic"
    assert lines[0].startswith("fold,train_rows")
    assert lines[-1].startswith("summary,")
    assert len(lines[0].split(",")) == len(lines[-1].split(","))


def test_logistic_walk_forward_can_select_flat_when_costs_dominate(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    with path.open("w") as handle:
        handle.write(
            "event_time,future_event_time,label,microprice_deviation,top_imbalance,bid,ask,future_bid,future_ask\n"
        )
        for row in _synthetic_rows(36):
            handle.write(
                ",".join(
                    [
                        row["event_time"],
                        row["future_event_time"],
                        row["label"],
                        row["microprice_deviation"],
                        row["top_imbalance"],
                        row["bid"],
                        row["ask"],
                        row["future_bid"],
                        row["future_ask"],
                    ]
                )
                + "\n"
            )

    folds = run_logistic_walk_forward(
        path,
        features=["microprice_deviation", "top_imbalance"],
        alpha_thresholds=[0.0, 0.1],
        train_size=12,
        validation_size=8,
        test_size=8,
        step_size=8,
        purge_label_overlap=False,
        epochs=60,
        learning_rate=0.1,
        taker_fee_bps=1000.0,
    )

    assert folds[0].result.name == "always_flat"


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
