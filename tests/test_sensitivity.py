from pathlib import Path

from lob_forge.sensitivity import format_sensitivity_points, run_latency_fee_grid


def test_latency_fee_grid_runs_from_feature_rows(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    path.write_text(
        "event_time,bid,ask,microprice_deviation\n"
        "0,100,100.1,0.2\n"
        "1000,100.2,100.3,0.2\n"
        "2000,100.4,100.5,0.2\n"
        "3000,100.6,100.7,0.2\n"
        "4000,100.8,100.9,0.2\n"
    )

    points = run_latency_fee_grid(
        path,
        latencies_ms=[0, 1000],
        fees_bps=[0.0, 1.0],
        threshold=0.1,
        horizon_ms=1000,
    )
    output = format_sensitivity_points(points)

    assert len(points) == 4
    assert points[0].trades > 0
    assert points[1].net_pnl < points[0].net_pnl
    assert output.startswith("latency_ms,fee_bps")
