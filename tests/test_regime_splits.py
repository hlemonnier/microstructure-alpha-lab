from pathlib import Path

from lob_forge.regime_splits import format_regime_splits, run_regime_splits


def test_regime_splits_cover_required_regimes(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    path.write_text(
        "event_time,trade_notional,trade_count,mid_return_5\n"
        "1704067200000,100,1,0.0001\n"
        "1704099600000,10000,2,0.0010\n"
    )

    splits = run_regime_splits(path)
    regime_types = {split.regime_type for split in splits}
    output = format_regime_splits(splits)

    assert {"time_of_day", "weekday", "funding_window", "trend_chop", "stress_volume"} <= regime_types
    assert "stress_high_volume" in output
