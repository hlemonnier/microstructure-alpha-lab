from pathlib import Path

from lob_forge.experiments import (
    combined_feature_is_current,
    feature_build_is_current,
    write_combined_feature_manifest,
    write_feature_build_marker,
)


def test_feature_build_marker_binds_configuration_inputs_and_output(tmp_path: Path) -> None:
    feature = tmp_path / "features.csv"
    marker = tmp_path / "features.csv.done"
    feature.write_text("event_time,label\n1,0\n")
    config = {"horizon_ms": 5000, "max_quote_buckets": 3600}
    inputs = {"book_ticker_sha256": "a" * 64, "agg_trades_sha256": "b" * 64, "book_depth_sha256": None}

    write_feature_build_marker(feature, marker, build_config=config, input_hashes=inputs)

    assert feature_build_is_current(feature, marker, build_config=config, input_hashes=inputs)
    assert not feature_build_is_current(
        feature,
        marker,
        build_config={**config, "max_quote_buckets": 7200},
        input_hashes=inputs,
    )
    assert not feature_build_is_current(
        feature,
        marker,
        build_config=config,
        input_hashes={**inputs, "book_ticker_sha256": "c" * 64},
    )

    feature.write_text("event_time,label\n1,1\n")
    assert not feature_build_is_current(feature, marker, build_config=config, input_hashes=inputs)


def test_legacy_feature_done_marker_fails_closed(tmp_path: Path) -> None:
    feature = tmp_path / "features.csv"
    marker = tmp_path / "features.csv.done"
    feature.write_text("event_time,label\n1,0\n")
    marker.write_text("ok\n")

    assert not feature_build_is_current(
        feature,
        marker,
        build_config={"horizon_ms": 5000},
        input_hashes={"book_ticker_sha256": "a" * 64},
    )


def test_combined_feature_manifest_binds_ordered_daily_inputs_and_output(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    combined = tmp_path / "combined.csv"
    first.write_text("value\n1\n")
    second.write_text("value\n2\n")
    combined.write_text("value\n1\n2\n")

    write_combined_feature_manifest(combined, [first, second])

    assert combined_feature_is_current(combined, [first, second])
    assert not combined_feature_is_current(combined, [second, first])
    combined.write_text("value\n1\n3\n")
    assert not combined_feature_is_current(combined, [first, second])
