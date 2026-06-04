from pathlib import Path

from lob_forge.holdout import (
    assert_no_holdout_leakage,
    build_holdout_manifest,
    read_holdout_manifest,
    verify_holdout_manifest,
    write_holdout_manifest,
)


def test_holdout_manifest_is_hash_locked(tmp_path: Path) -> None:
    data = tmp_path / "features.csv"
    manifest_path = tmp_path / "holdout.json"
    data.write_text("source_date,label\n2026-06-01,1\n")

    manifest = build_holdout_manifest(
        data,
        split_column="source_date",
        holdout_values=["2026-06-01"],
        created_at_utc="2026-06-03T00:00:00Z",
    )
    write_holdout_manifest(manifest, manifest_path)

    assert read_holdout_manifest(manifest_path) == manifest
    assert verify_holdout_manifest(manifest)


def test_holdout_leakage_is_rejected(tmp_path: Path) -> None:
    data = tmp_path / "features.csv"
    data.write_text("source_date,label\n2026-06-01,1\n")
    manifest = build_holdout_manifest(
        data,
        split_column="source_date",
        holdout_values=["2026-06-01"],
        created_at_utc="2026-06-03T00:00:00Z",
    )

    try:
        assert_no_holdout_leakage(data, manifest, training_values=["2026-06-01"])
    except ValueError as exc:
        assert "overlaps" in str(exc)
    else:
        raise AssertionError("expected holdout overlap to be rejected")
