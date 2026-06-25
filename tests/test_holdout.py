import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path

from lob_forge.cli import main as cli_main
from lob_forge.holdout import (
    assert_no_holdout_leakage,
    build_holdout_manifest,
    canonical_json_sha256,
    holdout_manifest_source_root,
    read_development_rows,
    read_holdout_rows,
    read_holdout_manifest,
    verify_holdout_manifest,
    verify_holdout_manifest_file,
    write_development_csv,
    write_final_holdout_result,
    write_holdout_manifest,
)


FIXTURE_GIT_COMMIT = "a" * 40


def test_holdout_manifest_is_hash_locked(tmp_path: Path) -> None:
    data = tmp_path / "features.csv"
    manifest_path = tmp_path / "holdout.json"
    data.write_text("source_date,label\n2026-06-01,1\n")

    manifest = build_holdout_manifest(
        data,
        split_column="source_date",
        holdout_values=["2026-06-01"],
        created_at_utc="2026-06-03T00:00:00Z",
        git_commit=FIXTURE_GIT_COMMIT,
    )
    write_holdout_manifest(manifest, manifest_path)

    assert read_holdout_manifest(manifest_path) == manifest
    assert verify_holdout_manifest(manifest)
    assert manifest.dataset_fingerprint == manifest.source_sha256
    assert manifest.date_range == ("2026-06-01", "2026-06-01")
    assert manifest.git_commit
    assert "git-commit" not in manifest.git_commit


def test_holdout_leakage_is_rejected(tmp_path: Path) -> None:
    data = tmp_path / "features.csv"
    data.write_text("source_date,label\n2026-06-01,1\n")
    manifest = build_holdout_manifest(
        data,
        split_column="source_date",
        holdout_values=["2026-06-01"],
        created_at_utc="2026-06-03T00:00:00Z",
        git_commit=FIXTURE_GIT_COMMIT,
    )

    try:
        assert_no_holdout_leakage(data, manifest, training_values=["2026-06-01"])
    except ValueError as exc:
        assert "overlaps" in str(exc)
    else:
        raise AssertionError("expected holdout overlap to be rejected")


def test_development_rows_filter_declared_holdout(tmp_path: Path) -> None:
    data = tmp_path / "features.csv"
    data.write_text("source_date,label\n2026-06-01,1\n2026-06-02,-1\n")
    manifest = build_holdout_manifest(
        data,
        split_column="source_date",
        holdout_values=["2026-06-02"],
        created_at_utc="2026-06-03T00:00:00Z",
        git_commit=FIXTURE_GIT_COMMIT,
    )

    rows = read_development_rows(data, manifest)

    assert [row["source_date"] for row in rows] == ["2026-06-01"]
    assert [row["source_date"] for row in read_holdout_rows(data, manifest)] == ["2026-06-02"]


def test_development_csv_is_physically_filtered_and_hash_locked(tmp_path: Path) -> None:
    data = tmp_path / "features.csv"
    development = tmp_path / "development.csv"
    other = tmp_path / "other.csv"
    data.write_text("source_date,label\n2026-06-01,1\n2026-06-02,-1\n")
    other.write_text("source_date,label\n2026-06-01,1\n2026-06-02,-1\n2026-06-03,0\n")
    manifest = build_holdout_manifest(
        data,
        split_column="source_date",
        holdout_values=["2026-06-02"],
        created_at_utc="2026-06-03T00:00:00Z",
        git_commit=FIXTURE_GIT_COMMIT,
    )

    result = write_development_csv(data, manifest, development)

    assert result.source_rows == 2
    assert result.development_rows == 1
    assert result.excluded_holdout_rows == 1
    assert development.read_text() == "source_date,label\n2026-06-01,1\n"
    try:
        write_development_csv(other, manifest, tmp_path / "bad.csv")
    except ValueError as exc:
        assert "content hash" in str(exc)
    else:
        raise AssertionError("expected source hash mismatch to be rejected")


def test_manifest_can_store_repository_relative_source_path(tmp_path: Path) -> None:
    data = tmp_path / "features.csv"
    data.write_text("source_date,label\n2026-06-01,1\n")
    manifest = build_holdout_manifest(
        data,
        split_column="source_date",
        holdout_values=["2026-06-01"],
        created_at_utc="2026-06-03T00:00:00Z",
        git_commit=FIXTURE_GIT_COMMIT,
        source_root=tmp_path,
    )

    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        assert manifest.source_path == "features.csv"
        assert verify_holdout_manifest(manifest)
    finally:
        os.chdir(previous_cwd)


def test_manifest_file_verification_resolves_source_path_from_manifest_ancestors(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    data_dir = repo / "examples" / "fixtures"
    manifest_dir = repo / "artifacts" / "holdout_manifests"
    data_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)
    data = data_dir / "features.csv"
    manifest_path = manifest_dir / "holdout.json"
    data.write_text("source_date,label\n2026-06-01,1\n")
    manifest = build_holdout_manifest(
        data,
        split_column="source_date",
        holdout_values=["2026-06-01"],
        created_at_utc="2026-06-03T00:00:00Z",
        git_commit=FIXTURE_GIT_COMMIT,
        source_root=repo,
    )
    write_holdout_manifest(manifest, manifest_path)

    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        assert not verify_holdout_manifest(read_holdout_manifest(manifest_path))
        assert verify_holdout_manifest_file(manifest_path)
        assert holdout_manifest_source_root(manifest_path) == repo.resolve()
    finally:
        os.chdir(previous_cwd)


def test_manifest_verification_rejects_non_hash_git_commit(tmp_path: Path) -> None:
    data = tmp_path / "features.csv"
    data.write_text("source_date,label\n2026-06-01,1\n")
    manifest = build_holdout_manifest(
        data,
        split_column="source_date",
        holdout_values=["2026-06-01"],
        created_at_utc="2026-06-03T00:00:00Z",
        git_commit="test-fixture-commit",
    )

    assert not verify_holdout_manifest(manifest)


def test_create_holdout_manifest_cli_writes_hash_locked_manifest(tmp_path: Path) -> None:
    data = tmp_path / "features.csv"
    manifest_path = tmp_path / "holdout.json"
    data.write_text("source_date,label\n2026-06-01,1\n2026-06-02,-1\n")

    output = io.StringIO()
    with redirect_stdout(output):
        code = cli_main(
            [
                "create-holdout-manifest",
                str(data),
                "--output",
                str(manifest_path),
                "--split-column",
                "source_date",
                "--holdout-values",
                "2026-06-02",
                "--git-commit",
                FIXTURE_GIT_COMMIT,
            ]
        )

    manifest = read_holdout_manifest(manifest_path)
    assert code == 0
    assert output.getvalue().startswith("holdout_manifest=")
    assert verify_holdout_manifest(manifest)
    assert manifest.holdout_values == ("2026-06-02",)


def test_final_holdout_result_requires_explicit_flag_and_is_immutable(tmp_path: Path) -> None:
    data = tmp_path / "features.csv"
    output = tmp_path / "holdout_result.json"
    data.write_text("source_date,label\n2026-06-01,1\n")
    manifest = build_holdout_manifest(
        data,
        split_column="source_date",
        holdout_values=["2026-06-01"],
        created_at_utc="2026-06-03T00:00:00Z",
        git_commit=FIXTURE_GIT_COMMIT,
    )

    try:
        write_final_holdout_result(
            manifest=manifest,
            metrics={"macro_f1": 0.0},
            output_path=output,
            explicit_final_evaluation=False,
        )
    except PermissionError:
        pass
    else:
        raise AssertionError("expected explicit final evaluation gate")

    write_final_holdout_result(
        manifest=manifest,
        metrics={"macro_f1": 0.0},
        output_path=output,
        explicit_final_evaluation=True,
    )
    try:
        write_final_holdout_result(
            manifest=manifest,
            metrics={"macro_f1": 0.1},
            output_path=output,
            explicit_final_evaluation=True,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected immutable holdout artifact")


def test_final_holdout_result_lock_blocks_same_manifest_candidate_pair(tmp_path: Path) -> None:
    data = tmp_path / "features.csv"
    output = tmp_path / "holdout_result.json"
    second_output = tmp_path / "holdout_result_second_path.json"
    data.write_text("source_date,label\n2026-06-01,1\n")
    manifest = build_holdout_manifest(
        data,
        split_column="source_date",
        holdout_values=["2026-06-01"],
        created_at_utc="2026-06-03T00:00:00Z",
        git_commit=FIXTURE_GIT_COMMIT,
    )

    write_final_holdout_result(
        manifest=manifest,
        metrics={"macro_f1": 0.0},
        output_path=output,
        explicit_final_evaluation=True,
        candidate_sha256="b" * 64,
        lock_dir=tmp_path / "locks",
    )
    try:
        write_final_holdout_result(
            manifest=manifest,
            metrics={"macro_f1": 0.1},
            output_path=second_output,
            explicit_final_evaluation=True,
            candidate_sha256="b" * 64,
            lock_dir=tmp_path / "locks",
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected manifest/candidate lock to block second final evaluation")


def test_final_holdout_rule_cli_consumes_frozen_candidate_once(tmp_path: Path) -> None:
    data = tmp_path / "features.csv"
    manifest_path = tmp_path / "holdout.json"
    candidate_path = tmp_path / "candidate.json"
    output_path = tmp_path / "final_holdout.json"
    data.write_text(
        "source_date,label,microprice_deviation,bid,ask,future_bid,future_ask\n"
        "2026-06-01,1,0.5,100.0,100.1,100.4,100.5\n"
        "2026-06-02,-1,-0.5,100.0,100.1,99.6,99.7\n"
    )
    candidate_path.write_text('{"feature":"microprice_deviation","threshold":0.1,"taker_fee_bps":0.0}\n')
    candidate_sha256 = canonical_json_sha256(json.loads(candidate_path.read_text()))
    manifest = build_holdout_manifest(
        data,
        split_column="source_date",
        holdout_values=["2026-06-02"],
        created_at_utc="2026-06-03T00:00:00Z",
        git_commit=FIXTURE_GIT_COMMIT,
        candidate_sha256=candidate_sha256,
    )
    write_holdout_manifest(manifest, manifest_path)

    output = io.StringIO()
    with redirect_stdout(output):
        code = cli_main(
            [
                "final-holdout-rule",
                str(data),
                "--holdout-manifest",
                str(manifest_path),
                "--candidate-json",
                str(candidate_path),
                "--output",
                str(output_path),
                "--lock-dir",
                str(tmp_path / "locks"),
                "--explicit-final-evaluation",
            ]
        )

    assert code == 0
    assert output.getvalue().startswith("final_holdout_result=")
    payload = json.loads(output_path.read_text())
    assert payload["final_evaluation"] is True
    assert payload["metrics"]["rows"] == 1
    assert payload["metrics"]["stateful_simulator"] is True
    assert payload["candidate_sha256"] == payload["metrics"]["candidate_sha256"]


def test_final_holdout_rule_cli_requires_pre_registered_candidate_hash(tmp_path: Path) -> None:
    data = tmp_path / "features.csv"
    manifest_path = tmp_path / "holdout.json"
    candidate_path = tmp_path / "candidate.json"
    output_path = tmp_path / "final_holdout.json"
    data.write_text(
        "source_date,label,microprice_deviation,bid,ask,future_bid,future_ask\n"
        "2026-06-01,1,0.5,100.0,100.1,100.4,100.5\n"
        "2026-06-02,-1,-0.5,100.0,100.1,99.6,99.7\n"
    )
    manifest = build_holdout_manifest(
        data,
        split_column="source_date",
        holdout_values=["2026-06-02"],
        created_at_utc="2026-06-03T00:00:00Z",
        git_commit=FIXTURE_GIT_COMMIT,
    )
    write_holdout_manifest(manifest, manifest_path)
    candidate_path.write_text('{"feature":"microprice_deviation","threshold":0.1,"taker_fee_bps":0.0}\n')

    try:
        cli_main(
            [
                "final-holdout-rule",
                str(data),
                "--holdout-manifest",
                str(manifest_path),
                "--candidate-json",
                str(candidate_path),
                "--output",
                str(output_path),
                "--lock-dir",
                str(tmp_path / "locks"),
                "--explicit-final-evaluation",
            ]
        )
    except ValueError as exc:
        assert "pre-registered candidate_sha256" in str(exc)
    else:
        raise AssertionError("expected final holdout to reject a manifest without candidate_sha256")


def test_final_holdout_rule_cli_resolves_relative_manifest_source_outside_repo_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    data_dir = repo / "data"
    manifest_dir = repo / "artifacts" / "holdout_manifests"
    run_dir = tmp_path / "run"
    data_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)
    run_dir.mkdir()
    data = data_dir / "features.csv"
    manifest_path = manifest_dir / "holdout.json"
    candidate_path = run_dir / "candidate.json"
    output_path = run_dir / "final_holdout.json"
    data.write_text(
        "source_date,label,microprice_deviation,bid,ask,future_bid,future_ask\n"
        "2026-06-01,1,0.5,100.0,100.1,100.4,100.5\n"
        "2026-06-02,-1,-0.5,100.0,100.1,99.6,99.7\n"
    )
    candidate_path.write_text('{"feature":"microprice_deviation","threshold":0.1,"taker_fee_bps":0.0}\n')
    candidate_sha256 = canonical_json_sha256(json.loads(candidate_path.read_text()))
    manifest = build_holdout_manifest(
        data,
        split_column="source_date",
        holdout_values=["2026-06-02"],
        created_at_utc="2026-06-03T00:00:00Z",
        git_commit=FIXTURE_GIT_COMMIT,
        source_root=repo,
        candidate_sha256=candidate_sha256,
    )
    write_holdout_manifest(manifest, manifest_path)

    previous_cwd = Path.cwd()
    output = io.StringIO()
    try:
        os.chdir(run_dir)
        with redirect_stdout(output):
            code = cli_main(
                [
                    "final-holdout-rule",
                    str(data),
                    "--holdout-manifest",
                    str(manifest_path),
                    "--candidate-json",
                    str(candidate_path),
                    "--output",
                    str(output_path),
                    "--lock-dir",
                    str(run_dir / "locks"),
                    "--explicit-final-evaluation",
                ]
            )
    finally:
        os.chdir(previous_cwd)

    assert code == 0
    assert output.getvalue().startswith("final_holdout_result=")
    payload = json.loads(output_path.read_text())
    assert payload["manifest"]["source_path"] == "data/features.csv"
    assert payload["metrics"]["stateful_simulator"] is True
