import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
from dataclasses import replace
import pytest

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
    reserve_final_holdout_evaluation,
    verify_final_holdout_result,
)


FIXTURE_GIT_COMMIT = "a" * 40


def test_final_holdout_reservation_blocks_overlapping_data_across_candidates_and_manifests(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("date,partition,label\nd1,first,1\nd2,second,-1\nd3,second,0\n")
    manifest = build_holdout_manifest(
        source, split_column="date", holdout_values=["d2"],
        created_at_utc="2026-09-07T00:00:00Z", git_commit=FIXTURE_GIT_COMMIT,
        candidate_sha256="a" * 64,
    )
    reserve_final_holdout_evaluation(manifest=manifest, candidate_sha256="a" * 64, explicit_final_evaluation=True)
    for altered in (
        replace(manifest, notes="different notes", candidate_sha256="b" * 64),
        replace(manifest, split_column="partition", holdout_values=("second",), candidate_sha256="b" * 64),
    ):
        with pytest.raises(FileExistsError, match="already reserved/evaluated"):
            reserve_final_holdout_evaluation(manifest=altered, candidate_sha256="b" * 64, explicit_final_evaluation=True)
    # A predeclared disjoint future segment can be evaluated separately.
    disjoint = replace(manifest, holdout_values=("d3",), candidate_sha256="c" * 64)
    reserve_final_holdout_evaluation(manifest=disjoint, candidate_sha256="c" * 64, explicit_final_evaluation=True)


def test_copying_the_same_data_within_a_study_does_not_reset_exposure(tmp_path: Path) -> None:
    sources = [tmp_path / name / "source.csv" for name in ("original", "copy")]
    for source in sources:
        source.parent.mkdir()
        source.write_text("date,label\nheldout,1\n")
    first = build_holdout_manifest(
        sources[0], split_column="date", holdout_values=["heldout"],
        created_at_utc="2026-09-07T00:00:00Z", git_commit=FIXTURE_GIT_COMMIT,
        candidate_sha256="a" * 64, source_root=tmp_path,
    )
    reserve_final_holdout_evaluation(
        manifest=first, candidate_sha256="a" * 64, explicit_final_evaluation=True, source_root=tmp_path,
    )
    second = replace(first, source_path="copy/source.csv", candidate_sha256="b" * 64)
    with pytest.raises(FileExistsError):
        reserve_final_holdout_evaluation(
            manifest=second, candidate_sha256="b" * 64, explicit_final_evaluation=True, source_root=tmp_path,
        )


def test_generated_feature_holdout_requires_current_semantics_and_build_provenance(tmp_path: Path) -> None:
    from lob_forge.features import FEATURE_SEMANTICS_VERSION
    from lob_forge.experiments import write_feature_build_marker
    source = tmp_path / "features.csv"
    source.write_text("date,bucket_start_ms,quote_updates_in_bucket,quote_ofi\nheldout,0,1,2\n")

    def manifest_for_source():
        return build_holdout_manifest(
            source, split_column="date", holdout_values=["heldout"],
            created_at_utc="2026-09-07T00:00:00Z", git_commit=FIXTURE_GIT_COMMIT,
            candidate_sha256="a" * 64,
        )

    with pytest.raises(ValueError, match="obsolete feature semantics"):
        reserve_final_holdout_evaluation(manifest=manifest_for_source(), candidate_sha256="a" * 64, explicit_final_evaluation=True)
    source.write_text("date,bucket_start_ms,quote_updates_in_bucket,quote_ofi,feature_semantics_version\n"
                      f"heldout,0,1,2,{FEATURE_SEMANTICS_VERSION}\n")
    with pytest.raises(ValueError, match="current build marker"):
        reserve_final_holdout_evaluation(manifest=manifest_for_source(), candidate_sha256="a" * 64, explicit_final_evaluation=True)
    write_feature_build_marker(source, source.with_suffix(".csv.done"), build_config={}, input_hashes={})
    token = reserve_final_holdout_evaluation(manifest=manifest_for_source(), candidate_sha256="a" * 64, explicit_final_evaluation=True)
    assert token["reservation_id"]


def test_final_holdout_verified_result_binds_source_candidate_and_completed_ledger(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("date,label\nholdout,1\n")
    candidate = {"feature": "signal", "threshold": 0.0}
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate))
    digest = canonical_json_sha256(candidate)
    manifest = build_holdout_manifest(
        source, split_column="date", holdout_values=["holdout"],
        created_at_utc="2026-09-07T00:00:00Z", git_commit=FIXTURE_GIT_COMMIT, candidate_sha256=digest,
    )
    token = reserve_final_holdout_evaluation(manifest=manifest, candidate_sha256=digest, explicit_final_evaluation=True)
    output = tmp_path / "result.json"
    write_final_holdout_result(
        manifest=manifest, candidate_sha256=digest, explicit_final_evaluation=True,
        output_path=output, candidate_path=candidate_path, reservation=token,
        metrics={"candidate": candidate, "candidate_sha256": digest, "net_pnl": -1.0},
    )
    assert verify_final_holdout_result(output) == (True, ())
    original = output.read_text()
    payload = json.loads(original)
    payload["metrics"]["net_pnl"] = 1000.0
    output.write_text(json.dumps(payload))
    assert not verify_final_holdout_result(output)[0]
    output.write_text(original)
    candidate_path.write_text(json.dumps({**candidate, "threshold": 2.0}))
    assert not verify_final_holdout_result(output)[0]


def test_holdout_schema_only_payload_is_not_evidence(tmp_path: Path) -> None:
    from lob_forge.evidence_gates import _final_holdout_result_gate_for_path
    output = tmp_path / "forged.json"
    output.write_text(json.dumps({
        "manifest": {"candidate_sha256": "a" * 64}, "candidate_sha256": "a" * 64,
        "final_evaluation": True,
        "metrics": {"candidate_sha256": "a" * 64, "rows": 1, "stateful_simulator": True},
    }))
    assert not _final_holdout_result_gate_for_path(result_path=output, todo="test").passed


def test_development_labels_are_purged_when_their_endpoints_cross_holdout(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text(
        "split,event_time,future_event_time,label\n"
        "train,0,8,1\ntrain,8,10,1\nholdout,10,20,-1\ntrain,20,21,0\n"
    )
    manifest = build_holdout_manifest(
        source, split_column="split", holdout_values=["holdout"],
        created_at_utc="2026-09-07T00:00:00Z", git_commit=FIXTURE_GIT_COMMIT,
    )
    rows = read_development_rows(source, manifest)
    assert [row["event_time"] for row in rows] == ["0", "20"]
    result = write_development_csv(source, manifest, tmp_path / "development.csv")
    assert result.purged_label_rows == 1
    assert result.source_rows == result.development_rows + result.excluded_holdout_rows
    with pytest.raises(ValueError, match="overwrite"):
        write_development_csv(source, manifest, source)


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


def test_final_holdout_result_requires_explicit_flag_candidate_hash_and_is_immutable(tmp_path: Path) -> None:
    data = tmp_path / "features.csv"
    output = tmp_path / "holdout_result.json"
    data.write_text("source_date,label\n2026-06-01,1\n")
    candidate_sha256 = "b" * 64
    manifest = build_holdout_manifest(
        data,
        split_column="source_date",
        holdout_values=["2026-06-01"],
        created_at_utc="2026-06-03T00:00:00Z",
        git_commit=FIXTURE_GIT_COMMIT,
        candidate_sha256=candidate_sha256,
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

    try:
        write_final_holdout_result(
            manifest=manifest,
            metrics={"macro_f1": 0.0},
            output_path=output,
            explicit_final_evaluation=True,
        )
    except ValueError as exc:
        assert "requires candidate_sha256" in str(exc)
    else:
        raise AssertionError("expected final holdout result to require a candidate hash")

    write_final_holdout_result(
        manifest=manifest,
        metrics={"macro_f1": 0.0},
        output_path=output,
        explicit_final_evaluation=True,
        candidate_sha256=candidate_sha256,
        lock_dir=tmp_path / "locks",
    )
    try:
        write_final_holdout_result(
            manifest=manifest,
            metrics={"macro_f1": 0.1},
            output_path=output,
            explicit_final_evaluation=True,
            candidate_sha256=candidate_sha256,
            lock_dir=tmp_path / "locks",
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected immutable holdout artifact path")

    try:
        write_final_holdout_result(
            manifest=manifest,
            metrics={"macro_f1": 0.1},
            output_path=output,
            explicit_final_evaluation=True,
            candidate_sha256="c" * 64,
            lock_dir=tmp_path / "locks",
        )
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("expected mismatched final holdout candidate hash to be rejected")


def test_final_holdout_result_requires_pre_registered_manifest_candidate_hash(tmp_path: Path) -> None:
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
            explicit_final_evaluation=True,
            candidate_sha256="b" * 64,
            lock_dir=tmp_path / "locks",
        )
    except ValueError as exc:
        assert "pre-register candidate_sha256" in str(exc)
    else:
        raise AssertionError("expected final holdout manifest to require a pre-registered candidate hash")


def test_final_holdout_result_lock_blocks_same_manifest_candidate_pair(tmp_path: Path) -> None:
    data = tmp_path / "features.csv"
    output = tmp_path / "holdout_result.json"
    second_output = tmp_path / "holdout_result_second_path.json"
    candidate_sha256 = "b" * 64
    data.write_text("source_date,label\n2026-06-01,1\n")
    manifest = build_holdout_manifest(
        data,
        split_column="source_date",
        holdout_values=["2026-06-01"],
        created_at_utc="2026-06-03T00:00:00Z",
        git_commit=FIXTURE_GIT_COMMIT,
        candidate_sha256=candidate_sha256,
    )

    write_final_holdout_result(
        manifest=manifest,
        metrics={"macro_f1": 0.0},
        output_path=output,
        explicit_final_evaluation=True,
        candidate_sha256=candidate_sha256,
        lock_dir=tmp_path / "locks",
    )
    try:
        write_final_holdout_result(
            manifest=manifest,
            metrics={"macro_f1": 0.1},
            output_path=second_output,
            explicit_final_evaluation=True,
            candidate_sha256=candidate_sha256,
            lock_dir=tmp_path / "locks",
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected manifest/candidate lock to block second final evaluation")


def test_final_holdout_default_lock_blocks_different_output_directories(tmp_path: Path) -> None:
    data = tmp_path / "features.csv"
    first_output = tmp_path / "first" / "holdout_result.json"
    second_output = tmp_path / "second" / "holdout_result.json"
    candidate_sha256 = "b" * 64
    data.write_text("source_date,label\n2026-06-01,1\n")
    manifest = build_holdout_manifest(
        data,
        split_column="source_date",
        holdout_values=["2026-06-01"],
        created_at_utc="2026-06-03T00:00:00Z",
        git_commit=FIXTURE_GIT_COMMIT,
        candidate_sha256=candidate_sha256,
    )

    write_final_holdout_result(
        manifest=manifest,
        metrics={"macro_f1": 0.0},
        output_path=first_output,
        explicit_final_evaluation=True,
        candidate_sha256=candidate_sha256,
    )

    try:
        write_final_holdout_result(
            manifest=manifest,
            metrics={"macro_f1": 0.1},
            output_path=second_output,
            explicit_final_evaluation=True,
            candidate_sha256=candidate_sha256,
        )
    except FileExistsError as exc:
        assert ".final_holdout_locks" in str(exc)
    else:
        raise AssertionError("expected default manifest/candidate lock to block another output directory")


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


def test_freeze_threshold_candidate_cli_selects_aggregate_candidate_for_final_holdout(tmp_path: Path) -> None:
    artifact_path = tmp_path / "threshold_walk_forward.csv"
    candidate_path = tmp_path / "candidate.json"
    data = tmp_path / "features.csv"
    manifest_path = tmp_path / "holdout.json"
    output_path = tmp_path / "final_holdout.json"
    artifact_path.write_text(
        "fold,feature,threshold,name,val_net_pnl,test_net_pnl,val_break_even_fee_bps,test_break_even_fee_bps\n"
        "1,microprice_deviation,0.1,microprice_deviation_threshold,10.0,1.0,0.2,0.1\n"
        "2,microprice_deviation,0.1,microprice_deviation_threshold,15.0,2.0,0.1,0.1\n"
        "3,trade_imbalance,0.2,trade_imbalance_threshold,20.0,-1.0,0.3,0.2\n"
        "summary,constant,0.0,always_flat,999.0,999.0,999.0,999.0\n"
    )
    data.write_text(
        "source_date,label,microprice_deviation,bid,ask,future_bid,future_ask\n"
        "2026-06-01,1,0.5,100.0,100.1,100.4,100.5\n"
        "2026-06-02,-1,-0.5,100.0,100.1,99.6,99.7\n"
    )

    output = io.StringIO()
    with redirect_stdout(output):
        code = cli_main(
            [
                "freeze-threshold-candidate",
                str(artifact_path),
                "--output",
                str(candidate_path),
                "--taker-fee-bps",
                "0",
                "--target-notional",
                "100",
            ]
        )

    candidate = json.loads(candidate_path.read_text())
    assert code == 0
    assert output.getvalue().startswith("candidate_json=")
    assert f"candidate_sha256={canonical_json_sha256(candidate)}" in output.getvalue()
    assert candidate["feature"] == "microprice_deviation"
    assert candidate["threshold"] == 0.1
    assert candidate["selection_metric"] == "validation_net_pnl"
    assert candidate["selection_score"] == 25.0
    assert candidate["selected_rows"] == 2
    assert candidate["validation_net_pnl"] == 25.0
    assert candidate["test_net_pnl"] == 3.0
    assert candidate["target_notional"] == 100.0

    manifest = build_holdout_manifest(
        data,
        split_column="source_date",
        holdout_values=["2026-06-02"],
        created_at_utc="2026-06-03T00:00:00Z",
        git_commit=FIXTURE_GIT_COMMIT,
        candidate_sha256=canonical_json_sha256(candidate),
    )
    write_holdout_manifest(manifest, manifest_path)

    with redirect_stdout(io.StringIO()):
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

    payload = json.loads(output_path.read_text())
    assert code == 0
    assert payload["metrics"]["candidate"]["source_artifact"] == str(artifact_path)
    assert payload["metrics"]["candidate"]["selection_grain"] == "fold_rows_aggregated_by_feature_threshold"
    assert payload["metrics"]["stateful_simulator"] is True


def test_freeze_threshold_candidate_cli_rejects_edge_model_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "edge_walk_forward.csv"
    candidate_path = tmp_path / "candidate.json"
    artifact_path.write_text("fold,edge_threshold_bps,val_net_pnl\n1,0.5,10.0\n")

    try:
        cli_main(
            [
                "freeze-threshold-candidate",
                str(artifact_path),
                "--output",
                str(candidate_path),
            ]
        )
    except ValueError as exc:
        assert "threshold-rule artifact" in str(exc)
    else:
        raise AssertionError("expected edge-model artifacts without feature/threshold columns to be rejected")


def test_freeze_edge_candidate_cli_selects_artifact_threshold_for_final_holdout(tmp_path: Path) -> None:
    data = tmp_path / "features.csv"
    development_manifest_path = tmp_path / "development_holdout.json"
    final_manifest_path = tmp_path / "final_holdout_manifest.json"
    artifact_path = tmp_path / "edge_walk_forward.csv"
    candidate_path = tmp_path / "edge_candidate.json"
    output_path = tmp_path / "edge_final_holdout.json"
    data.write_text(
        "source_date,event_time,future_event_time,label,microprice_deviation,bid,ask,future_bid,future_ask\n"
        "2026-06-01,1000,1100,1,1.0,100.0,100.1,101.0,101.1\n"
        "2026-06-01,2000,2100,-1,-1.0,100.0,100.1,99.0,99.1\n"
        "2026-06-02,3000,3100,1,0.8,100.0,100.1,100.8,100.9\n"
        "2026-06-03,4000,4100,1,1.2,100.0,100.1,101.2,101.3\n"
    )
    artifact_path.write_text(
        "fold,name,edge_threshold_bps,val_net_pnl,test_net_pnl,val_break_even_fee_bps,test_break_even_fee_bps\n"
        "1,ridge_expected_edge,0.0,1.0,0.5,0.2,0.1\n"
        "2,ridge_expected_edge,0.5,3.0,0.1,0.4,0.1\n"
        "3,ridge_expected_edge,0.5,2.0,0.2,0.3,0.1\n"
        "summary,,,,,,\n"
    )
    development_manifest = build_holdout_manifest(
        data,
        split_column="source_date",
        holdout_values=["2026-06-03"],
        created_at_utc="2026-06-03T00:00:00Z",
        git_commit=FIXTURE_GIT_COMMIT,
    )
    write_holdout_manifest(development_manifest, development_manifest_path)

    output = io.StringIO()
    with redirect_stdout(output):
        code = cli_main(
            [
                "freeze-edge-candidate",
                str(data),
                "--holdout-manifest",
                str(development_manifest_path),
                "--output",
                str(candidate_path),
                "--features",
                "microprice_deviation",
                "--walk-forward-artifact",
                str(artifact_path),
                "--taker-fee-bps",
                "0",
                "--target-notional",
                "100",
            ]
        )

    candidate = json.loads(candidate_path.read_text())
    assert code == 0
    assert output.getvalue().startswith("candidate_json=")
    assert candidate["candidate_type"] == "ridge_expected_edge_v1"
    assert candidate["features"] == ["microprice_deviation"]
    assert candidate["edge_threshold_bps"] == 0.5
    assert candidate["selection_score"] == 5.0
    assert candidate["selected_rows"] == 2
    assert len(candidate["long_weights"]) == 2
    assert len(candidate["short_weights"]) == 2

    final_manifest = build_holdout_manifest(
        data,
        split_column="source_date",
        holdout_values=["2026-06-03"],
        created_at_utc="2026-06-03T00:00:00Z",
        git_commit=FIXTURE_GIT_COMMIT,
        candidate_sha256=canonical_json_sha256(candidate),
    )
    write_holdout_manifest(final_manifest, final_manifest_path)

    with redirect_stdout(io.StringIO()):
        code = cli_main(
            [
                "final-holdout-edge",
                str(data),
                "--holdout-manifest",
                str(final_manifest_path),
                "--candidate-json",
                str(candidate_path),
                "--output",
                str(output_path),
                "--lock-dir",
                str(tmp_path / "locks"),
                "--explicit-final-evaluation",
            ]
        )

    payload = json.loads(output_path.read_text())
    assert code == 0
    assert payload["candidate_sha256"] == canonical_json_sha256(candidate)
    assert payload["metrics"]["candidate_type"] == "ridge_expected_edge_v1"
    assert payload["metrics"]["candidate"]["edge_threshold_bps"] == 0.5
    assert payload["metrics"]["rows"] == 1
    assert payload["metrics"]["stateful_simulator"] is True


def test_freeze_edge_candidate_cli_requires_threshold_or_artifact(tmp_path: Path) -> None:
    data = tmp_path / "features.csv"
    manifest_path = tmp_path / "holdout.json"
    candidate_path = tmp_path / "edge_candidate.json"
    data.write_text(
        "source_date,label,microprice_deviation,bid,ask,future_bid,future_ask\n"
        "2026-06-01,1,1.0,100.0,100.1,101.0,101.1\n"
        "2026-06-02,-1,-1.0,100.0,100.1,99.0,99.1\n"
    )
    manifest = build_holdout_manifest(
        data,
        split_column="source_date",
        holdout_values=["2026-06-02"],
        created_at_utc="2026-06-03T00:00:00Z",
        git_commit=FIXTURE_GIT_COMMIT,
    )
    write_holdout_manifest(manifest, manifest_path)

    try:
        cli_main(
            [
                "freeze-edge-candidate",
                str(data),
                "--holdout-manifest",
                str(manifest_path),
                "--output",
                str(candidate_path),
                "--features",
                "microprice_deviation",
            ]
        )
    except ValueError as exc:
        assert "--edge-threshold-bps or --walk-forward-artifact" in str(exc)
    else:
        raise AssertionError("expected edge candidate freezing to require a threshold source")


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
