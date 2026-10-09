"""Build and verify the public project in disposable, isolated environments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from check_repository import public_files


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SOURCE_FILES = {
    "LICENSE",
    "README.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "MANIFEST.in",
    "pyproject.toml",
    "requirements-build.txt",
    "requirements-ci.txt",
    "requirements-research.txt",
    ".source-git-commit",
    "cpp/l2_replay.cpp",
    "examples/fixtures/feature_fixture.csv",
    "examples/fixtures/l2_replay_fixture.csv",
    "examples/fixtures/l2_sequence_fixture.csv",
    "examples/fixtures/l2_sequence_e2e_fixture.csv",
    "scripts/run_reduced_e2e.py",
    "scripts/run_public_e2e.py",
    "docs/README.md",
    "data/README.md",
}
REPLAY_REFERENCE = """
import csv, sys
from pathlib import Path
from lob_forge.data_sources import NormalizedL2Row
from lob_forge.l2_replay import replay_l2_rows
with Path(sys.argv[1]).open(newline='') as handle:
    rows = [NormalizedL2Row(
        event_type=r['event_type'], exchange_timestamp=int(r['exchange_timestamp']),
        local_timestamp=int(r['local_timestamp']), side=r['side'], price=float(r['price']),
        size=float(r['size']), sequence=int(r['sequence']) if r['sequence'] else None,
        update_id=int(r['update_id']) if r['update_id'] else None, venue=r['venue'], symbol=r['symbol'],
    ) for r in csv.DictReader(handle)]
_, updates, _ = replay_l2_rows(rows)
writer = csv.writer(sys.stdout, lineterminator='\\n')
writer.writerow(['row_index','sequence_gap','reset','crossed','best_bid','best_ask','needs_resnapshot'])
for u in updates:
    writer.writerow([u.row_index, int(u.sequence_gap), int(u.reset), int(u.crossed),
        '' if u.best_bid is None else format(u.best_bid,'g'),
        '' if u.best_ask is None else format(u.best_ask,'g'), int(u.needs_resnapshot)])
"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def inspect_archive(names: list[str]) -> None:
    files = {"/".join(Path(name).parts[1:]) for name in names if not name.endswith("/")}
    require(REQUIRED_SOURCE_FILES <= files, f"Source archive missing: {sorted(REQUIRED_SOURCE_FILES - files)}")
    for name in files:
        parts = Path(name).parts
        require(
            not any(p in {".git", ".venv", "__pycache__", "artifacts", "results", "build", "dist"} for p in parts),
            f"Generated/local path in source archive: {name}",
        )
        require(not name.startswith("data/") or name == "data/README.md", f"Market data in source archive: {name}")
        require(
            " 2." not in name and not name.endswith((".pyc", ".bak", ".tmp")), f"Local backup in source archive: {name}"
        )


def extract_source(archive: tarfile.TarFile, destination: Path) -> None:
    for member in archive.getmembers():
        require(member.isfile() or member.isdir(), "Source archive must contain regular files and directories")
        require(
            not Path(member.name).is_absolute() and ".." not in Path(member.name).parts, "Invalid source archive path"
        )
    if hasattr(tarfile, "data_filter"):
        archive.extractall(destination, filter="data")
    else:
        # Compatibility with Python 3.9 versions predating extraction filters.
        archive.extractall(destination)


def inspect_pipeline(source: Path, commit: str, with_models: bool) -> dict[str, object]:
    research = json.loads((source / "artifacts/research_manifest.json").read_text())
    result = json.loads((source / research["reduced_e2e_manifest"]).read_text())
    holdout = json.loads((source / result["holdout_manifest"]).read_text())
    development = csv_rows(source / result["development_fixture"])
    registry = [json.loads(line) for line in (source / result["experiment_registry"]).read_text().splitlines()]
    fills = csv_rows(source / result["stateful_fills"])
    positions = csv_rows(source / result["stateful_positions"])
    require(research["git_commit"] == commit, "Synthetic evidence identifies a different source revision")
    require(research["working_tree_dirty"] is None, "Exported-source provenance must identify missing Git metadata")
    require(
        research["final_holdout"]["status"] == "not_run", "Synthetic flow must not claim empirical final evaluation"
    )
    require(holdout["source_sha256"] == sha256(source / result["fixture"]), "Fixture holdout hash mismatch")
    require(result["holdout_rows_excluded"] > 0 and len(development) > 0, "Holdout filtering was not exercised")
    require(
        len(development) + result["holdout_rows_excluded"] == result["source_rows_before_holdout_filter"],
        "Development/holdout row accounting mismatch",
    )
    require(
        not set(holdout["holdout_values"]) & {row[holdout["split_column"]] for row in development},
        "Holdout rows entered model selection",
    )
    require(len(registry) == result["registered_candidate_count"] > 1, "Search family is incomplete")
    require(all(row["selection_metric"].startswith("validation_") for row in registry), "Test-driven model selection")
    selected = [row for row in registry if row["selected"]]
    require(len(selected) == 1, "Expected one validation-selected threshold procedure")
    require(
        selected[0]["validation_net_pnl"] == max(row["validation_net_pnl"] for row in registry),
        "Selected procedure does not maximize the declared validation objective",
    )
    require(bool(fills) and bool(positions), "Execution ledgers were not exercised")
    require(
        abs(float(positions[-1]["equity"]) - result["stateful_final_equity"]) < 1e-9,
        "Final equity differs from the execution ledger",
    )
    models = {name: value["status"] for name, value in result["sequence_smokes"].items()}
    if with_models:
        require(set(models) == {"sequence_tcn", "sequence_transformer"}, "Both sequence architectures are required")
        require(
            all(value == "pipeline_completed" for value in models.values()), f"Incomplete CPU sequence flow: {models}"
        )
    else:
        require(
            all(value == "skipped" for value in models.values()),
            "Minimal wheel environment gained optional dependencies",
        )
    return {
        "source_rows": result["source_rows_before_holdout_filter"],
        "development_rows": len(development),
        "holdout_rows_excluded": result["holdout_rows_excluded"],
        "registered_candidates": len(registry),
        "fill_rows": len(fills),
        "position_rows": len(positions),
        "sequence_models": models,
        "claim_scope": result["claim_scope"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--with-models", action="store_true", help="Install pinned CPU Torch/NumPy and require both neural flows"
    )
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or ROOT / "artifacts/public_e2e" / stamp).resolve()
    require(not output.exists(), "Evidence output already exists; choose a new directory")
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    (output / "packages").mkdir()
    evidence: dict[str, object] = {
        "status": "running",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "with_models": args.with_models,
        "claim_scope": "Software E2E using synthetic fixtures only; no empirical alpha claim.",
        "steps": [],
    }
    env = os.environ.copy()
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "LOB_FORGE_SOURCE_GIT_COMMIT",
        "LOB_FORGE_SOURCE_WORKING_TREE_DIRTY",
    ):
        env.pop(name, None)
    env.update({"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"})

    def run(
        name: str, command: list[str | Path], cwd: Path, *, reject: str = "", extra_env: dict[str, str] | None = None
    ) -> Path:
        log = output / "logs" / f"{len(evidence['steps']) + 1:02d}-{name}.log"
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [str(item) for item in command],
                cwd=cwd,
                env={**env, **(extra_env or {})},
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=600,
            )
        passed = (completed.returncode != 0 and reject in log.read_text()) if reject else completed.returncode == 0
        evidence["steps"].append(
            {
                "name": name,
                "passed": passed,
                "returncode": completed.returncode,
                "expected_rejection": reject or None,
                "log": log.relative_to(output).as_posix(),
            }
        )
        require(passed, f"Step failed: {name}; inspect {log}")
        print(f"passed: {name}", flush=True)
        return log

    try:
        with tempfile.TemporaryDirectory(prefix="microstructure-public-e2e-") as temporary:
            work = Path(temporary)
            source = work / "source"
            snapshot = output / "packages/source-snapshot.tar.gz"
            if (ROOT / ".git").exists():
                status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
                require(not status.strip(), "Commit the source changes before running the provenance-bound E2E")
                commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
                run(
                    "source-snapshot",
                    ["git", "archive", "--format=tar.gz", "--prefix=source/", f"--output={snapshot}", "HEAD"],
                    ROOT,
                )
            else:
                commit = (ROOT / ".source-git-commit").read_text().strip()
                require(
                    len(commit) in (40, 64) and all(c in "0123456789abcdef" for c in commit),
                    "Missing source archive commit",
                )
                with tarfile.open(snapshot, "w:gz") as archive:
                    for path in public_files(ROOT):
                        require(path.is_file() and not path.is_symlink(), "Source snapshot requires regular files")
                        archive.add(path, arcname="source/" + path.relative_to(ROOT).as_posix())
            evidence["git_commit"] = commit
            with tarfile.open(snapshot) as archive:
                inspect_archive(archive.getnames())
                extract_source(archive, work)
            require((source / ".source-git-commit").read_text().strip() == commit, "Exported commit marker mismatch")
            run(
                "repository-hygiene",
                [
                    sys.executable,
                    source / "scripts/check_repository.py",
                    "--project-root",
                    source,
                    "--output",
                    output / "repository-check.json",
                ],
                work,
            )
            builder = work / "builder"
            run("build-environment", [sys.executable, "-m", "venv", builder], work)
            build_python = builder / "bin/python"
            run(
                "pinned-build-tools",
                [build_python, "-m", "pip", "install", "-r", source / "requirements-build.txt"],
                work,
            )
            run(
                "build-distributions",
                [build_python, "-m", "build", "--no-isolation", "--outdir", output / "packages", source],
                work,
            )
            sdist = next((output / "packages").glob("microstructure_alpha_lab-*.tar.gz"))
            rebuilt_source = work / "from-sdist"
            rebuilt_source.mkdir()
            with tarfile.open(sdist) as archive:
                inspect_archive(archive.getnames())
                extract_source(archive, rebuilt_source)
            package_source = next(rebuilt_source.iterdir())
            run(
                "rebuild-from-sdist",
                [
                    build_python,
                    "-m",
                    "build",
                    "--wheel",
                    "--no-isolation",
                    "--outdir",
                    output / "packages/rebuilt",
                    package_source,
                ],
                work,
            )
            runtime = work / "runtime"
            run("runtime-environment", [sys.executable, "-m", "venv", runtime], work)
            python = runtime / "bin/python"
            wheel = next((output / "packages/rebuilt").glob("*.whl"))
            run("install-wheel", [python, "-m", "pip", "install", "--no-deps", "--no-index", wheel], work)
            run(
                "installed-package",
                [
                    python,
                    "-c",
                    "from pathlib import Path; import sys, lob_forge; "
                    "assert Path(sys.prefix) in Path(lob_forge.__file__).parents; "
                    "from importlib.metadata import metadata; m = metadata('microstructure-alpha-lab'); "
                    "assert not [v for v in m.get_all('Requires-Dist', []) if 'extra ==' not in v]; "
                    "assert 'MIT' in m['License']; print(lob_forge.__version__)",
                ],
                work,
            )
            for executable in ("microstructure-alpha-lab", "lob-forge"):
                run(f"cli-{executable}", [runtime / "bin" / executable, "--help"], work)
            if args.with_models:
                run("cpu-numpy", [python, "-m", "pip", "install", "numpy==2.0.2"], work)
                torch_command = [python, "-m", "pip", "install", "torch==2.8.0"]
                if platform.system() == "Linux":
                    torch_command.extend(["--index-url", "https://download.pytorch.org/whl/cpu"])
                run("cpu-torch", torch_command, work)
            versions = run("runtime-versions", [python, "-m", "pip", "list", "--format=json"], work)
            evidence["runtime_packages"] = json.loads(versions.read_text())
            run("synthetic-pipeline", [python, source / "scripts/run_reduced_e2e.py"], source)
            require((source / "artifacts/research_manifest.json").is_file(), "Pipeline evidence is absent")
            run(
                "pipeline-verifier",
                [python, source / "scripts/verify_reduced_e2e_artifacts.py", "--project-root", source],
                source,
            )
            evidence["pipeline"] = inspect_pipeline(source, commit, args.with_models)
            cli = runtime / "bin/microstructure-alpha-lab"
            fixture = source / "examples/fixtures/feature_fixture.csv"
            manifest = source / "artifacts/reduced_e2e/holdout_manifest.json"
            run("reject-missing-holdout", [cli, "baseline", fixture], source, reject="--holdout-manifest")
            run(
                "reject-test-selection",
                [cli, "baseline", fixture, "--holdout-manifest", manifest, "--sort-by", "test_net_pnl"],
                source,
                reject="invalid choice",
            )
            stale = json.loads(manifest.read_text())
            stale["source_sha256"] = "0" * 64
            stale_manifest = source / "artifacts/stale-holdout.json"
            stale_manifest.write_text(json.dumps(stale))
            run(
                "reject-stale-holdout",
                [cli, "baseline", fixture, "--holdout-manifest", stale_manifest],
                source,
                reject="holdout manifest verification failed",
            )
            require(shutil.which("c++") is not None, "A C++17 compiler is required for replay parity")
            binary = work / "l2_replay"
            run("build-cpp", ["bash", source / "scripts/build_cpp_l2_replay.sh", binary], source)
            replay_fixture = source / "examples/fixtures/l2_replay_fixture.csv"
            cpp = run("cpp-replay", [binary, replay_fixture], work)
            reference = run("python-replay", [python, "-c", textwrap.dedent(REPLAY_REFERENCE), replay_fixture], work)
            require(csv_rows(cpp) == csv_rows(reference), "C++ replay differs from the installed Python package")
            evidence["cpp_python_parity"] = {"matched": True, "update_rows": len(csv_rows(cpp))}
            # Probe the original packaging failures using disposable source-only files.
            (source / "docs/public_e2e_note 2.md").write_text("local backup; must not be packaged\n")
            cache = source / "src/lob_forge/__pycache__"
            cache.mkdir(exist_ok=True)
            (cache / "public_e2e.pyc").write_bytes(b"local cache")
            package_env = {"DIST_DIR": str(output / "packages"), "PACKAGE_NAME": "source-handoff.zip"}
            run("source-handoff", ["bash", source / "scripts/package_cloud_handoff.sh"], source, extra_env=package_env)
            with zipfile.ZipFile(output / "packages/source-handoff.zip") as archive:
                require(archive.testzip() is None, "Source handoff ZIP is corrupt")
                inspect_archive(archive.namelist())
                require(
                    archive.read("microstructure-alpha-lab/.source-git-commit").decode().strip() == commit,
                    "Source handoff identifies a different revision",
                )
            run(
                "reject-package-overwrite",
                ["bash", source / "scripts/package_cloud_handoff.sh"],
                source,
                reject="Package already exists",
                extra_env=package_env,
            )
            shutil.copytree(source / "artifacts/reduced_e2e", output / "pipeline/reduced_e2e")
            shutil.copy2(source / "artifacts/research_manifest.json", output / "pipeline/research_manifest.json")
            evidence["status"] = "passed"
    except (
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        StopIteration,
        ValueError,
        KeyError,
        tarfile.TarError,
        zipfile.BadZipFile,
    ) as exc:
        evidence["status"] = "failed"
        evidence["error"] = str(exc)
        print(str(exc), file=sys.stderr)
    finally:
        evidence["artifact_sha256"] = {
            path.relative_to(output).as_posix(): sha256(path) for path in sorted(output.rglob("*")) if path.is_file()
        }
        (output / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        print(f"evidence={output / 'evidence.json'}", flush=True)
    return 0 if evidence["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
