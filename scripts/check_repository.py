"""Inspect distributable files without printing potential secret values."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "LICENSE",
    "CONTRIBUTING.md",
    "CITATION.cff",
    "pyproject.toml",
    "docs/README.md",
    "docs/research/README.md",
    "cpp/l2_replay.cpp",
    "examples/fixtures/feature_fixture.csv",
    "requirements-build.txt",
)
LOCAL_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "build",
    "dist",
    "artifacts",
    "results",
    "node_modules",
}
TOKEN_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[0-9A-Z]{16})\b"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{40,}\b"),
)
LINK = re.compile(r'!?\[[^\]]*\]\((<[^>]+>|[^\s)]+)(?:\s+"[^"]*")?\)')
WORKSTATION_PATH = re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/")


def public_files(root: Path) -> list[Path]:
    if (root / ".git").exists():
        output = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
        ).decode("utf-8")
        return sorted({root / name for name in output.split("\0") if name})
    files = []
    for directory, dirs, names in os.walk(root):
        dirs[:] = [name for name in dirs if name not in LOCAL_DIRS and not name.endswith(".egg-info")]
        files.extend(Path(directory) / name for name in names if name != "WORKLOG.md")
    return sorted(files)


def inspect_repository(root: Path) -> dict[str, object]:
    errors = []
    hashes = {}
    links_checked = 0
    for required in REQUIRED:
        if not (root / required).is_file():
            errors.append({"path": required, "reason": "required public file is missing"})
    for path in public_files(root):
        relative = path.relative_to(root)
        name = relative.as_posix()
        if path.is_symlink() or not path.is_file():
            errors.append({"path": name, "reason": "public file must be a regular file"})
            continue
        forbidden = (
            any(part in LOCAL_DIRS or part.endswith(".egg-info") for part in relative.parts)
            or (relative.parts[0] == "data" and name != "data/README.md")
            or path.name in {"WORKLOG.md", ".DS_Store", ".env", "credentials.json"}
            or (path.name.startswith(".env.") and path.name != ".env.example")
            or path.suffix in {".pyc", ".pyo", ".pt", ".pkl", ".zip", ".bak", ".tmp", ".swp"}
            or re.search(r" [0-9]+\.[^.]+$", path.name) is not None
        )
        if forbidden:
            errors.append({"path": name, "reason": "local/generated/private file in public inventory"})
        raw = path.read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(content.splitlines(), 1):
            if any(pattern.search(line) for pattern in TOKEN_PATTERNS):
                errors.append({"path": name, "line": number, "reason": "credential pattern (value withheld)"})
        if path.suffix != ".md":
            continue
        if WORKSTATION_PATH.search(content):
            errors.append({"path": name, "reason": "workstation path in public documentation"})
        prose = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
        for match in LINK.finditer(prose):
            target = unquote(match.group(1).strip("<>").split("#")[0])
            if not target or re.match(r"[A-Za-z][A-Za-z0-9+.-]*:", target):
                continue
            links_checked += 1
            resolved = (path.parent / target).resolve()
            if not resolved.is_relative_to(root) or not resolved.exists():
                errors.append({"path": name, "reason": "invalid relative documentation target", "target": target})
    return {
        "status": "failed" if errors else "passed",
        "scope": "Public file inventory, relative Markdown targets and selected credential patterns; not a full security audit.",
        "files_checked": len(hashes),
        "relative_links_checked": links_checked,
        "errors": errors,
        "file_sha256": hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "repository_check.json")
    args = parser.parse_args()
    report = inspect_repository(args.project_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"repository_check={report['status']} files={report['files_checked']} links={report['relative_links_checked']}"
    )
    for error in report["errors"]:
        print(json.dumps(error, sort_keys=True))
    print(f"report={args.output}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
