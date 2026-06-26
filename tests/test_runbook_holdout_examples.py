from pathlib import Path


GATED_COMMANDS = {
    "baseline",
    "calendar-walk-forward",
    "conditional-walk-forward",
    "edge-shadow-decisions",
    "edge-walk-forward",
    "eval-rule",
    "fee-sweep",
    "logistic-walk-forward",
    "regime",
    "walk-forward",
}


def _markdown_code_blocks(text: str) -> list[str]:
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in text.splitlines():
        if line.startswith("```"):
            if current is None:
                current = []
            else:
                blocks.append(current)
                current = None
            continue
        if current is not None:
            current.append(line)
    return ["\n".join(block) for block in blocks]


def test_runbook_gated_cli_examples_include_holdout_manifest() -> None:
    repo = Path(__file__).resolve().parents[1]
    markdown_paths = sorted([repo / "README.md", *repo.glob("docs/**/*.md")])
    failures: list[str] = []
    for path in markdown_paths:
        if not path.exists():
            continue
        for index, block in enumerate(_markdown_code_blocks(path.read_text()), start=1):
            if "-m lob_forge.cli" not in block:
                continue
            gated = [command for command in GATED_COMMANDS if f"lob_forge.cli {command}" in block]
            if gated and "--holdout-manifest" not in block:
                failures.append(f"{path.relative_to(repo)} block {index}: {', '.join(sorted(gated))}")

    assert failures == []
