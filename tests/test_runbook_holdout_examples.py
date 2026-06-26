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


def _shell_command_blocks(text: str) -> list[tuple[int, str]]:
    lines = text.splitlines()
    blocks: list[tuple[int, str]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue
        start = index + 1
        current = [line]
        if stripped.endswith("=("):
            index += 1
            while index < len(lines):
                current.append(lines[index])
                if lines[index].strip() == ")":
                    index += 1
                    break
                index += 1
        else:
            index += 1
            while current[-1].rstrip().endswith("\\") and index < len(lines):
                current.append(lines[index])
                index += 1
        blocks.append((start, "\n".join(current)))
    return blocks


def _gated_commands_in(block: str) -> list[str]:
    return [command for command in GATED_COMMANDS if f"lob_forge.cli {command}" in block]


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
            gated = _gated_commands_in(block)
            if gated and "--holdout-manifest" not in block:
                failures.append(f"{path.relative_to(repo)} block {index}: {', '.join(sorted(gated))}")

    assert failures == []


def test_shell_runbook_gated_cli_commands_include_holdout_manifest() -> None:
    repo = Path(__file__).resolve().parents[1]
    shell_paths = sorted((repo / "scripts").glob("*.sh"))
    failures: list[str] = []
    for path in shell_paths:
        for line_number, block in _shell_command_blocks(path.read_text()):
            if "-m lob_forge.cli" not in block:
                continue
            gated = _gated_commands_in(block)
            if gated and "--holdout-manifest" not in block:
                failures.append(f"{path.relative_to(repo)}:{line_number}: {', '.join(sorted(gated))}")

    assert failures == []
