import csv
import shutil
import subprocess
import time
from pathlib import Path

from lob_forge.data_sources import NormalizedL2Row
from lob_forge.l2_replay import replay_l2_rows


def test_cpp_l2_replay_matches_python_reference(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        return
    root = Path(__file__).resolve().parents[1]
    source = root / "cpp" / "l2_replay.cpp"
    fixture = root / "examples" / "fixtures" / "l2_replay_fixture.csv"
    binary = tmp_path / "l2_replay"

    subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True)

    started = time.perf_counter()
    output = subprocess.check_output([str(binary), str(fixture)], text=True)
    elapsed = time.perf_counter() - started
    cpp_rows = list(csv.DictReader(output.splitlines()))
    python_rows = _python_replay_rows(fixture)

    assert cpp_rows == python_rows
    assert elapsed >= 0.0


def _python_replay_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = [
            NormalizedL2Row(
                event_type=row["event_type"],
                exchange_timestamp=int(row["exchange_timestamp"]),
                local_timestamp=int(row["local_timestamp"]),
                side=row["side"],
                price=float(row["price"]),
                size=float(row["size"]),
                sequence=int(row["sequence"]) if row["sequence"] else None,
                update_id=int(row["update_id"]) if row["update_id"] else None,
                venue=row["venue"],
                symbol=row["symbol"],
            )
            for row in csv.DictReader(handle)
        ]
    _, updates, _ = replay_l2_rows(rows)
    formatted = []
    for update in updates:
        formatted.append(
            {
                "row_index": str(update.row_index),
                "sequence_gap": str(int(update.sequence_gap)),
                "reset": str(int(update.reset)),
                "crossed": str(int(update.crossed)),
                "best_bid": "" if update.best_bid is None else f"{update.best_bid:g}",
                "best_ask": "" if update.best_ask is None else f"{update.best_ask:g}",
                "needs_resnapshot": str(int(update.needs_resnapshot)),
            }
        )
    return formatted
