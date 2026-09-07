"""Render a standalone figure from completed, source-backed forecast evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def render(summary_path, diagnostics_path, gate_path, output):
    results = json.loads(summary_path.read_text())
    diagnostics = json.loads(diagnostics_path.read_text())
    gate = json.loads(gate_path.read_text())
    if hashlib.sha256(summary_path.read_bytes()).hexdigest() != diagnostics["confirmation_summary_sha256"]:
        raise ValueError("Diagnostics and confirmation summary must refer to the same result")
    if gate["confirmation_summary_sha256"] != diagnostics["confirmation_summary_sha256"]:
        raise ValueError("The research gate must refer to the same confirmation result")
    procedures = {name: results["procedures"][name] for name in ["original_blend", "pooled_balanced_tree_blend"]}
    if set(procedures) != {"original_blend", "pooled_balanced_tree_blend"} or any(
        p["distinct_dates"] != 20 or p["asset_date_assessments"] != 40 for p in procedures.values()
    ):
        raise ValueError("A complete paired twenty-date confirmation is required")
    keys = ["original_reference", "matched_data_control", "original_blend", "pooled_balanced_tree_blend"]
    names = ["Noon reference", "Full-day control", "Neural + tree", "Neural + shared tree"]
    metrics = {
        "original_reference": procedures["original_blend"]["mean_metrics"]["original_reference"],
        "matched_data_control": procedures["original_blend"]["mean_metrics"]["matched_data_control"],
        **{key: value["mean_metrics"]["candidate"] for key, value in procedures.items()},
    }
    for key in keys:
        np.testing.assert_allclose(
            metrics[key]["log_loss"], diagnostics["mean_metrics"][key]["three_class_log_loss_with_shared_smoothing"], atol=1e-8,
        )
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(14, 10), facecolor="white")
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1.1, 1.25], hspace=0.65, wspace=0.4)
    colors = ["#355cac", "#008269"]
    title = "Substantial-gain gate passed" if gate["substantial_predictive_gain_confirmed"] else "Substantial-gain gate not met"
    fig.suptitle(f"Independent forecast confirmation · {title}", x=0.06, y=0.975, ha="left", fontsize=18, weight="bold")
    fig.text(0.06, 0.936, "BTCUSDT + ETHUSDT  |  24 May–12 June 2023  |  20 paired date clusters  |  282,800 assessment decisions", color="#555555")

    ax = fig.add_subplot(grid[0, 0])
    intervals = [gate["procedures"][name]["interval"] for name in procedures]
    for index, (interval, color) in enumerate(zip(intervals, colors)):
        lo, mu, hi = [100 * interval[k] for k in ["lower", "mean", "upper"]]
        ax.errorbar(mu, index, xerr=[[mu - lo], [hi - mu]], fmt="o", color=color, capsize=5, markersize=7, linewidth=2)
        ax.text(hi + 0.2, index, f"{mu:+.2f} pp", va="center", color=color, weight="bold")
    ax.set_yticks([0, 1], names[2:])
    ax.set_ylim(-0.7, 1.7)
    ax.invert_yaxis()
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.axvline(5, color="#777777", linestyle="--", linewidth=1, label="Registered +5 pp target")
    ax.set_xlim(min(-1, 100 * min(i["lower"] for i in intervals) - 1), max(6.5, 100 * max(i["upper"] for i in intervals) + 2))
    ax.set_xlabel("Balanced-accuracy gain over noon reference (pp)")
    ax.set_title("Mean gain and intervals for the research gate", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = fig.add_subplot(grid[0, 1])
    ax.axis("off")
    cells = [[names[i], f"{metrics[k]['balanced_accuracy']:.2%}", f"{metrics[k]['natural_accuracy']:.2%}", f"{metrics[k]['log_loss']:.4f}"] for i, k in enumerate(keys)]
    table = ax.table(cellText=cells, colLabels=["Procedure", "Balanced", "Ordinary", "Log loss ↓"], colWidths=[0.44, 0.19, 0.19, 0.18], loc="center", cellLoc="right")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#e2e4e8")
        if row == 0:
            cell.set_facecolor("#edf1f7")
            cell.set_text_props(weight="bold")
        if column == 0:
            cell.set_text_props(ha="left")
    ax.set_title("Equal-weight means across assets and dates", loc="left", weight="bold")

    ax = fig.add_subplot(grid[1, :])
    for name, procedure, color in zip(names[2:], procedures.values(), colors):
        rows = procedure["paired_date_balanced_accuracy_deltas"]
        dates = np.array([row["date"] for row in rows], dtype="datetime64[D]")
        ax.plot(dates, [100 * row["original_reference"] for row in rows], marker="o", markersize=4, linewidth=1.6, color=color, label=name)
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.grid(axis="y", alpha=0.2)
    ax.set_ylabel("Gain over reference (pp)")
    ax.set_title("Every registered assessment date, keeping both assets paired", loc="left", weight="bold")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(frameon=False, loc="upper left", ncol=2)

    ax = fig.add_subplot(grid[2, 0])
    movement = [diagnostics["mean_metrics"][k]["movement_binary_log_loss"] for k in keys]
    direction = [diagnostics["mean_metrics"][k]["direction_log_loss_contribution_per_decision"] for k in keys]
    ax.bar(np.arange(4), movement, color="#a7b3c7", label="Movement")
    ax.bar(np.arange(4), direction, bottom=movement, color="#355cac", label="Direction contribution")
    ax.set_xticks(np.arange(4), ["Noon\nreference", "Full-day\ncontrol", "Neural +\ntree", "Neural +\nshared tree"])
    ax.set_ylabel("Log loss per decision · lower is better")
    ax.set_title("Where probability quality comes from", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_ylim(0, 1.2 * max(np.array(movement) + direction))
    ax.grid(axis="y", alpha=0.15)

    ax = fig.add_subplot(grid[2, 1])
    for field, marker, color, label in [
        ("movement_roc_auc", "o", "#355cac", "Movement ranking"),
        ("conditional_direction_roc_auc", "D", "#008269", "Direction ranking, on realized moves"),
    ]:
        values = [diagnostics["mean_metrics"][k][field] for k in keys]
        ax.scatter(values, np.arange(4), marker=marker, color=color, s=40, label=label)
    ax.set_yticks(np.arange(4), names)
    ax.set_ylim(4, -0.7)
    ax.axvline(0.5, color="#888888", linestyle=":", linewidth=1)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("ROC AUC · higher is better")
    ax.set_title("Movement and direction are different tasks", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.09)
    fig.text(0.06, 0.028, "Intervals: 20,000 paired date resamples; 98.75% marginal coverage for two procedures in the first research round.\nExact threshold labels. Historical prediction evidence; direction conditioning is a diagnostic, not an available trade-selection rule.", fontsize=8, color="#555555")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180, facecolor="white")
    svg = output.with_suffix(".svg")
    fig.savefig(svg, facecolor="white")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    identity = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [summary_path, diagnostics_path, gate_path, Path(__file__)]}
    output.with_suffix(".inputs.json").write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    print(output.with_suffix(".png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "results/boundary_pooled_tree_confirmation_20260907/summary.json")
    parser.add_argument("--diagnostics", type=Path, default=ROOT / "docs/research/boundary_confirmation_diagnostics_20260907.json")
    parser.add_argument("--gate", type=Path, default=ROOT / "docs/research/boundary_research_gate_result_20260907.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/research/boundary_confirmation_20260907")
    args = parser.parse_args()
    render(args.summary, args.diagnostics, args.gate, args.output)
