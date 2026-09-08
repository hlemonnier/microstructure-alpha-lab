"""Bounded synthetic CPU throughput check; no market observations or labels."""

from __future__ import annotations

import argparse
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from lob_forge.boundary_embedding import ARCHITECTURES, build_embedding_network
from lob_forge.boundary_pooled import build_member_network
from run_boundary_confirmation import write_json


def benchmark():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    rows, records = 1024, []
    for features in (220, 399):
        for name, (embedded, members) in ARCHITECTURES.items():
            torch.manual_seed(20260907)
            if embedded:
                network = build_embedding_network([32] * features, dimensions=4, members=members)
                values = (torch.randint(32, (rows, features)), torch.rand(rows, features),
                          (torch.arange(rows) % 2 * 2 - 1).float())
            else:
                network = build_member_network(features + 1, members=members)
                values = (torch.randn(rows, features + 1),)
            target = (torch.arange(rows) % 3)[:, None].expand(-1, members).reshape(-1)
            optimizer = torch.optim.AdamW(network.parameters(), lr=.001, weight_decay=.01)
            elapsed = []
            for step in range(6):
                start = time.monotonic()
                logits = network(*values)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 3), target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
                optimizer.step()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite synthetic throughput step")
                if step:
                    elapsed.append(time.monotonic() - start)
            records.append({"features": features, "architecture": name, "batch_size": rows,
                "warmup_steps": 1, "timed_steps": 5, "step_seconds": elapsed,
                "mean_step_seconds": float(np.mean(elapsed)),
                "parameter_count": sum(p.numel() for p in network.parameters())})
    return {"evidence_status": "synthetic_operational_measurement_only", "market_data_read": False,
        "model_screen_fits": 0, "threads": 2, "python": platform.python_version(),
        "torch": version("torch"), "records": records,
        "interpretation": "Single CPU run of forward/backward/optimizer steps; excludes preprocessing and validation. No predictive inference follows."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark()
    write_json(args.output, result)
    for record in result["records"]:
        print(record["features"], record["architecture"], f"{record['mean_step_seconds']:.4f}s/step")
