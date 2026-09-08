"""Date-excluded, class-balanced historical retrieval for three-class forecasts."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from lob_forge.boundary_forecasts import classification_metrics

SEMANTICS = "date_excluded_weighted_soft_neighbor_natural_posterior_v1"
ARCHITECTURES = ("fixed_metric", "linear_nca", "nonlinear_nca", "parametric")
CONFIG = {"seed": 20260908, "epochs": 6, "batch_size": 1024, "width": 128,
          "keys_per_cell": 128, "learning_rate": .001, "weight_decay": .01, "gradient_norm_limit": 5.}


def build_retrieval_network(dimensions, architecture, *, width=128):
    import torch
    from torch import nn

    if dimensions < 1 or width < 1 or architecture not in ARCHITECTURES:
        raise ValueError("Positive dimensions and a registered retrieval architecture required")

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.dimensions, self.architecture, self.width = dimensions, architecture, width
            self.register_buffer("location", torch.empty(0))
            if architecture == "fixed_metric":
                self.encoder = nn.Identity()
            elif architecture == "linear_nca":
                self.encoder = nn.Linear(dimensions, width, bias=False)
            else:
                self.encoder = nn.Sequential(nn.Linear(dimensions, width), nn.BatchNorm1d(width),
                    nn.Linear(width, 2 * width), nn.ReLU(), nn.Dropout(.1),
                    nn.Linear(2 * width, width), nn.BatchNorm1d(width))
            self.head = nn.Linear(width, 3) if architecture == "parametric" else None

        def forward(self, matrix):
            return self.encoder(matrix)

    torch.manual_seed(CONFIG["seed"])
    return Network()


class HistoricalBankSampler:
    """Stratified sampling and exact empirical class-mixture importance weights."""

    def __init__(self, labels, assets, dates, *, keys_per_cell=128):
        self.labels, self.assets, self.dates = (np.asarray(v) for v in (labels, assets, dates))
        if (self.labels.ndim != 1 or not len(self.labels) or any(v.shape != self.labels.shape for v in (self.assets, self.dates))
            or not all(np.issubdtype(v.dtype, np.integer) for v in (self.labels, self.assets, self.dates))
            or not np.isin(self.labels, [-1, 0, 1]).all() or not np.isin(self.assets, [0, 1]).all()
            or not np.array_equal(np.unique(self.dates), np.arange(4)) or keys_per_cell < 1):
            raise ValueError("Aligned integer labels, two assets and four historical date IDs required")
        self.keys_per_cell = int(keys_per_cell)
        self.cells = {(d, a, c): np.flatnonzero((self.dates == d) & (self.assets == a) & (self.labels == c - 1))
            for d in range(4) for a in range(2) for c in range(3)}
        self.counts = np.array([len(self.cells[d, a, c]) for d in range(4) for a in range(2) for c in range(3)]).reshape(4, 2, 3)
        if (self.counts < self.keys_per_cell).any():
            raise ValueError("Insufficient historical support for the exact registered bank size")
        self.priors = {a: self.counts[:, a].sum(axis=0) / self.counts[:, a].sum() for a in range(2)}
        self.query_weights = (len(self.labels) / (6 * self.counts.sum(axis=0)[self.assets, self.labels + 1])).astype(np.float32)

    def sample(self, rng):
        return np.concatenate([rng.choice(rows, self.keys_per_cell, replace=False) for rows in self.cells.values()])

    def log_weights(self, query_assets, query_dates, keys):
        qa, qd, keys = (np.asarray(v) for v in (query_assets, query_dates, keys))
        if (qa.ndim != 1 or not len(qa) or qd.shape != qa.shape or keys.ndim != 1
            or not np.isin(qa, [0, 1]).all() or not np.isin(qd, range(4)).all()
            or len(keys) != 24 * self.keys_per_cell or len(np.unique(keys)) != len(keys)
            or not np.issubdtype(keys.dtype, np.integer) or (keys < 0).any() or (keys >= len(self.labels)).any()):
            raise ValueError("Valid queries and the unique complete stratified bank required")
        kd, ka, kc = self.dates[keys], self.assets[keys], self.labels[keys] + 1
        realized = np.bincount(kd * 6 + ka * 3 + kc, minlength=24)
        if not np.all(realized == self.keys_per_cell):
            raise ValueError("Every sampled date/asset/class cell must match its registered count")
        result = np.full((len(qa), len(keys)), -np.inf, dtype=np.float32)
        totals = self.counts.sum(axis=0)
        for day in range(4):
            for asset in range(2):
                query_rows = np.flatnonzero((qa == asset) & (qd == day))
                eligible = np.flatnonzero((ka == asset) & (kd != day))
                numerator = self.counts[kd[eligible], asset, kc[eligible]]
                denominator = self.keys_per_cell * (totals[asset, kc[eligible]] - self.counts[day, asset, kc[eligible]])
                result[np.ix_(query_rows, eligible)] = np.log(numerator / denominator)
        return result


def kernel_class_log_sums(query, keys, classes, log_weights):
    """Unnormalized class scores; all masking occurs before normalization."""
    import torch

    squared = (query.square().sum(-1, keepdim=True) + keys.square().sum(-1)[None] - 2 * query @ keys.T).clamp_min(0)
    scores = -(squared + 1e-12).sqrt() + log_weights
    return torch.stack([torch.logsumexp(scores.masked_fill(classes[None] != c, -torch.inf), dim=1) for c in range(3)], dim=1)


def kernel_log_prob(query, keys, classes, log_weights):
    import torch

    scores = kernel_class_log_sums(query, keys, classes, log_weights)
    return scores - torch.logsumexp(scores, dim=1, keepdim=True)


def retrieval_training_loss(network, matrix, labels, sampler, query_rows, key_rows):
    import torch

    if network.architecture == "fixed_metric":
        raise ValueError("The fixed metric has no optimization objective")
    device = network.location.device
    network.train()
    joined = np.concatenate([matrix[query_rows], matrix[key_rows]])
    encoded = network(torch.from_numpy(joined).to(device))
    query, keys = encoded[:len(query_rows)], encoded[len(query_rows):]
    target = torch.as_tensor(labels[query_rows] + 1, dtype=torch.long, device=device)
    if network.head is not None:
        losses = torch.nn.functional.cross_entropy(network.head(query), target, reduction="none")
    else:
        weights = sampler.log_weights(sampler.assets[query_rows], sampler.dates[query_rows], key_rows)
        log_q = kernel_log_prob(query, keys, torch.as_tensor(labels[key_rows] + 1, device=device), torch.from_numpy(weights).to(device))
        losses = -log_q.gather(1, target[:, None]).squeeze(1)
    loss = (losses * torch.from_numpy(sampler.query_weights[query_rows]).to(device)).mean()
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite date-excluded retrieval training objective")
    return loss


def encode_rows(network, matrix, *, chunk_rows=8192):
    import torch

    x = np.asarray(matrix)
    if x.ndim != 2 or not len(x) or x.shape[1] != network.dimensions or not np.isfinite(x).all() or chunk_rows < 1:
        raise ValueError("Nonempty finite input matrix with the frozen feature dimension required")
    network.eval()
    with torch.no_grad():
        return torch.cat([network(torch.as_tensor(x[start:start + chunk_rows], dtype=torch.float32, device=network.location.device))
            for start in range(0, len(x), chunk_rows)])


def streamed_log_prob(query, keys, classes, *, key_chunk_rows=8192):
    import torch

    if (query.ndim != 2 or keys.ndim != 2 or query.shape[1] != keys.shape[1] or not len(query)
        or not len(keys) or classes.shape != (len(keys),) or key_chunk_rows < 1):
        raise ValueError("Aligned nonempty query/key embeddings and class labels required")
    counts = torch.bincount(classes, minlength=3)
    if len(counts) != 3 or (counts <= 0).any():
        raise ValueError("Every historical retrieval class requires positive support")
    weights = -counts.to(query.dtype).log()[classes]
    scores = query.new_full((len(query), 3), -torch.inf)
    for start in range(0, len(keys), key_chunk_rows):
        end = start + key_chunk_rows
        part = kernel_class_log_sums(query, keys[start:end], classes[start:end], weights[start:end][None])
        scores = torch.logaddexp(scores, part)
    return scores - torch.logsumexp(scores, dim=1, keepdim=True)


@dataclass
class RetrievalForecaster:
    network: Any
    normalizer: Any
    active: np.ndarray
    columns: list[str]
    priors: dict[int, np.ndarray]
    banks: dict[int, dict[str, np.ndarray]]

    def matrix(self, frame, asset):
        if list(frame.columns) != self.columns or asset not in self.priors:
            raise ValueError("Exact historical feature schema and asset required")
        raw = frame.to_numpy(dtype=float)
        if not np.isfinite(raw).all():
            raise ValueError("Finite prediction features required")
        return np.column_stack([self.normalizer.transform(raw[:, self.active]), np.full(len(raw), 2 * asset - 1)]).astype(np.float32)

    def refresh_bank(self, matrix, labels, assets):
        if self.network.head is not None:
            self.banks = {}
            return
        encoded = encode_rows(self.network, matrix).cpu().numpy()
        self.banks = {a: {"embeddings": encoded[assets == a].copy(), "classes": (labels[assets == a] + 1).astype(np.int64)} for a in (0, 1)}

    def predict_matrix(self, matrix, asset, *, batch_size=256, key_chunk_rows=8192):
        import torch

        if asset not in self.priors or batch_size < 1 or key_chunk_rows < 1 or not len(matrix):
            raise ValueError("Supported asset, nonempty queries and positive chunk sizes required")
        device = self.network.location.device
        self.network.eval()
        output = []
        if self.network.head is None:
            bank = self.banks[asset]
            keys = torch.from_numpy(bank["embeddings"]).to(device)
            classes = torch.from_numpy(bank["classes"]).to(device)
        with torch.no_grad():
            for start in range(0, len(matrix), batch_size):
                encoded = encode_rows(self.network, matrix[start:start + batch_size])
                log_q = (torch.log_softmax(self.network.head(encoded), dim=-1) if self.network.head is not None
                    else streamed_log_prob(encoded, keys, classes, key_chunk_rows=key_chunk_rows))
                scores = log_q.cpu().numpy().astype(float) + np.log(self.priors[asset])
                scores -= scores.max(axis=1, keepdims=True)
                p = np.exp(scores)
                p /= p.sum(axis=1, keepdims=True)
                output.append(p)
        result = np.concatenate(output)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite historical retrieval forecast")
        return result

    def predict_proba(self, frame, asset, **options):
        return self.predict_matrix(self.matrix(frame, asset), asset, **options)

    def save(self, directory):
        import joblib
        import torch

        directory.mkdir(parents=True)
        joblib.dump(self.normalizer, directory / "normalizer.joblib")
        torch.save({"semantics": SEMANTICS, "architecture": self.network.architecture, "dimensions": self.network.dimensions,
            "width": self.network.width, "state_dict": {k: v.detach().cpu() for k, v in self.network.state_dict().items()},
            "active": self.active.tolist(), "columns": self.columns, "priors": {a: p.tolist() for a, p in self.priors.items()}}, directory / "model.pt")
        np.savez_compressed(directory / "banks.npz", **{f"{a}_{name}": value for a, bank in self.banks.items() for name, value in bank.items()})

    @classmethod
    def load(cls, directory, *, device="cpu"):
        import joblib
        import torch

        state = torch.load(directory / "model.pt", weights_only=True, map_location="cpu")
        if state["semantics"] != SEMANTICS:
            raise ValueError("Incompatible historical retrieval posterior semantics")
        network = build_retrieval_network(state["dimensions"], state["architecture"], width=state["width"]).to(device)
        network.load_state_dict(state["state_dict"])
        network.eval()
        with np.load(directory / "banks.npz", allow_pickle=False) as saved:
            banks = {a: {name: saved[f"{a}_{name}"].copy() for name in ("embeddings", "classes")} for a in (0, 1)} if network.head is None else {}
        return cls(network, joblib.load(directory / "normalizer.joblib"), np.array(state["active"], dtype=bool), state["columns"],
            {int(a): np.array(p) for a, p in state["priors"].items()}, banks)


def fit_retrieval(original, matrix, labels, assets, dates, validation, validation_labels, *, architecture, device="cpu",
                  epochs=6, batch_size=1024, keys_per_cell=128):
    import torch

    sampler = HistoricalBankSampler(labels, assets, dates, keys_per_cell=keys_per_cell)
    labels, assets, dates = sampler.labels, sampler.assets, sampler.dates
    x = np.asarray(matrix)
    if x.dtype != np.float32 or x.ndim != 2 or len(x) != len(labels) or not np.isfinite(x).all():
        raise ValueError("Aligned finite float32 historical matrix required")
    for asset in (0, 1):
        np.testing.assert_array_equal(sampler.priors[asset], original.priors[asset])
    if epochs < 1 or batch_size < 2:
        raise ValueError("Positive epochs and minibatches compatible with batch normalization required")
    if set(validation) != {0, 1} or set(validation_labels) != {0, 1} or any(
        len(validation[a]) != len(validation_labels[a]) or not len(validation[a]) for a in (0, 1)
    ):
        raise ValueError("Both aligned, nonempty historical validation assets required")
    network = build_retrieval_network(x.shape[1], architecture).to(device)
    learner = RetrievalForecaster(network, original.normalizer, original.active.copy(), list(original.columns),
        {a: p.copy() for a, p in original.priors.items()}, {})
    if architecture == "fixed_metric":
        learner.refresh_bank(x, labels, assets)
        return learner, {"history": [], "best_epoch": None, "optimizer_steps": 0, "config": dict(CONFIG)}
    optimizer = torch.optim.AdamW(network.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    order_rng, bank_rng = (np.random.default_rng(CONFIG["seed"]) for _ in range(2))
    history, best_key, best_state, best_epoch, steps = [], (-np.inf, -np.inf), None, None, 0
    for epoch in range(1, epochs + 1):
        order, losses = order_rng.permutation(len(x)), []
        for start in range(0, len(order), batch_size):
            query_rows, key_rows = order[start:start + batch_size], sampler.sample(bank_rng)
            optimizer.zero_grad()
            loss = retrieval_training_loss(network, x, labels, sampler, query_rows, key_rows)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), CONFIG["gradient_norm_limit"], error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach().cpu()))
        learner.refresh_bank(x, labels, assets)
        metrics = {a: classification_metrics(SimpleNamespace(priors=learner.priors[a]), learner.predict_matrix(validation[a], a), validation_labels[a]) for a in (0, 1)}
        key = (float(np.mean([m["balanced_accuracy"] for m in metrics.values()])), -float(np.mean([m["log_loss"] for m in metrics.values()])))
        if key > best_key:
            best_key, best_state, best_epoch = key, copy.deepcopy(network.state_dict()), epoch
        history.append({"epoch": epoch, "mean_training_minibatch_loss": float(np.mean(losses)), "validation": metrics})
    network.load_state_dict(best_state)
    learner.refresh_bank(x, labels, assets)
    network.eval()
    return learner, {"history": history, "best_epoch": best_epoch, "optimizer_steps": steps,
        "config": {**CONFIG, "epochs": epochs, "batch_size": batch_size, "keys_per_cell": keys_per_cell},
        "historical_cell_counts": sampler.counts.tolist(), "trainable_parameters": sum(p.numel() for p in network.parameters())}
