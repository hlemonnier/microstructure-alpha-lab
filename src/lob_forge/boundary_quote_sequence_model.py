"""Pooled BBO sequence encoders with training-only preprocessing and selection."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from lob_forge.boundary_confirm_model import CONFIG, SYMBOLS
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_neural import natural_posteriors
from lob_forge.boundary_pooled import balanced_asset_weights
from lob_forge.boundary_quote_sequence import EVENT_STRIDES, SEQUENCE_CHANNELS, SEQUENCE_LENGTH

ENCODERS = ("pooled_events", "event_tcn", "event_mlp")
SEQUENCE_SHAPE = (len(EVENT_STRIDES), SEQUENCE_LENGTH, len(SEQUENCE_CHANNELS))


def checked_sequences(values):
    array = np.asarray(values)
    if array.ndim != 4 or array.shape[1:] != SEQUENCE_SHAPE or not len(array) or not np.isfinite(array).all():
        raise ValueError("Nonempty finite registered quote sequences required")
    mask = array[..., -1]
    if not np.isin(mask, [0, 1]).all() or (array[mask == 0] != 0).any() or (mask[:, :, -1] != 1).any():
        raise ValueError("Zero padding, binary availability, and a present latest block required")
    return array


def fit_sequence_scaler(sequences):
    values = checked_sequences(sequences)
    shape = (SEQUENCE_SHAPE[0], SEQUENCE_SHAPE[2])
    total, square, count = np.zeros(shape), np.zeros(shape), np.zeros((shape[0], 1))
    for start in range(0, len(values), 2048):
        part = values[start : start + 2048].astype(np.float64)
        mask = part[..., -1, None]
        total += np.sum(part, axis=(0, 2))
        square += np.sum(part * part, axis=(0, 2))
        count += np.sum(mask, axis=(0, 2))
    mean = total / count
    scale = np.sqrt(np.maximum(0, square / count - mean**2))
    scale = np.maximum(scale, 1e-4)
    mean[:, -1], scale[:, -1] = 0, 1
    return mean.astype(np.float32), scale.astype(np.float32)


def normalize_sequences(sequences, mean, scale):
    values = np.asarray(sequences, dtype=np.float32)
    if mean.shape != (SEQUENCE_SHAPE[0], SEQUENCE_SHAPE[2]) or scale.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Finite fitted sequence normalization required")
    mask = values[..., -1, None]
    return (np.clip((values - mean[None, :, None]) / scale[None, :, None], -8, 8) * mask).astype(np.float32)


def build_quote_sequence_network(feature_count, encoder):
    import torch
    from torch import nn

    if encoder not in ENCODERS or feature_count < 1:
        raise ValueError("Registered encoder and positive observed feature count required")

    class CausalBlock(nn.Module):
        def __init__(self, dilation):
            super().__init__()
            self.dilation = dilation
            self.convolution = nn.Conv1d(12, 12, 3, dilation=dilation)

        def forward(self, x):
            return torch.relu(x + self.convolution(torch.nn.functional.pad(x, (2 * self.dilation, 0))))

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.context = nn.Sequential(nn.Linear(feature_count, 64), nn.ReLU())
            if encoder == "event_mlp":
                self.events = nn.Sequential(nn.Flatten(), nn.Linear(int(np.prod(SEQUENCE_SHAPE)), 64), nn.ReLU())
                event_width = 64
            else:
                self.events = nn.Sequential(nn.Conv1d(SEQUENCE_SHAPE[2], 12, 1), nn.ReLU(),
                    *(CausalBlock(d) for d in (1, 4, 16))) if encoder == "event_tcn" else nn.Sequential(
                        nn.Conv1d(SEQUENCE_SHAPE[2], 12, 1), nn.ReLU(), nn.Conv1d(12, 12, 1), nn.ReLU())
                event_width = SEQUENCE_SHAPE[0] * 12 * 3
            self.head = nn.Sequential(nn.Linear(64 + event_width, 64), nn.ReLU(), nn.Linear(64, 3))

        def forward(self, context, sequence):
            if encoder == "event_mlp":
                events = self.events(sequence)
            else:
                n = len(sequence)
                stream = sequence.reshape(n * SEQUENCE_SHAPE[0], SEQUENCE_SHAPE[1], SEQUENCE_SHAPE[2])
                mask = stream[:, :, -1].unsqueeze(1)
                hidden = self.events(stream.transpose(1, 2))
                mean = (hidden * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1)
                maximum = hidden.masked_fill(mask == 0, -torch.inf).max(dim=2).values
                events = torch.cat([mean, maximum, hidden[:, :, -1]], dim=1).reshape(n, -1)
            return self.head(torch.cat([self.context(context), events], dim=1))

    return Network()


@dataclass
class QuoteSequenceForecaster:
    network: Any
    normalizer: Any
    active: np.ndarray
    columns: list[str]
    priors: dict[int, np.ndarray]
    sequence_mean: np.ndarray
    sequence_scale: np.ndarray
    encoder: str

    def predict_proba(self, features, sequences, asset_id, *, batch_size=1024):
        import torch

        if list(features.columns) != self.columns or asset_id not in self.priors or len(features) != len(sequences):
            raise ValueError("Prediction schema, sequence rows and asset must match training")
        sequence = checked_sequences(sequences)
        raw = features.to_numpy(dtype=float)
        if not np.isfinite(raw).all():
            raise ValueError("Finite observed tabular features required")
        context = np.column_stack([self.normalizer.transform(raw[:, self.active]), np.full(len(raw), 2 * asset_id - 1)]).astype(np.float32)
        result = []
        self.network.eval()
        with torch.no_grad():
            for start in range(0, len(raw), batch_size):
                stop = start + batch_size
                normalized = normalize_sequences(sequence[start:stop], self.sequence_mean, self.sequence_scale)
                logits = self.network(torch.from_numpy(context[start:stop]), torch.from_numpy(normalized))
                result.append(torch.softmax(logits, dim=-1).numpy())
        return natural_posteriors(np.concatenate(result), self.priors[asset_id])

    def save(self, directory):
        import joblib
        import torch

        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.normalizer, directory / "normalizer.joblib")
        torch.save({"state_dict": self.network.state_dict(), "active": self.active.tolist(), "columns": self.columns,
            "priors": {k: v.tolist() for k, v in self.priors.items()}, "encoder": self.encoder,
            "sequence_mean": self.sequence_mean.tolist(), "sequence_scale": self.sequence_scale.tolist()}, directory / "model.pt")

    @classmethod
    def load(cls, directory):
        import joblib
        import torch

        state = torch.load(directory / "model.pt", weights_only=True, map_location="cpu")
        active = np.array(state["active"], dtype=bool)
        network = build_quote_sequence_network(int(active.sum()) + 1, state["encoder"])
        network.load_state_dict(state["state_dict"])
        network.eval()
        return cls(network, joblib.load(directory / "normalizer.joblib"), active, state["columns"],
            {int(k): np.array(v) for k, v in state["priors"].items()}, np.array(state["sequence_mean"], dtype=np.float32),
            np.array(state["sequence_scale"], dtype=np.float32), state["encoder"])


def fit_quote_sequence_model(training_features, training_sequences, training_labels,
                             validation_features, validation_sequences, validation_labels, *, encoder, config=None):
    import torch
    from sklearn.preprocessing import QuantileTransformer
    from threadpoolctl import threadpool_limits

    cfg = dict(CONFIG) if config is None else {**CONFIG, **config}
    if encoder not in ENCODERS or any(set(m) != set(SYMBOLS) for m in (
        training_features, training_sequences, training_labels, validation_features, validation_sequences, validation_labels)):
        raise ValueError("Both registered assets and a registered sequence encoder required")
    columns = list(training_features[SYMBOLS[0]].columns)
    for features, sequences, labels in ((training_features, training_sequences, training_labels),
                                       (validation_features, validation_sequences, validation_labels)):
        for symbol in SYMBOLS:
            if list(features[symbol].columns) != columns or not len(labels[symbol]) or len(features[symbol]) != len(labels[symbol]) or len(sequences[symbol]) != len(labels[symbol]):
                raise ValueError("Aligned nonempty partitions and identical feature schemas required")
            checked_sequences(sequences[symbol])
            if not np.isin(labels[symbol], [-1, 0, 1]).all() or not np.isfinite(features[symbol].to_numpy(dtype=float)).all():
                raise ValueError("Finite observed features and three-class labels required")
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    raw = pd.concat([training_features[s] for s in SYMBOLS], ignore_index=True).to_numpy(dtype=float)
    y = np.concatenate([training_labels[s] for s in SYMBOLS])
    asset_ids = np.concatenate([np.full(len(training_labels[s]), i, dtype=int) for i, s in enumerate(SYMBOLS)])
    priors, weights = balanced_asset_weights(y, asset_ids)
    active = np.ptp(raw, axis=0) > 0
    if not active.any():
        raise ValueError("At least one varying observed feature required")
    normalizer = QuantileTransformer(n_quantiles=min(1024, len(raw)), output_distribution="normal",
        subsample=min(100000, len(raw)), random_state=cfg["seed"])
    normalizer.fit(raw[:, active])
    context = np.column_stack([normalizer.transform(raw[:, active]), 2 * asset_ids - 1]).astype(np.float32)
    sequence = np.concatenate([training_sequences[s] for s in SYMBOLS]).astype(np.float32, copy=False)
    sequence_mean, sequence_scale = fit_sequence_scaler(sequence)
    for start in range(0, len(sequence), 2048):
        sequence[start : start + 2048] = normalize_sequences(sequence[start : start + 2048], sequence_mean, sequence_scale)
    network = build_quote_sequence_network(context.shape[1], encoder)
    model = QuoteSequenceForecaster(network, normalizer, active, columns, priors, sequence_mean, sequence_scale, encoder)
    optimizer = torch.optim.AdamW(network.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    target, sample_weights = torch.tensor(y + 1, dtype=torch.long), torch.from_numpy(weights)
    best_key, best_state, best_epoch, history = (-np.inf, -np.inf), None, None, []
    with threadpool_limits(limits=cfg["threads"]):
        for epoch in range(1, cfg["epochs"] + 1):
            network.train()
            order, losses = rng.permutation(len(y)), []
            for start in range(0, len(order), cfg["batch_size"]):
                rows = order[start : start + cfg["batch_size"]]
                logits = network(torch.from_numpy(context[rows]), torch.from_numpy(sequence[rows]))
                loss = (torch.nn.functional.cross_entropy(logits, target[rows], reduction="none") * sample_weights[rows]).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite sequence training loss")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), cfg["gradient_norm_limit"])
                optimizer.step()
                losses.append(float(loss.detach()))
            metrics = {s: classification_metrics(SimpleNamespace(priors=priors[i]),
                model.predict_proba(validation_features[s], validation_sequences[s], i), validation_labels[s]) for i, s in enumerate(SYMBOLS)}
            key = (float(np.mean([m["balanced_accuracy"] for m in metrics.values()])), -float(np.mean([m["log_loss"] for m in metrics.values()])))
            if key > best_key:
                best_key, best_state, best_epoch = key, copy.deepcopy(network.state_dict()), epoch
            history.append({"epoch": epoch, "mean_training_loss": float(np.mean(losses)), "validation": metrics})
    network.load_state_dict(best_state)
    network.eval()
    return model, {"history": history, "best_epoch": best_epoch, "config": cfg,
        "encoder": encoder, "parameters": sum(p.numel() for p in network.parameters())}
