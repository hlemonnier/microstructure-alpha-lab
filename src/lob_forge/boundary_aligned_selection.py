"""Compare checkpoint selectors along one unchanged neural optimization path.

All selectors see only preceding-day validation. The original selector must
reproduce its frozen checkpoint exactly. No selection changes gradient updates.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd

from lob_forge.boundary_adaptive_priors import forecast_priors
from lob_forge.boundary_confirm_model import CONFIG, SYMBOLS
from lob_forge.boundary_forecasts import classification_metrics
from lob_forge.boundary_pooled import PooledForecaster, balanced_asset_weights, build_member_network

SELECTORS = ("registered", "forecast_3600", "natural_log_loss")


def fit_aligned_selection(training_features, training_labels, validation_features, validation_labels,
                          validation_times, validation_initial_priors):
    import torch
    from sklearn.preprocessing import QuantileTransformer
    from threadpoolctl import threadpool_limits

    if any(
        set(mapping) != set(SYMBOLS)
        for mapping in [training_features, training_labels, validation_features, validation_labels, validation_times, validation_initial_priors]
    ):
        raise ValueError("Both registered assets are required in training and validation")
    torch.set_num_threads(CONFIG["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(CONFIG["seed"])
    rng = np.random.default_rng(CONFIG["seed"])
    columns = list(training_features[SYMBOLS[0]].columns)
    if any(list(frame.columns) != columns for frame in [*training_features.values(), *validation_features.values()]):
        raise ValueError("Training and validation feature schemas must match")
    for features, labels in ((training_features, training_labels), (validation_features, validation_labels)):
        for symbol in SYMBOLS:
            if not len(labels[symbol]) or len(features[symbol]) != len(labels[symbol]) or not np.isin(labels[symbol], [-1, 0, 1]).all() or not np.isfinite(features[symbol].to_numpy(dtype=float)).all():
                raise ValueError("Aligned finite observations and nonempty three-class labels required")
    for symbol in SYMBOLS:
        forecast_priors(np.full((len(validation_labels[symbol]), 3), 1 / 3), validation_times[symbol], validation_initial_priors[symbol], half_life_seconds=3600)
    train_x = pd.concat([training_features[symbol] for symbol in SYMBOLS], ignore_index=True)
    train_y = np.concatenate([training_labels[symbol] for symbol in SYMBOLS])
    assets = np.concatenate(
        [np.full(len(training_labels[symbol]), SYMBOLS.index(symbol), dtype=int) for symbol in SYMBOLS]
    )
    if len(train_x) != len(train_y):
        raise ValueError("Aligned training features and labels required")
    priors, weights = balanced_asset_weights(train_y, assets)
    raw_matrix = train_x.to_numpy(dtype=float)
    active = np.ptp(raw_matrix, axis=0) > 0
    normalizer = QuantileTransformer(
        n_quantiles=min(1024, len(train_x)),
        output_distribution="normal",
        subsample=min(100000, len(train_x)),
        random_state=CONFIG["seed"],
    )
    normalizer.fit(raw_matrix[:, active])
    normalized = np.column_stack([normalizer.transform(raw_matrix[:, active]), 2 * assets - 1]).astype(np.float32)
    network = build_member_network(normalized.shape[1], members=1, hidden_size=CONFIG["hidden_size"])
    learner = PooledForecaster(network, normalizer, active, columns, priors, 1, CONFIG["hidden_size"])
    optimizer = torch.optim.AdamW(network.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    target, sample_weights = torch.tensor(train_y + 1, dtype=torch.long), torch.from_numpy(weights)
    best_key = {name: (-np.inf, -np.inf) for name in SELECTORS}
    best_state, best_epoch, history = {}, {}, []
    with threadpool_limits(limits=CONFIG["threads"]):
        for epoch in range(1, CONFIG["epochs"] + 1):
            network.train()
            order = rng.permutation(len(train_y))
            losses = []
            for start in range(0, len(order), CONFIG["batch_size"]):
                rows = order[start : start + CONFIG["batch_size"]]
                logits = network(torch.from_numpy(normalized[rows]))
                repeated_y = target[rows, None].expand(-1, 1).reshape(-1)
                member_losses = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, 3), repeated_y, reduction="none"
                ).reshape(len(rows), -1)
                loss = (member_losses.mean(dim=1) * sample_weights[rows]).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite confirmation training loss")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), CONFIG["gradient_norm_limit"])
                optimizer.step()
                losses.append(float(loss.detach()))
            validation, adaptive = {}, {}
            for asset, symbol in enumerate(SYMBOLS):
                p = learner.predict_proba(validation_features[symbol], asset)
                validation[symbol] = classification_metrics(
                    SimpleNamespace(priors=priors[asset]), p, validation_labels[symbol]
                )
                decision_priors = forecast_priors(p, validation_times[symbol], validation_initial_priors[symbol], half_life_seconds=3600)
                adaptive[symbol] = classification_metrics(
                    SimpleNamespace(priors=decision_priors), p, validation_labels[symbol]
                )
            registered_ba = float(np.mean([m["balanced_accuracy"] for m in validation.values()]))
            forecast_ba = float(np.mean([m["balanced_accuracy"] for m in adaptive.values()]))
            natural_ll = float(np.mean([m["log_loss"] for m in validation.values()]))
            keys = {"registered": (registered_ba, -natural_ll), "forecast_3600": (forecast_ba, -natural_ll),
                    "natural_log_loss": (-natural_ll, registered_ba)}
            for name, key in keys.items():
                if key > best_key[name]:
                    best_key[name], best_state[name], best_epoch[name] = key, copy.deepcopy(network.state_dict()), epoch
            history.append({"epoch": epoch, "mean_training_loss": float(np.mean(losses)),
                            "validation": validation, "validation_forecast_3600": adaptive})
    selected = {}
    for name in SELECTORS:
        chosen = build_member_network(normalized.shape[1], members=1, hidden_size=CONFIG["hidden_size"])
        chosen.load_state_dict(best_state[name])
        chosen.eval()
        selected[name] = PooledForecaster(chosen, normalizer, active.copy(), columns, priors, 1, CONFIG["hidden_size"])
    return selected, {"history": history, "best_epochs": best_epoch, "config": dict(CONFIG), "model_fits": 1,
                      "selection_procedures": len(SELECTORS), "gradient_path_shared": True}
