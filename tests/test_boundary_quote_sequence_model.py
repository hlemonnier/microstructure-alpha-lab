import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from lob_forge.boundary_quote_sequence_model import (  # noqa: E402
    ENCODERS,
    SEQUENCE_SHAPE,
    QuoteSequenceForecaster,
    build_quote_sequence_network,
    fit_quote_sequence_model,
    fit_sequence_scaler,
    normalize_sequences,
)


def test_sequence_scaling_ignores_padding_and_preserves_observed_mask():
    seq = np.zeros((2, *SEQUENCE_SHAPE), dtype=np.float32)
    seq[0, :, -2:, :] = 2
    seq[1, :, -1:, :] = 8
    seq[..., -1] = (seq[..., 0] != 0).astype(np.float32)
    mean, scale = fit_sequence_scaler(seq)
    np.testing.assert_allclose(mean[:, :-1], 4)
    np.testing.assert_allclose(scale[:, :-1], np.sqrt(8))
    transformed = normalize_sequences(seq, mean, scale)
    np.testing.assert_array_equal(transformed[..., -1], seq[..., -1])
    np.testing.assert_array_equal(transformed[seq[..., -1] == 0], 0)
    # New assessment extremes cannot alter the frozen training statistics.
    extreme = seq.copy()
    extreme[..., 0] *= 100000
    normalize_sequences(extreme, mean, scale)
    np.testing.assert_allclose(mean[:, :-1], 4)


def test_pooled_control_is_invariant_to_nonfinal_event_permutation():
    torch.manual_seed(9)
    seq = torch.randn(3, *SEQUENCE_SHAPE)
    seq[..., -1] = 1
    context = torch.randn(3, 4)
    order = torch.cat([torch.arange(62, -1, -1), torch.tensor([63])])
    pooled = build_quote_sequence_network(4, "pooled_events").eval()
    np.testing.assert_allclose(pooled(context, seq).detach().numpy(),
        pooled(context, seq[:, :, order]).detach().numpy(), rtol=1e-5, atol=1e-6)
    temporal = build_quote_sequence_network(4, "event_tcn").eval()
    assert not np.allclose(temporal(context, seq).detach().numpy(), temporal(context, seq[:, :, order]).detach().numpy())


@pytest.mark.parametrize("encoder", ENCODERS)
def test_sequence_models_preserve_training_inputs_priors_and_checkpoints(tmp_path, encoder):
    rng = np.random.default_rng(19)
    seq = rng.normal(size=(48, *SEQUENCE_SHAPE)).astype(np.float32)
    seq[..., -1] = 1
    seq[:, :, :5] = 0
    frame = pd.DataFrame({"x": np.arange(48), "y": np.sin(np.arange(48))})
    labels = np.array([-1] * 12 + [0] * 24 + [1] * 12)
    symbols = ("BTCUSDT", "ETHUSDT")
    x, s, y = {k: frame.copy() for k in symbols}, {k: seq.copy() for k in symbols}, {k: labels.copy() for k in symbols}
    validation = {k: frame.iloc[:18].copy() for k in symbols}
    validation_s, validation_y = {k: seq[:18].copy() for k in symbols}, {k: np.tile([-1, 0, 1], 6) for k in symbols}
    model, evidence = fit_quote_sequence_model(x, s, y, validation, validation_s, validation_y,
        encoder=encoder, config={"epochs": 2, "batch_size": 24})
    model.save(tmp_path / encoder)
    restored = QuoteSequenceForecaster.load(tmp_path / encoder)
    for asset, symbol in enumerate(symbols):
        np.testing.assert_array_equal(model.priors[asset], [0.25, 0.5, 0.25])
        p = model.predict_proba(validation[symbol], validation_s[symbol], asset)
        np.testing.assert_array_equal(p, restored.predict_proba(validation[symbol], validation_s[symbol], asset))
        np.testing.assert_allclose(p.sum(axis=1), 1)
        np.testing.assert_array_equal(s[symbol], seq)
        pd.testing.assert_frame_equal(x[symbol], frame)
    assert evidence["best_epoch"] in [1, 2]
    with pytest.raises(ValueError):
        model.predict_proba(validation[symbol], validation_s[symbol][:-1], asset)
