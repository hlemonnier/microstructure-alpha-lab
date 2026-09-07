import pytest

np = pytest.importorskip("numpy")

from lob_forge.boundary_direction_transfer import STRENGTHS, transfer_direction  # noqa: E402


def test_zero_strength_is_exact_and_transfer_preserves_movement_mass():
    own = np.array([[0.2, 0.5, 0.3], [0.3, 0.4, 0.3]])
    peer = np.array([[0.1, 0.2, 0.7], [0.6, 0.1, 0.3]])
    np.testing.assert_array_equal(transfer_direction(own, peer, strength=0), own)
    changed = transfer_direction(own, peer, strength=0.5)
    np.testing.assert_array_equal(changed[:, 1], own[:, 1])
    np.testing.assert_allclose(changed[:, [0, 2]].sum(axis=1), own[:, [0, 2]].sum(axis=1))
    np.testing.assert_allclose(changed.sum(axis=1), 1)
    assert changed[0, 2] > own[0, 2]
    assert changed[1, 2] < own[1, 2]
    np.testing.assert_array_equal(own, [[0.2, 0.5, 0.3], [0.3, 0.4, 0.3]])


def test_direction_transfer_respects_sign_reflection_and_row_prefixes():
    own = np.array([[0.1, 0.6, 0.3], [0.3, 0.5, 0.2]])
    peer = np.array([[0.2, 0.1, 0.7], [0.6, 0.3, 0.1]])
    for strength in STRENGTHS:
        full = transfer_direction(own, peer, strength=strength)
        reflected = transfer_direction(own[:, ::-1], peer[:, ::-1], strength=strength)
        np.testing.assert_allclose(reflected, full[:, ::-1], atol=1e-15)
        np.testing.assert_array_equal(full[:1], transfer_direction(own[:1], peer[:1], strength=strength))


def test_zero_probability_components_remain_finite_and_invalid_sources_fail():
    own = np.array([[0.0, 1.0, 0.0], [0, 0, 1], [1, 0, 0]])
    peer = own[::-1]
    for strength in STRENGTHS:
        p = transfer_direction(own, peer, strength=strength)
        assert np.isfinite(p).all()
        assert (p >= 0).all()
        np.testing.assert_allclose(p.sum(axis=1), 1)
    with pytest.raises(ValueError):
        transfer_direction(own, peer[:1], strength=0.5)
    with pytest.raises(ValueError):
        transfer_direction(own, peer * 2, strength=0.5)
