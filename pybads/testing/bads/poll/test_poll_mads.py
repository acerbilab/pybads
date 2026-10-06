import numpy as np
import pytest

from pybads.poll import poll_mads_2n


def _check_poll_set(B, D, poll_scale, n_max):
    """`B` holds a basis of LTMADS directions and its negation, divided by
    `poll_scale`: undoing the division gives integers of magnitude at most
    `n_max`, with determinant `+-n_max**D`, so that the 2D directions
    positively span the space."""
    assert B.shape == (2 * D, D)
    assert np.array_equal(B[D:], -B[:D])
    basis = B[:D] * poll_scale
    assert np.allclose(basis, np.round(basis))
    assert np.max(np.abs(basis)) == pytest.approx(n_max)
    assert abs(np.linalg.det(basis)) == pytest.approx(n_max**D)


def test_poll_mads_2n_ones():
    poll_scale = np.ones((1, 3))
    search_mesh_size = 9.7656e-4
    mesh_size = 1.0
    D = 3
    B = poll_mads_2n(
        D,
        poll_scale,
        search_mesh_size,
        mesh_size,
        rng=np.random.default_rng(0),
    )
    _check_poll_set(B, D, poll_scale, n_max=1)


def test_poll_mads_2n():
    poll_scale = np.array([[0.5133, 0.493, 3.9511]])
    search_mesh_size = 9.7656e-4
    mesh_size = 0.0312
    D = 3
    B = poll_mads_2n(
        D,
        poll_scale,
        search_mesh_size,
        mesh_size,
        rng=np.random.default_rng(0),
    )
    _check_poll_set(B, D, poll_scale, n_max=1)


def test_poll_mads_2n_dense():
    """A search mesh coarser than the poll mesh gives `n_max > 1`, where the
    basis can have more than D nonzero entries (at `n_max = 1` it is a
    signed permutation matrix)."""
    poll_scale = np.array([[0.5133, 0.493, 3.9511]])
    search_mesh_size = 1.0
    mesh_size = 0.25
    D = 3
    B = poll_mads_2n(
        D,
        poll_scale,
        search_mesh_size,
        mesh_size,
        rng=np.random.default_rng(0),
    )
    _check_poll_set(B, D, poll_scale, n_max=4)
    assert np.count_nonzero(B[:D]) > D
