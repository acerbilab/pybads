import numpy as np
import pytest

from pybads.poll import poll_mads_2n


def _check_poll_set(B, D, poll_scale, n_max):
    """`B` holds a basis of LTMADS directions and its negation, in units of
    the poll size and divided by `poll_scale`: undoing the division and
    multiplying by `n_max` gives integers of magnitude at most `n_max`, with
    determinant `+-n_max**D`, so that the 2D directions positively span the
    space. Up to a permutation of the coordinates, the basis is the
    transpose of a lower-triangular matrix whose diagonal is `+-n_max`. The
    integers have no common divisor, so that the bound is `n_max` (with a
    smaller bound they would all be multiples of `n_max` over it)."""
    assert B.shape == (2 * D, D)
    assert np.array_equal(B[D:], -B[:D])
    basis = B[:D] * poll_scale * n_max
    assert np.allclose(basis, np.round(basis))
    basis = np.round(basis)
    assert np.max(np.abs(basis)) == pytest.approx(n_max)
    assert abs(np.linalg.det(basis)) == pytest.approx(n_max**D)
    # The entry of magnitude n_max of each direction, in distinct coordinates
    diagonal = np.argmax(np.abs(basis), axis=1)
    assert sorted(diagonal) == list(range(D))
    triangular = basis[:, diagonal].T
    assert np.array_equal(np.abs(np.diag(triangular)), np.full(D, n_max))
    assert np.array_equal(np.tril(triangular), triangular)
    assert np.gcd.reduce(basis.astype(np.int64).ravel()) == 1


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
    _check_poll_set(B, D, poll_scale, n_max=1024)


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
    _check_poll_set(B, D, poll_scale, n_max=32)


def test_poll_mads_2n_coarse_search_mesh():
    """A search mesh coarser than the poll mesh, which no default state
    gives, gives `n_max = 1`: the basis is a signed permutation matrix, and
    the poll steps along the coordinates."""
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
    _check_poll_set(B, D, poll_scale, n_max=1)
    assert np.count_nonzero(B[:D]) == D


@pytest.mark.parametrize("mesh_size_integer", [0, -1, -5, -10])
def test_poll_mads_2n_default_mesh_sizes(mesh_size_integer):
    """At the default mesh sizes, the poll mesh `2**k` and the search mesh
    `2**min(0, 2k - 10)`, the bound is their ratio, `2**(10 - k)`, as in
    LTMADS (MATLAB BADS takes the inverse ratio, and 1). The poll's
    directions, the basis times `mesh_size`, step by `mesh_size` along one
    coordinate each, are not all coordinate directions, and lie on the
    search mesh."""
    D = 4
    poll_scale = np.array([[0.5133, 0.493, 3.9511, 1.0]])
    k = mesh_size_integer
    mesh_size = 2.0**k
    search_mesh_size = 2.0 ** min(0, 2 * k - 10)
    B = poll_mads_2n(
        D,
        poll_scale,
        search_mesh_size,
        mesh_size,
        rng=np.random.default_rng(0),
    )
    _check_poll_set(B, D, poll_scale, n_max=2 ** (10 - k))
    # The poll vectors, as the poll computes them, in search mesh units
    vv = (B * mesh_size) * poll_scale
    steps = vv / search_mesh_size
    assert np.allclose(np.max(np.abs(vv), axis=1), mesh_size)
    assert np.count_nonzero(np.round(steps)) > 2 * D
    assert np.allclose(steps, np.round(steps), rtol=0, atol=1e-6)
