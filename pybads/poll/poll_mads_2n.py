import numpy as np

from pybads.rng import get_rng


def poll_mads_2n(dim_x, poll_scale, search_mesh_size, mesh_size, rng=None):
    """
    Draw the poll basis of MATLAB BADS's ``pollMADS2N``, and its negation.

    The basis has the shape of LTMADS's [1]: up to a permutation of the
    coordinates, the transpose of a lower-triangular integer matrix whose
    diagonal entries are ``n_max`` or ``-n_max`` and whose entries below
    the diagonal are drawn uniformly from ``-n_max + 1, ..., n_max - 1``.
    With its negation, it gives ``2D`` directions that positively span the
    space, which the poll takes in units of ``mesh_size``. As in MATLAB
    BADS (``pollMADS2N.m:7``), the bound is ``n_max = max(1,
    round(search_mesh_size / mesh_size))``. The search mesh is finer than
    the poll mesh at every default state, so ``n_max`` is 1 and the basis
    is a signed permutation of the identity: the poll steps along one
    coordinate at a time. LTMADS bounds the basis by the inverse ratio, the
    poll size over the mesh size; PyBADS keeps MATLAB BADS's poll, which
    LTMADS's directions made worse on PyBADS's benchmark
    (``dev/experiments/port_review_20260925/verification/wave3.md``,
    W3-24). A new basis is drawn at each poll.

    The basis is divided by ``poll_scale``, which counteracts the poll's
    multiplication of the directions by ``poll_scale``.

    The random draws come from ``rng``, a ``numpy.random.Generator``; if
    ``None``, a generator is derived from NumPy's global random state
    (``pybads.rng.get_rng``).

    Parameters
    ----------
    dim_x : int
        The number of variables ``D``.
    poll_scale : np.ndarray
        The poll's scale of each variable, of shape ``(D,)`` or ``(1, D)``.
    search_mesh_size : float
        The size of the search mesh.
    mesh_size : float
        The size of the poll mesh.
    rng : numpy.random.Generator, optional
        The generator of the random draws.

    Returns
    -------
    B_new : np.ndarray
        The basis and its negation, divided by ``poll_scale``, of shape
        ``(2D, D)``.

    References
    ----------
    .. [1] Audet, C., & Dennis, J. E., Jr. (2006). Mesh adaptive direct
       search algorithms for constrained optimization. SIAM Journal on
       Optimization, 17(1), 188-217. https://doi.org/10.1137/040603371
    """
    rng = get_rng(rng)
    n_max = np.maximum(1, np.round(search_mesh_size / mesh_size))

    if n_max > 0:
        D = rng.integers(1, n_max * 2, size=(dim_x, dim_x)) - n_max
        D = np.tril(D, -1)
    else:
        D = np.zeros((dim_x), dtype="float")

    diag = n_max * 2 * (rng.integers(1, 3, dim_x) - 1.5)
    D = D + np.eye(dim_x) * diag

    # Random permutation of the rows and then transpose (MATLAB BADS also
    # permutes the columns, which only reorders the directions)
    D = np.transpose(rng.permutation(D))

    # Counteract subsequent multiplication by pollscale
    D = D / poll_scale
    B_new = np.vstack((D, -D))
    return B_new
