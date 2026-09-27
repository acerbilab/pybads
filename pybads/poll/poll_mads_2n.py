import numpy as np
from gpyreg.gaussian_process import GP

from pybads.rng import get_rng


def poll_mads_2n(dim_x, poll_scale, search_mesh_size, mesh_size, rng=None):
    """
    Draw a basis of LTMADS poll directions [1], and its negation.

    The basis's ``D`` rows are the directions, in units of the poll size
    ``mesh_size``. It is an integer matrix divided by ``n_max``: up to a
    permutation of the coordinates, the transpose of a lower-triangular
    matrix whose diagonal entries are ``n_max`` or ``-n_max`` and whose
    entries below the diagonal are drawn uniformly from ``-n_max + 1, ...,
    n_max - 1``. So its diagonal entries are ``+-1`` and the others
    multiples of ``1 / n_max``, and with its negation it gives ``2D``
    directions that positively span the space. As in LTMADS, the bound
    ``n_max = max(1, round(mesh_size / search_mesh_size))`` is the ratio of
    the poll size to the mesh size: each direction steps by ``mesh_size``
    along one coordinate, tilted along the others by multiples of
    ``mesh_size / n_max``, the search mesh size at default, so that the poll
    points lie on the search mesh when the incumbent does. A new basis is
    drawn at each poll.

    PyBADS departs from MATLAB BADS here: ``pollMADS2N.m`` takes the inverse
    ratio, ``search_mesh_size / mesh_size``, which is below 1 at every
    default state, and steps by ``mesh_size``, so that its basis is always a
    signed permutation of the identity and its poll steps along one
    coordinate at a time.

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
        The poll's scale of each variable, of shape ``(1, D)``.
    search_mesh_size : float
        The size of the search mesh.
    mesh_size : float
        The size of the poll mesh, the poll size.
    rng : numpy.random.Generator, optional
        The generator of the random draws.

    Returns
    -------
    B_new : np.ndarray
        The basis and its negation, in units of the poll size and divided
        by ``poll_scale``, of shape ``(2D, D)``.

    References
    ----------
    .. [1] Audet, C., & Dennis, J. E., Jr. (2006). Mesh adaptive direct
       search algorithms for constrained optimization. SIAM Journal on
       Optimization, 17(1), 188-217. https://doi.org/10.1137/040603371
    """
    rng = get_rng(rng)
    n_max = np.maximum(1, np.round(mesh_size / search_mesh_size))

    if n_max > 0:
        D = rng.integers(1, n_max * 2, size=(dim_x, dim_x)) - n_max
        D = np.tril(D, -1)
    else:
        D = np.zeros((dim_x), dtype="float")

    diag = n_max * 2 * (rng.integers(1, 3, dim_x) - 1.5)
    D = D + np.eye(dim_x) * diag

    # Random permutation of rows and then transpose (MATLAB BADS also
    # permutes the columns, which only reorders the directions)
    D = np.transpose(rng.permutation(D))

    # In units of the poll size
    D = D / n_max

    # Counteract subsequent multiplication by pollscale
    D = D / poll_scale
    B_new = np.vstack((D, -D))
    return B_new
