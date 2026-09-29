import numpy as np
from scipy.spatial.distance import cdist

from pybads.rounding import round_half_away
from pybads.utils.period_check import period_check
from pybads.variable_transformer import VariableTransformer


def force_to_grid(x, search_mesh_size, tol=None):
    """
    Put ``x`` on the grid of step ``tol``, as MATLAB BADS's ``force2grid``.

    Each coordinate goes to the nearest multiple of ``tol``; one halfway
    between two multiples goes to the one farther from zero, as MATLAB's
    ``round`` takes it, a rounding that
    ``pybads.rounding.round_half_away`` computes exactly.

    Parameters
    ----------
    x : np.ndarray
        The coordinates, of any shape.
    search_mesh_size : float
        The step of the grid if ``tol`` is ``None``.
    tol : float, optional
        The step of the grid. If ``None`` (default), ``search_mesh_size``.

    Returns
    -------
    x_grid : np.ndarray
        The multiples of ``tol`` nearest to ``x``, of the shape of ``x``.
    """
    if tol is None:
        tol = search_mesh_size

    # MATLAB's round (force2grid.m), which takes halves away from zero
    return tol * round_half_away(x / tol)


def force_to_grid_periodic(u, search_mesh_size, lb, ub, periodic_vars):
    """
    Wrap the periodic coordinates of ``u`` into their period and put ``u`` on
    the grid, as MATLAB BADS's ``periodCheck`` and ``force2grid`` do.

    The grid can take a periodic coordinate to its upper bound, the same
    point as its lower bound, or past a bound, so the periodic coordinates
    are wrapped and put on the grid a second time: they stay on the grid,
    and one that lands on the upper bound becomes the lower bound where the
    grid holds it, so that a point already evaluated there is recognized.
    A coordinate that the second step puts out of bounds is left to the
    projection of ``contraints_check``.

    Parameters
    ----------
    u : np.ndarray
        The points, one per row.
    search_mesh_size : float
        The step of the grid.
    lb, ub : np.ndarray
        The bounds, of shape ``(1, D)`` or ``(D,)``, whose width is the period
        of a periodic variable.
    periodic_vars : np.ndarray or None
        The boolean mask of the periodic variables, of shape ``(1, D)`` or
        ``(D,)``, or ``None``.

    Returns
    -------
    u_grid : np.ndarray
        The points on the grid; without periodic variables,
        ``force_to_grid(u, search_mesh_size)``.
    """
    u = force_to_grid(period_check(u, lb, ub, periodic_vars), search_mesh_size)
    if periodic_vars is None or not np.any(periodic_vars):
        return u
    mask = np.ravel(periodic_vars).astype(bool)
    u = period_check(u, lb, ub, periodic_vars)
    rows = np.atleast_2d(u)
    rows[:, mask] = force_to_grid(rows[:, mask], search_mesh_size)
    return u


def grid_units(x, var_trans: VariableTransformer = None, x0=None, scale=None):
    """
    grid_units convert vector(s) coordinates to grid-normalized units
    """
    if var_trans is not None:
        if len(x) == 1:
            u = var_trans(x)
        else:
            # var_trans.D columns: a transform with fixed variables leaves
            # them out of its points
            u = np.zeros((x.shape[0], var_trans.D))
            for i in range(0, len(x)):
                u[i, :] = var_trans(x[i, :])

    else:
        u = (x - x0) / scale
    return u


def udist(U, u2, len_scale, lb, ub, bound_scale, periodic_vars):
    """
    Squared distances between the rows of ``U`` and of ``u2``, in units of
    ``len_scale``, as MATLAB BADS's ``udist``.

    Along a periodic variable the difference is taken the shorter way round
    its period, ``(ub - lb) / bound_scale``.

    Parameters
    ----------
    U : np.ndarray
        The points, of shape ``(N, D)``.
    u2 : np.ndarray
        The points, of shape ``(M, D)``, or one point of shape ``(D,)``.
    len_scale : float or np.ndarray
        The length scale, one for all the variables or one per variable.
    lb, ub : np.ndarray
        The bounds, of shape ``(1, D)`` or ``(D,)``, which set the period of
        a periodic variable.
    bound_scale : float
        The scale of the grid (``optim_state["scale"]``).
    periodic_vars : np.ndarray or None
        The boolean mask of the periodic variables, of shape ``(1, D)`` or
        ``(D,)``, or ``None``.

    Returns
    -------
    dist : np.ndarray
        The squared distances, of shape ``(N, M)``.
    """
    if periodic_vars is not None and np.any(periodic_vars):
        mask = np.ravel(periodic_vars).astype(bool)
        A = np.atleast_2d(U)
        B = np.atleast_2d(u2)
        # The differences of every pair, shape (N, M, D)
        diff = np.abs(A[:, None, :] - B[None, :, :])
        period = (np.ravel(ub) - np.ravel(lb))[mask] / bound_scale
        wrapped = np.mod(diff[:, :, mask], period)
        diff[:, :, mask] = np.minimum(wrapped, period - wrapped)
        return np.sum((diff / np.ravel(len_scale)) ** 2, axis=2)

    else:
        dist = cdist(
            np.atleast_2d(U) / len_scale, np.atleast_2d(u2) / len_scale
        )
        return dist**2
