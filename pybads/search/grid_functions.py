import numpy as np
from matplotlib.pyplot import axis
from scipy.spatial.distance import cdist

from pybads.variable_transformer import VariableTransformer


def force_to_grid(x, search_mesh_size, tol=None):

    if tol is None:
        tol = search_mesh_size

    return tol * np.round(x / tol)


def grid_units(x, var_trans: VariableTransformer = None, x0=None, scale=None):
    """
    grid_units convert vector(s) coordinates to grid-normalized units
    """
    if var_trans is not None:
        if len(x) == 1:
            u = var_trans(x)
        else:
            u = np.zeros(x.shape)
            for i in range(0, len(x)):
                u[i, :] = var_trans(x[i, :])

    else:
        u = (x - x0) / scale
    return u


def udist(U, u2, len_scale, lb, ub, bound_scale, periodic_vars):
    """

    Parameters
    --------

    periodic_vars: bool array
    """
    idx_periods = np.nonzero(periodic_vars)[0]
    if len(idx_periods) > 0:
        # Can be improved by using cdist, but we have to be careful with the scaling and
        A = np.atleast_2d(U)
        B = np.atleast_2d(u2)
        A2 = np.sum(A**2, axis=-1)
        B2 = np.sum(B**2, axis=-1)
        diff = A2[:, None] - 2 * A @ B.T + B2[None, :]

        w_s = (ub - lb) / bound_scale  # scaled width
        diff[idx_periods] = np.minimum(
            np.abs(diff[idx_periods]),
            np.sum(np.atleast_2d(w_s) - np.abs(diff)[idx_periods], axis=-1)
            ** 2,
        )

        diff = diff / (len_scale) ** 2
        return diff

    else:
        dist = cdist(
            np.atleast_2d(U) / len_scale, np.atleast_2d(u2) / len_scale
        )
        return dist**2
