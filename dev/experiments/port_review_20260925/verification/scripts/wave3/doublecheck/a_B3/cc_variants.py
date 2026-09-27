"""Variants of contraints_check for comparison: the port's code with
MATLAB's round in its bins, and uCheck.m transcribed."""
import numpy as np
from ucheck_ref import mround, ucheck


def port_variant(
    U,
    lb,
    ub,
    tol_mesh,
    function_logger,
    proj=True,
    non_box_cons=None,
    rounding=np.round,
):
    """The code of contraints_check at 0d866e8, with the rounding of its
    bins as a parameter."""
    if proj:
        U_new = np.maximum(np.minimum(U, ub), lb)
    else:
        idx = np.any(U > ub, axis=1) | np.any(U < lb, axis=1)
        U_new = U[~idx].copy()
    _, idx_sort = np.unique(U_new, axis=0, return_index=True)
    U_new = U_new[np.sort(idx_sort), :]
    if U_new.size > 0:
        tol = tol_mesh / 2.0
        u1 = rounding(U_new / tol)
        X_max_idx = function_logger.X_max_idx
        u2 = rounding(function_logger.X[: X_max_idx + 1] / tol)
        tmp_u = np.vstack((u1, u2))
        _, idx_sort, idx_bin = np.unique(
            tmp_u, axis=0, return_index=True, return_inverse=True
        )
        evaluated = np.zeros(len(idx_sort), dtype=bool)
        evaluated[np.ravel(idx_bin)[len(u1) :]] = True
        u1_idx = idx_sort[(idx_sort < len(u1)) & ~evaluated]
        U_new = U_new[u1_idx]
    if non_box_cons is not None:
        X = function_logger.variable_transformer.inverse_transf(U_new)
        C = np.ravel(non_box_cons(X))
        U_new = U_new[C <= 0]
    return U_new


def port_mround(*args, **kwargs):
    return port_variant(*args, rounding=mround, **kwargs)


def matlab_ucheck(
    U, lb, ub, tol_mesh, function_logger, proj=True, non_box_cons=None
):
    fl = function_logger
    X_eval = fl.X[: fl.X_max_idx + 1]
    orig = None
    if non_box_cons is not None:
        orig = fl.variable_transformer.inverse_transf
    return ucheck(
        U,
        tol_mesh,
        X_eval,
        lb,
        ub,
        lb,
        ub,
        proj,
        nonbcon=non_box_cons,
        origunits=orig,
    )
