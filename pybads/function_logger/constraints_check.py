from multiprocessing.sharedctypes import Value
from typing import Callable

import numpy as np

from .function_logger import FunctionLogger


def contraints_check(
    U: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    tol_mesh,
    function_logger: FunctionLogger,
    proj=True,
    non_box_cons: Callable = None,
):
    """
    Return the candidates ``U`` that the search or the poll may evaluate.

    The candidates outside the bounds ``lb`` and ``ub`` are projected onto
    them (``proj=True``) or removed, and duplicates are removed. The
    candidates are then binned on a grid of ``tol_mesh / 2``: the first of
    each bin is kept, unless the bin holds a point already evaluated, and
    the bins come out sorted, as from MATLAB BADS's ``uCheck``. Last, the
    candidates that violate ``non_box_cons`` are removed.
    """

    if proj:
        # Project vectors outside bounds on search mesh points closest to bounds
        U_new = np.maximum(np.minimum(U, ub), lb)
    else:
        idx = np.any(U > ub, axis=1) | np.any(U < lb, axis=1)
        U_new = U[~idx].copy()

    # Remove duplicate vectors, keeping the first of each (the removal of the
    # evaluated vectors below returns them sorted, as MATLAB's setdiff does)
    _, idx_sort = np.unique(U_new, axis=0, return_index=True)
    U_new = U_new[np.sort(idx_sort), :]

    # Remove previously evaluated vectors (within tol_mesh): keep the first
    # vector of each bin that holds no evaluated vector, the bins sorted, as
    # MATLAB's setdiff(u1, u2, 'rows'). np.round takes a half to the even
    # integer, where uCheck.m's round takes it away from zero, so the bins
    # differ from MATLAB's once the search mesh is finer than tol_mesh / 2
    if U_new.size > 0:
        tol = tol_mesh / 2.0
        u1 = np.round(U_new / tol)
        X_max_idx = function_logger.X_max_idx
        u2 = np.round(function_logger.X[: X_max_idx + 1] / tol)
        tmp_u = np.vstack((u1, u2))
        _, idx_sort, idx_bin = np.unique(
            tmp_u, axis=0, return_index=True, return_inverse=True
        )
        evaluated = np.zeros(len(idx_sort), dtype=bool)
        evaluated[np.ravel(idx_bin)[len(u1) :]] = True
        u1_idx = idx_sort[(idx_sort < len(u1)) & ~evaluated]
        U_new = U_new[u1_idx]

    if non_box_cons is not None:
        if function_logger is None:
            raise ValueError(
                "contraints_check: function_logger not passed, non bondcons requires it."
            )
        X = function_logger.variable_transformer.inverse_transf(U_new)
        # one violation per point, of shape (N,) or (N, 1)
        C = np.ravel(non_box_cons(X))
        idx = C <= 0
        U_new = U_new[idx]

    return U_new
