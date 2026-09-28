from typing import Callable

import numpy as np

from pybads.rounding import round_half_away

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
    them (``proj=True``) or removed. The candidates are then binned on a
    grid of ``tol_mesh / 2``, as in MATLAB BADS's ``uCheck``: the first
    candidate of each bin is kept (``uCheck`` keeps the smallest), unless
    the bin holds a point already evaluated, and the bins come out sorted.
    Duplicates share a bin, so only the first of them is kept. Last, the
    candidates that violate ``non_box_cons`` are removed.
    """

    if proj:
        # Project vectors outside bounds on search mesh points closest to
        # bounds
        U_new = np.maximum(np.minimum(U, ub), lb)
    else:
        idx = np.any(U > ub, axis=1) | np.any(U < lb, axis=1)
        U_new = U[~idx].copy()

    # Remove duplicate and previously evaluated vectors (within tol_mesh):
    # bin the vectors on a grid of tol_mesh / 2, rounding halves away from
    # zero as uCheck.m's round does, and keep the first vector of each bin
    # that holds no evaluated vector, the bins sorted, as MATLAB's
    # setdiff(u1, u2, 'rows')
    if U_new.size > 0:
        n_u = U_new.shape[0]
        X_evaluated = function_logger.X[: function_logger.X_max_idx + 1]
        tol = tol_mesh / 2.0
        bins = round_half_away(np.vstack((U_new, X_evaluated)) / tol)
        order = _lexsort_rows(bins)
        sorted_bins = bins[order]
        starts = np.ones(len(order), dtype=bool)
        starts[1:] = np.any(sorted_bins[1:] != sorted_bins[:-1], axis=1)
        # The sort is stable, so each bin starts with its first vector
        first = order[starts]
        evaluated = np.zeros(len(first), dtype=bool)
        evaluated[np.cumsum(starts)[order >= n_u] - 1] = True
        U_new = U_new[first[(first < n_u) & ~evaluated]]

    if non_box_cons is not None:
        if function_logger is None:
            raise ValueError(
                "contraints_check: function_logger not passed, non bondcons "
                "requires it."
            )
        X = function_logger.variable_transformer.inverse_transf(U_new)
        # one violation per point, of shape (N,) or (N, 1)
        C = np.ravel(non_box_cons(X))
        idx = C <= 0
        U_new = U_new[idx]

    return U_new


def _lexsort_rows(A: np.ndarray):
    """
    Return the stable order of the rows of ``A``, sorted lexicographically
    from the first column, as ``np.lexsort(A.T[::-1])`` does.

    The rows are sorted by their first column, then only those that tie on
    it by all the columns, which is faster when few rows tie.
    """
    order = np.argsort(A[:, 0], kind="stable")
    a0 = A[order, 0]
    # Re-sort by every column the rows that are not strictly between their
    # neighbours in the first column: those with equal first values, and
    # NaNs, which sort last. The rows being sorted by the first column, this
    # moves a row only within its run of equal first values (or of NaNs).
    tie = ~(a0[1:] > a0[:-1])
    tied = np.zeros(len(order), dtype=bool)
    tied[1:] = tie
    tied[:-1] |= tie
    sub = order[tied]
    order[tied] = sub[np.lexsort(A[sub].T[::-1])]
    return order
