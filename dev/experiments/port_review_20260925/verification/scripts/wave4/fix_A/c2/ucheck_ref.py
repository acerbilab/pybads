"""A transcription of MATLAB BADS's utils/uCheck.m (74919c0), with MATLAB's
round (halves away from zero) and setdiff(u1, u2, 'rows') (sorted rows of
u1 not in u2, the index of the first occurrence of each)."""
import numpy as np


def mround(q):
    """MATLAB's round, exact: halves away from zero."""
    frac, r = np.modf(q)
    return r + np.sign(frac) * (np.abs(frac) >= 0.5)


def ucheck(
    U,
    tol_mesh,
    X_eval,
    lb_s,
    ub_s,
    lb,
    ub,
    proj=True,
    nonbcon=None,
    origunits=None,
    rounding=mround,
):
    U = np.atleast_2d(U)
    if proj:
        U = np.maximum(np.minimum(U, ub_s), lb_s)
    else:
        idx = np.any((U > ub) | (U < lb), axis=1)
        U = U[~idx]
    # unique(U, 'rows'): sorted unique rows (lexicographic)
    if U.shape[0] > 0:
        order = np.lexsort(U.T[::-1])
        U = U[order]
        keep = np.ones(len(U), dtype=bool)
        keep[1:] = np.any(U[1:] != U[:-1], axis=1)
        U = U[keep]
    if U.size > 0:
        t = tol_mesh / 2
        u1 = rounding(U / t)
        u2 = rounding(X_eval / t)
        set2 = {tuple(r) for r in (u2 + 0.0)}  # +0.0: -0 == 0 as in MATLAB
        seen = {}
        for i, r in enumerate(u1 + 0.0):
            k = tuple(r)
            if k in set2 or k in seen:
                continue
            seen[k] = i
        keys = sorted(seen.keys())
        U = U[[seen[k] for k in keys]].reshape(-1, U.shape[1])
    if nonbcon is not None:
        C = np.ravel(nonbcon(origunits(U)))
        U = U[C <= 0]
    return U
