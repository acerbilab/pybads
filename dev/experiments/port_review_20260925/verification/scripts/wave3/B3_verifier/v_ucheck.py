"""F1 / B3-K3 / B3-K4: contraints_check against a transcription of uCheck.m."""
from types import SimpleNamespace

import numpy as np
import vhdr  # noqa

from pybads.function_logger.constraints_check import contraints_check


def ucheck_matlab(
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
):
    """Transcription of utils/uCheck.m (74919c0). setdiff(...,'rows') returns
    the rows of u1 not in u2, sorted, with ia pointing at the first occurrence.
    """
    U = np.atleast_2d(U)
    if proj:
        U = np.maximum(np.minimum(U, ub_s), lb_s)
    else:
        idx = np.any((U > ub) | (U < lb), axis=1)
        U = U[~idx]
    # unique(U,'rows'): sorted unique rows
    U = np.unique(U, axis=0)
    if U.size > 0:
        t = tol_mesh / 2
        u1 = np.round(
            U / t
        )  # MATLAB round: halves away from zero; no halves here
        u2 = np.round(X_eval / t)
        # setdiff rows: sorted unique rows of u1 not in u2, first occurrence
        set2 = {tuple(r) for r in u2}
        seen = {}
        for i, r in enumerate(u1):
            k = tuple(r)
            if k in set2 or k in seen:
                continue
            seen[k] = i
        keys = sorted(seen.keys())
        idx = [seen[k] for k in keys]
        U = U[idx]
    if nonbcon is not None:
        C = np.ravel(nonbcon(origunits(U)))
        U = U[C <= 0]
    return U


D = 2
tol_mesh = 2.0**-19
X_eval = np.array([[0.5, 0.25], [0.0, 0.0], [-0.25, 0.75]])
fl = SimpleNamespace(X=np.vstack([X_eval, np.zeros((5, D))]), X_max_idx=2)
lb_s = np.full((1, D), -1.0)
ub_s = np.full((1, D), 1.0)

U = np.array(
    [
        [0.5, 0.25],  # evaluated
        [0.75, 0.0],
        [0.0, 0.0],  # evaluated
        [0.0, 0.0],  # exact duplicate
        [0.0, 1e-8],  # same tol bin as [0,0] (evaluated)
        [0.25, -0.25],
        [2.0, 0.0],  # projected to [1, 0]
        [-0.25, 0.75 + 3e-7],
    ]
)  # same bin as evaluated [-0.25,0.75]? (bin 2^-20 ~ 9.5e-7)
py = contraints_check(U, lb_s, ub_s, tol_mesh, fl, True)
ml = ucheck_matlab(U, tol_mesh, X_eval, lb_s, ub_s, None, None, True)
print("input rows:\n", U)
print("PyBADS contraints_check:\n", py)
print("MATLAB uCheck transcription:\n", ml)
ev = {tuple(np.round(r / (tol_mesh / 2))) for r in X_eval}
print(
    "PyBADS rows whose bin was evaluated:",
    sum(tuple(np.round(r / (tol_mesh / 2))) in ev for r in py),
)
print(
    "MATLAB rows whose bin was evaluated:",
    sum(tuple(np.round(r / (tol_mesh / 2))) in ev for r in ml),
)

# Order check (B3-K4): random candidates, nothing evaluated in range
rng = np.random.default_rng(0)
fl2 = SimpleNamespace(X=np.full((3, D), 50.0), X_max_idx=2)
agree = 0
input_order = 0
for trial in range(200):
    U2 = np.round(rng.uniform(-1, 1, size=(30, D)) * 64) / 64
    p = contraints_check(U2, lb_s, ub_s, tol_mesh, fl2, True)
    m = ucheck_matlab(U2, tol_mesh, fl2.X, lb_s, ub_s, None, None, True)
    agree += p.shape == m.shape and np.array_equal(p, m)
    # would the commented-out np.sort variant (input order) agree?
    _, first = np.unique(U2, axis=0, return_index=True)
    inorder = U2[np.sort(first)]
    input_order += inorder.shape == m.shape and np.array_equal(inorder, m)
print(
    f"order: PyBADS == MATLAB on {agree}/200 random sets; input-order variant == MATLAB on {input_order}/200"
)

# Representative within one bin when the grid is finer than the bin
fl3 = SimpleNamespace(X=np.full((1, D), 50.0), X_max_idx=0)
tiny = 2.0**-24
U3 = np.array([[0.1 + 3 * tiny, 0.2], [0.1 + tiny, 0.2]])
print(
    "same-bin representative: PyBADS",
    contraints_check(U3, lb_s, ub_s, tol_mesh, fl3, True).tolist(),
    "MATLAB",
    ucheck_matlab(U3, tol_mesh, fl3.X, lb_s, ub_s, None, None, True).tolist(),
)
