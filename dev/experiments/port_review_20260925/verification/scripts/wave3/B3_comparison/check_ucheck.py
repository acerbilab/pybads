import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
from pybads.function_logger.constraints_check import contraints_check


class FL:  # minimal stand-in for FunctionLogger
    def __init__(self, X):
        self.X = np.vstack([X, np.full((5, X.shape[1]), np.nan)])
        self.X_max_idx = X.shape[0] - 1
        self.variable_transformer = None


def ucheck_matlab(U, tol_mesh, lb_s, ub_s, Ulog):
    """Transcription of utils/uCheck.m with proj=1 and no nonbcon."""
    U = np.maximum(np.minimum(U, ub_s), lb_s)
    # unique(U,'rows'): sorted unique rows
    U = np.unique(U, axis=0)
    if U.size > 0:
        tol = tol_mesh / 2
        u1 = np.round(U / tol)
        u2 = np.round(Ulog / tol)
        # [~,idx] = setdiff(u1,u2,'rows'): rows of u1 not in u2, unique, sorted
        keep = {}
        set2 = {tuple(r) for r in u2}
        for i, r in enumerate(u1):
            t = tuple(r)
            if t not in set2 and t not in keep:
                keep[t] = i
        idx = [keep[t] for t in sorted(keep)]
        U = U[idx]
    return U


D = 2
lb_s = np.full((1, D), -1.0)
ub_s = np.full((1, D), 1.0)
Ulog = np.array([[0.0, 0.0], [0.5, 0.25], [-0.25, 0.75]])
U = np.array(
    [[0.0, 0.0], [0.5, 0.25], [0.125, 0.125], [0.125, 0.125], [2.0, 0.0]]
)
tol_mesh = 1e-6
py = contraints_check(U, lb_s, ub_s, tol_mesh, FL(Ulog), True, None)
ml = ucheck_matlab(U, tol_mesh, lb_s, ub_s, Ulog)
print("candidates:\n", U)
print("evaluated:\n", Ulog)
print("Python contraints_check:\n", py)
print("MATLAB uCheck transcription:\n", ml)
