"""I-F6 / C-F2: contraints_check (proj=False, the poll's call) keeps points
already in the function log; MATLAB's uCheck (utils/uCheck.m:17-27) drops
them. Transcription of uCheck's set difference, compared on the same input."""
import numpy as np
import vhdr

from pybads.function_logger import FunctionLogger
from pybads.function_logger.constraints_check import contraints_check

D = 2
fl = FunctionLogger(lambda x: float(np.sum(np.ravel(x) ** 2)), D, False, 0)
evaluated = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, -0.5]])
for x in evaluated:
    fl(x)
u = np.zeros((1, D))
mesh = 0.5
U = np.vstack([u + mesh * np.eye(D), u - mesh * np.eye(D)])  # the 2D poll set
lb, ub, tol_mesh = -np.ones(D), np.ones(D), 1e-6
port = contraints_check(U, lb, ub, tol_mesh, fl, False, None)


def ucheck_matlab(U, lb, ub, tol_mesh, Uevals):
    idx = np.any((U > ub) | (U < lb), axis=1)
    U = U[~idx]
    U = np.unique(U, axis=0)  # unique(U,'rows'), sorted
    if U.size:
        tol = tol_mesh / 2
        u1 = np.round(U / tol)
        u2 = np.round(Uevals / tol)
        keep = [
            i
            for i in range(len(u1))
            if not np.any(np.all(u2 == u1[i], axis=1))
        ]
        U = U[keep]  # setdiff(u1,u2,'rows') returns sorted rows
    return U


mat = ucheck_matlab(U, lb, ub, tol_mesh, fl.X[: fl.X_max_idx + 1])
print("poll set:\n", U)
print("port keeps:\n", port)
print("MATLAB keeps:\n", mat)
print(
    "port keeps evaluated rows:",
    [r.tolist() for r in port if np.any(np.all(np.isclose(evaluated, r), 1))],
)
