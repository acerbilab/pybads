import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
from pybads.function_logger import FunctionLogger, contraints_check

fl = FunctionLogger(lambda x: float(np.sum(x**2)), 2, False, 0)
for x in [np.array([0.5, 0.25]), np.array([0.0, 0.0]), np.array([-0.5, 1.0])]:
    fl(x)
U = np.array(
    [
        [0.5, 0.25],  # evaluated
        [0.75, 0.0],  # new
        [0.0, 0.0],  # evaluated
        [0.0, 0.0],  # duplicate
        [0.0, 0.0 + 1e-8],  # evaluated within tol_mesh/2
        [0.25, -0.25],
    ]
)  # new
lb = np.full((1, 2), -1.0)
ub = np.full((1, 2), 1.0)
out = contraints_check(U, lb, ub, 1e-6, fl, True, None)
print("input\n", U, "\noutput\n", out)
# order check
U2 = np.array([[0.9, 0.1], [0.1, 0.9], [-0.3, 0.2]])
print(
    "order in:",
    U2.tolist(),
    "out:",
    contraints_check(U2, lb, ub, 1e-6, fl, True, None).tolist(),
)
