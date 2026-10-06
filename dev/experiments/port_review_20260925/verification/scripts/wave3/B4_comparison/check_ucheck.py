import gpyreg
import numpy as np

import pybads

print(pybads.__file__)
print(gpyreg.__file__)
from pybads.function_logger.constraints_check import contraints_check


class FL:  # minimal stand-in for the function logger
    def __init__(self, X):
        self.X = X
        self.X_max_idx = X.shape[0] - 1


U = np.array([[0.5, 0.0], [0.0, 0.5], [-0.5, 0.0], [0.0, -0.5]])
X_eval = np.array(
    [[0.0, 0.0], [0.5, 0.0], [0.0, -0.5]]
)  # two poll points already evaluated
lb = -np.ones(2)
ub = np.ones(2)
out = contraints_check(U, lb, ub, 1e-6, FL(X_eval), False, None)
print("kept:\n", out)
