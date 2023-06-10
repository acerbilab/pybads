# PyBADS Example 1: Basic usage
# (code only - see Jupyter notebook for the tutorial)

import numpy as np

from pybads import BADS


def rosenbrocks_fcn(x):
    """Rosenbrock's 'banana' function in any dimension."""
    x_2d = np.atleast_2d(x)
    return np.sum(
        100 * (x_2d[:, 0:-1] ** 2 - x_2d[:, 1:]) ** 2
        + (x_2d[:, 0:-1] - 1) ** 2,
        axis=1,
    )


target = rosenbrocks_fcn

lower_bounds = np.array([-20, -20])
upper_bounds = np.array([20, 20])
plausible_lower_bounds = np.array([-5, -5])
plausible_upper_bounds = np.array([5, 5])
x0 = np.array([0, 0])
# Starting point

bads = BADS(target, x0, lower_bounds, upper_bounds, plausible_lower_bounds, plausible_upper_bounds)
optimize_result = bads.optimize()

x_min = optimize_result["x"]
fval = optimize_result["fval"]

print(f"BADS minimum at: x_min = {x_min.flatten()}, fval = {fval:.4g}")
print(
    f"total f-count: {optimize_result['func_count']}, time: {round(optimize_result['total_time'], 2)} s"
)
