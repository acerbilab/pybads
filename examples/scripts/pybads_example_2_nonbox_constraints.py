# PyBADS Example 2: Non-box constraints
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


x0 = np.array([0, 0])  # Starting point
lower_bounds = np.array([-1, -1])
upper_bounds = np.array([1, 1])


def circle_constr(x):
    """Return constraints violation outside the unit circle."""
    x_2d = np.atleast_2d(x)
    # Note that nonboxcons assumes the function takes a 2D input
    return np.sum(x_2d**2, axis=1) > 1


options = {}
options["rng_seed"] = 3
bads = BADS(rosenbrocks_fcn, x0, lower_bounds, upper_bounds, non_box_cons=circle_constr)
optimize_result = bads.optimize()

x_min = optimize_result["x"]
fval = optimize_result["fval"]

print(f"BADS minimum at: x_min = {x_min.flatten()}, fval = {fval:.4g}")
print(
    f"total f-count: {optimize_result['func_count']}, time: {round(optimize_result['total_time'], 2)} s"
)
print(f"Problem type: {optimize_result['problem_type']}")
