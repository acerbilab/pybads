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


def periodic_fcn(x):
    """Rosenbrock's function of x_1, x_2, plus a cosine of x_3, of period 4,
    and one of x_4, of period 2."""
    x_2d = np.atleast_2d(x)
    return (
        rosenbrocks_fcn(x_2d[:, 0:2])
        + np.cos(np.pi * x_2d[:, 2] / 2)
        + np.cos(np.pi * x_2d[:, 3])
        + 2
    )


lower_bounds = np.array([-10, -5, -2, -1])
upper_bounds = np.array([5, 10, 2, 1])
plausible_lower_bounds = np.array([-2, -2, -2, -1])
plausible_upper_bounds = np.array([2, 2, 2, 1])
x0 = np.array([-3, -3, -1, -1])  # Starting point

options = {
    "periodic_vars": [2, 3],  # The third and fourth variables are periodic
    "random_seed": 0,  # Makes the run reproducible
}


bads = BADS(
    periodic_fcn,
    x0,
    lower_bounds,
    upper_bounds,
    plausible_lower_bounds,
    plausible_upper_bounds,
    options=options,
)
optimize_result = bads.optimize()


x_min = optimize_result["x"]
fval = optimize_result["fval"]

print(f"BADS minimum at: x_min = {x_min.flatten()}, fval = {fval:.4g}")
print(
    f"total f-count: {optimize_result['func_count']}, time: {round(optimize_result['total_time'], 2)} s"
)
