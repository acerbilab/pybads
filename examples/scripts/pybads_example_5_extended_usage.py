# PyBADS Example 5: Extended usage
# (code only - see Jupyter notebook for a tutorial)

import numpy as np

from pybads import BADS


def camelback6(x):
    """Six-hump camelback function."""
    x_2d = np.atleast_2d(x)
    x1 = x_2d[:, 0]
    x2 = x_2d[:, 1]
    f = (
        (4 - 2.1 * (x1 * x1) + (x1 * x1 * x1 * x1) / 3.0) * (x1 * x1)
        + x1 * x2
        + (-4 + 4 * (x2 * x2)) * (x2 * x2)
    )
    return f


lower_bounds = np.array([-3, -2])
upper_bounds = np.array([3, 2])
plausible_lower_bounds = np.array([-2.9, -1.9])
plausible_upper_bounds = np.array([2.9, 1.9])

options = {
    "display": "off",  # We switch off the printing
    "uncertainty_handling": False,  # Good to specify that this is a deterministic function
}

num_opts = 10
optimize_results = []
x_vec = np.zeros((num_opts,lower_bounds.shape[0]))
fval_vec = np.zeros(num_opts)

for opt_count in range(num_opts):
    print('Running optimization ' + str(opt_count) + '...')
    options['random_seed'] = opt_count
    bads = BADS(
        camelback6, None, lower_bounds, upper_bounds, plausible_lower_bounds, plausible_upper_bounds, options=options
    )
    optimize_results.append(bads.optimize())
    x_vec[opt_count] = optimize_results[opt_count].x
    fval_vec[opt_count] = optimize_results[opt_count].fval

print("Found solutions:")
print(x_vec)

print("Function values at solutions:")
print(fval_vec)

idx_best = np.argmin(fval_vec)
result_best = optimize_results[idx_best]

x_min = result_best["x"]
fval = result_best["fval"]

print(f"BADS minimum at x_min = {x_min.flatten()}")
print(f"Function value at minimum fval = {fval}")

result_best
