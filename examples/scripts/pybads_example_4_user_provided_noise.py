# PyBADS Example 4: Noisy objective with user-provided noise estimates
# (code only - see Jupyter notebook for a tutorial)

import numpy as np

from pybads import BADS


def noisy_sphere_estimated_noise(x, scale=1.0):
    """Quadratic function with heteroskedastic noise; also return noise estimate."""
    x_2d = np.atleast_2d(x)
    f = np.sum(x_2d**2, axis=1)
    sigma = scale * (1.0 + np.sqrt(f))
    y = f + sigma * np.random.normal(size=x_2d.shape[0])
    return y, sigma


x0 = np.array([-3, -3])  # Starting point
lower_bounds = np.array([-5, -5])
upper_bounds = np.array([5, 5])
plausible_lower_bounds = np.array([-2, -2])
plausible_upper_bounds = np.array([2, 2])

options = {
    "uncertainty_handling": True,
    "specify_target_noise": True,
    "noise_final_samples": 100,
}

bads = BADS(
    noisy_sphere_estimated_noise, x0, lower_bounds, upper_bounds, plausible_lower_bounds, plausible_upper_bounds, 
    options=options
)
optimize_result = bads.optimize()

x_min = optimize_result["x"]
fval = optimize_result["fval"]
fsd = optimize_result["fsd"]

x_min = optimize_result["x"]
fval = optimize_result["fval"]
fsd = optimize_result["fsd"]

print(
    f"BADS minimum at: x_min = {x_min.flatten()}, fval (estimated) = {fval:.4g} +/- {fsd:.2g}"
)
print(
    f"total f-count: {optimize_result['func_count']}, time: {round(optimize_result['total_time'], 2)} s"
)
print(f"final evaluations (shape): {optimize_result['yval_vec'].shape}")
print(f"final evaluations SD (shape): {optimize_result['ysd_vec'].shape}")

print(
    f"The true, noiseless value of f(x_min) is {noisy_sphere_estimated_noise(x_min,scale=0)[0][0]:.3g}."
)
