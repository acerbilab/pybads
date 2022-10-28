# PyBADS Example 3: Noisy objective function
# (code only - see Jupyter notebook version for a full description)

import numpy as np
from pybads.bads.bads import BADS

def noisy_sphere(x,sigma=1.0):
    """Simple quadratic function with added noise."""
    x_2d = np.atleast_2d(x)
    f = np.sum(x_2d**2, axis=1)
    noise = sigma*np.random.normal(size=x_2d.shape[0])
    return f + noise

x0 = np.array([[-3, -3]]);      # Starting point
lb = np.array([[-5, -5]])       # Lower bounds
ub = np.array([[5, 5]])         # Upper bounds
plb = np.array([[-2, -2]])      # Plausible lower bounds
pub = np.array([[2, 2]])        # Plausible upper bounds

options = {"uncertaintyhandling": True, "maxfunevals": 300, "noisefinalsamples": 100}
bads = BADS(noisy_sphere, x0, lb, ub, plb, pub, user_options=options)
x_min, fval = bads.optimize()

print(f"BADS minimum at: x_min = {x_min.flatten()}, fval (estimated) = {fval:.3g} +/- {bads.fsd:.2g}")
print(f"total f-count: {bads.function_logger.func_count}, time: {round(bads.optim_state['total_time'], 2)} s")
print(f"The true, noiseless value of f(x_min) is {noisy_sphere(x_min,sigma=0)[0]:.3g}.")