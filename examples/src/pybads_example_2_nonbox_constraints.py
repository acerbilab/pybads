# PyBADS Example 2: Non-box constraints
# (code only - see Jupyter notebook version for a full description)

import numpy as np
from pybads.bads.bads import BADS

def rosenbrocks_fcn(x):
    """Rosenbrock's 'banana' function in any dimension."""
    x_2d = np.atleast_2d(x)
    return np.sum(100 * (x_2d[:, 0:-1]**2 - x_2d[:, 1:])**2 + (x_2d[:, 0:-1]-1)**2, axis=1)

x0 = np.array([[0, 0]]);      # Starting point
lb = np.array([[-1, -1]])     # Lower bounds
ub = np.array([[1, 1]])       # Upper bounds

def circle_constr(x):
    """Return constraints violation outside the unit circle."""
    x_2d = np.atleast_2d(x)
    # Note that nonboxcons assumes the function takes a 2D input 
    return np.sum(x_2d**2, axis=1) > 1

bads = BADS(rosenbrocks_fcn, x0, lb, ub, None, None, nonbondcons=circle_constr)
x_min, fval = bads.optimize()

print(f"BADS minimum at: x_min = {x_min.flatten()}, fval = {fval:.4g}")
print(f"total f-count: {bads.function_logger.func_count-1}, time: {round(bads.optim_state['total_time'], 2)} s")