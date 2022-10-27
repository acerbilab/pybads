# PyBADS Example 1: Basic usage
# (code only - see Jupyter notebook version for a full description)

import numpy as np
from pybads.bads.bads import BADS

from pybads.function_examples import rosenbrocks_fcn
target = rosenbrocks_fcn;

lb = np.array([[-20, -20]])     # Lower bounds
ub = np.array([[20, 20]])       # Upper bounds
plb = np.array([[-5, -5]])      # Plausible lower bounds
pub = np.array([[5, 5]])        # Plausible upper bounds
x0 = np.array([[0, 0]]);        # Starting point

bads = BADS(target, x0, lb, ub, plb, pub)
x_min, fval = bads.optimize()

print(f"BADS minimum at: x = {x_min.flatten()}, fval = {fval:.4g}")
print(f"total f-count: {bads.function_logger.func_count}, time: {round(bads.optim_state['total_time'], 2)} s")