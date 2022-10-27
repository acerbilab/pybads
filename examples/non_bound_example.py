import numpy as np
from pybads.bads.bads import BADS
from pybads.function_examples import rosenbrocks_fcn, quadratic_non_bound_constr

x0 = np.array([[0, 0]]);        # Starting point
lb = np.array([[-1, -1]])     # Lower bounds
ub = np.array([[1, 1]])       # Upper bounds

title = 'Non-bound constraints '
print("\n *** Example 2: " + title)
print("\t We force the input to stay in a circle with unit radius. BADS will complain because the plausible bounds are not specified explicitly.")

bads = BADS(rosenbrocks_fcn, x0, lb, ub, None, None, nonboxcons=quadratic_non_bound_constr)
x_min, fval = bads.optimize()
print(f"BADS minimum at: \n\n\t x = {x_min.flatten()} \n\t fval = {fval}\n\t total time: {round(bads.optim_state['total_time'], 2)} s")
print(f"The true global minimum is at x = [0.786, 0.618], where fval = 0.046\n")
