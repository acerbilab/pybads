import numpy as np
from pybads.bads.bads import BADS
from pybads.function_examples import rosenbrocks

x0 = np.array([[0, 0, 0]]);        # Starting point
lb = np.array([[-20, -20, -20]])     # Lower bounds
ub = np.array([[20, 20, 20]])       # Upper bounds
plb = np.array([[-5, -5, -5]])      # Plausible lower bounds
pub = np.array([[5, 5, 5]])        # Plausible upper bounds

bads = BADS(rosenbrocks, x0, lb, ub, plb, pub)
bads.optimize()

assert np.all(bads.plausible_lower_bounds == np.array([[-1. , -1]])), 'Wrong init. transformed PLB'
assert np.all(bads.plausible_upper_bounds == np.array([[1. , 1]])), 'Wrong init transformed PUB'
assert np.all(bads.lower_bounds == np.array([[-4. , -4]])), 'Wrong init transformed LB'
assert np.all(bads.upper_bounds == np.array([[4. , 4]])), 'Wrong init transformed UB'
assert np.all(bads.x0 == x0)

