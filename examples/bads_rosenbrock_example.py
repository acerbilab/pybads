import numpy as np
from pybads.bads.bads import BADS
from pybads.function_examples import rosenbrocks

x0 = np.array([[0, 0]]);        # Starting point
lb = np.array([[-20, -20]])     # Lower bounds
ub = np.array([[20, 20]])       # Upper bounds
plb = np.array([[-5, -5]])      # Plausible lower bounds
pub = np.array([[5, 5]])        # Plausible upper bounds

bads = BADS(rosenbrocks, x0, lb, ub, plb, pub)
x_min, fval = bads.optimize()
print(f"BADS minimum at x = {x_min.flatten()} with value fval= {fval}")
