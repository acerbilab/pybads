import numpy as np
from pybads.bads.bads import BADS
from pybads.function_examples import quadratic_noisy_fcn

x0 = np.array([[-3, -3]]);        # Starting point
lb = np.array([[-5, -5]])     # Lower bounds
ub = np.array([[5, 5]])       # Upper bounds
plb = np.array([[-2, -2]])      # Plausible lower bounds
pub = np.array([[2, 2]])        # Plausible upper bounds

print("\n *** Example 3: Noise objective function")
print("\t We test BADS on a noisy quadratic function with unit Gaussian noise.")
bads = BADS(quadratic_noisy_fcn, x0, lb, ub, plb, pub)
x_min, fval = bads.optimize()
print(f"BADS minimum at: \n\n\t x = {x_min.flatten()} \n\t fval= {fval}")
print(f"The true, noiseless minimum is at x = {np.sum(x_min**2, axis=1)} \n")
print(f"The true global minimum is at x = [0, 0], where fval = 0\n")