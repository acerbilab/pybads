import numpy as np
from pybads.bads.bads import BADS
from pybads.function_examples import quadratic_noisy_fcn, extra_noisy_quadratic_fcn

x0 = np.array([[-3, -3]]);        # Starting point
lb = np.array([[-5, -5]])     # Lower bounds
ub = np.array([[5, 5]])       # Upper bounds
plb = np.array([[-2, -2]])      # Plausible lower bounds
pub = np.array([[2, 2]])        # Plausible upper bounds

title = 'Noise objective function'
print("\n *** Example 3: " + title)
print("\t We test BADS on a noisy quadratic function with unit Gaussian noise.")
bads = BADS(quadratic_noisy_fcn, x0, lb, ub, plb, pub)
x_min, fval = bads.optimize()
print(f"BADS minimum at: \n\n\t x = {x_min.flatten()} \n\t fval= {fval}")
print(f"The true, noiseless minimum is at x = {np.sum(x_min**2)} \n")
print(f"The true global minimum is at x = [0, 0], where fval = 0\n")

extra_noise = True
if extra_noise:
    title = 'Extra Noise objective function'
    print("\n *** Example 4: " + title)
    print("\t We test BADS on a particularly noisy function.")
    bads = BADS(extra_noisy_quadratic_fcn, x0, lb, ub, plb, pub)
    x_min, fval = bads.optimize()
    print(f"BADS minimum at: \n\n\t x = {x_min.flatten()} \n\t fval= {fval}")
    print(f"The true, noiseless minimum is at x = {np.sum(x_min**2)} \n")
    print(f"The true global minimum is at x = [1, 1], where fval = 0\n")