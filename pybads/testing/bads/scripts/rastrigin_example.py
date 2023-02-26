import numpy as np
from pybads import BADS
from pybads.function_examples import rastrigin

lb = np.array([[-20, -20]])     # Lower bounds
ub = np.array([[20, 20]])       # Upper bounds
plb = np.ones((1, 2)) * -5.12
pub = np.ones((1, 2)) * 5.12
x0 = np.random.uniform(low=lb+1, high=ub)

print("\n *** Rastrigin Example")
print("\t Simple usage of BADS on Ackley's function in 2D.")

bads = BADS(rastrigin, x0, lb, ub)
optimize_result = bads.optimize()
x_min = optimize_result['x']
fval = optimize_result['fval']
print(f"BADS minimum at: \n\n\t x = {x_min.flatten()} \n\t fval = {fval}\n\t \
    total time: {round(bads.optim_state['total_time'], 2)} s \n\t overhead: {round(bads.optim_state['overhead'], 2)}")
print(f"The true global minimum is at x = [0, 0], where fval = 0\n\n")