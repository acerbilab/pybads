from tkinter import FALSE
import numpy as np
from pybads.bads.bads import BADS
from pybads.function_examples import ackley_fcn, rastrigin

lb = np.array([[-32, -32]])     # Lower bounds
ub = np.array([[32, 32]])       # Upper bounds
np.random.seed(7)
x0 = np.random.uniform(low=lb+1, high=ub)
#plb = lb.copy() if plb is None else lb 
#pub = ub.copy() if pub is None else pub
x0 = np.array([[-27.11626948006674, 17.914802703367336]])
title = 'Basic usage'
print("\n *** Example 1: " + title)
print("\t Simple usage of BADS on Ackley's function in 2D.")

bads = BADS(ackley_fcn, x0, lb, ub)
x_min, fval = bads.optimize()
print(f"BADS minimum at: \n\n\t x = {x_min.flatten()} \n\t fval = {fval}\n\t \
    total time: {round(bads.optim_state['total_time'], 2)} s \n\t overhead: {round(bads.optim_state['overhead'], 2)}")
print(f"The true global minimum is at x = [0, 0], where fval = 0\n\n")