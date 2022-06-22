from tkinter import FALSE
import numpy as np
from pybads.bads.bads import BADS
from pybads.function_examples import rastrigin

x0 = None # np.array([[0, 0]]);        # Starting point
lb = np.array([[-5.12, -5.12]])     # Lower bounds
ub = np.array([[5.12, 5.12]])       # Upper bounds
#plb = lb.copy() if plb is None else lb 
#pub = ub.copy() if pub is None else pub

title = 'Basic usage'
print("\n *** Example 1: " + title)
print("\t Simple usage of BADS on Ackley's function in 2D.")

bads = BADS(rastrigin, x0, lb, ub)
x_min, fval = bads.optimize()
print(f"BADS minimum at: \n\n\t x = {x_min.flatten()} \n\t fval = {fval}\n\t \
    total time: {round(bads.optim_state['total_time'], 2)} s \n\t overhead: {round(bads.optim_state['overhead'], 2)}")
print(f"The true global minimum is at x = [1, 1], where fval = 0\n\n")