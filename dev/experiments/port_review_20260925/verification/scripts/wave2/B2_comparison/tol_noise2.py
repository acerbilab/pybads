import logging

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
print(gpyreg.__file__)
logging.getLogger("BADS").setLevel(logging.WARNING)
D = 2
x0 = np.full(D, 0.5)
lb = np.full(D, -5.0)
ub = np.full(D, 5.0)
plb = np.full(D, -2.0)
pub = np.full(D, 2.0)
calls = [0]


def fun(x):
    # deterministic value, with a last-bit wobble on alternate calls (as from
    # a summation whose order varies between calls)
    calls[0] += 1
    f = float(np.sum(x**2)) + 0.1
    return f + (calls[0] % 2) * np.spacing(f)


b = BADS(
    fun,
    x0,
    lb,
    ub,
    plb,
    pub,
    options=dict(display="off", random_seed=0, max_fun_evals=100),
)
res = b.optimize()
d = abs(fun(x0) - fun(x0))
print(
    "difference between two calls at x0:",
    d,
    "> PyBADS tol_noise",
    b.options["tol_noise"],
    ":",
    d > b.options["tol_noise"],
    "; > MATLAB tol_noise",
    np.sqrt(np.spacing(1.0)) * 1e-3,
    ":",
    d > np.sqrt(np.spacing(1.0)) * 1e-3,
)
print(
    "target_type:",
    res["target_type"],
    "func_count:",
    res["func_count"],
    "eff_starting_points:",
    b.optim_state["eff_starting_points"],
    "tol_stall_iters:",
    b.options["tol_stall_iters"],
    "n_train_min:",
    b.options["n_train_min"],
    "fsd:",
    res["fsd"],
    "fval:",
    res["fval"],
)
