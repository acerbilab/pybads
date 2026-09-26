"""C-F1: tol_noise = eps*tol_fun (PyBADS) vs sqrt(eps)*TolFun (MATLAB, bads.m:195).
A deterministic target whose value depends on the order of a summation."""
import common
import numpy as np

from pybads import BADS
from pybads.bads.options import Options

prng = np.random.default_rng(7)
w = np.random.default_rng(3).uniform(0.1, 3.0, 40)


def f(x):
    x = np.ravel(x)
    terms = (
        np.concatenate([w * x[0] ** 2, w * (x[1] - 0.3) ** 2]) / 40
        + np.pi * 1e-3
    )
    return float(np.sum(prng.permutation(terms)))  # same value up to rounding


x0 = np.array([0.5, 0.5])
vals = [f(x0) for _ in range(6)]
print("repeats at x0:", [repr(v) for v in vals])
print("max |diff| %.3g" % (max(vals) - min(vals)))
b = BADS(
    f,
    x0,
    np.full(2, -5.0),
    np.full(2, 5.0),
    np.full(2, -2.0),
    np.full(2, 2.0),
    options=dict(display="off", random_seed=0, max_fun_evals=100),
)
print(
    "PyBADS tol_noise %.3g ; MATLAB's would be %.3g"
    % (
        b.options["tol_noise"],
        np.sqrt(np.finfo(float).eps) * b.options["tol_fun"],
    )
)
r = b.optimize()
print(
    "target_type:",
    r["target_type"],
    "| level",
    b.optim_state["uncertainty_handling_level"],
    "| design points",
    b.optim_state["eff_starting_points"] - 1,
    "| tol_stall_iters",
    b.options["tol_stall_iters"],
    "| n_train_min",
    b.options["n_train_min"],
    "| func_count",
    r["func_count"],
    "| fval %.6g fsd %.3g" % (r["fval"], r["fsd"]),
    "| x",
    np.round(r["x"], 4),
)
