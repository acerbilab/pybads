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
r = np.random.default_rng(0)


# A deterministic function evaluated with a nondeterministic summation order:
# results differ by at most a few ulps between calls.
def fun(x):
    terms = np.concatenate([x**2, [0.1, 0.2, 0.3]])
    r.shuffle(terms)
    s = 0.0
    for t in terms:
        s += t
    return s


vals = [fun(x0) for _ in range(20)]
print(
    "distinct values at x0:",
    sorted(set(vals)),
    "max diff",
    max(vals) - min(vals),
)
b = BADS(
    fun,
    x0,
    lb,
    ub,
    plb,
    pub,
    options=dict(display="off", random_seed=0, max_fun_evals=60),
)
print(
    "tol_noise PyBADS =",
    b.options["tol_noise"],
    " MATLAB default sqrt(eps)*TolFun =",
    np.sqrt(np.spacing(1.0)) * 1e-3,
)
res = b.optimize()
print(
    "target_type:",
    res["target_type"],
    "func_count:",
    res["func_count"],
    "eff_starting_points:",
    b.optim_state["eff_starting_points"],
)
