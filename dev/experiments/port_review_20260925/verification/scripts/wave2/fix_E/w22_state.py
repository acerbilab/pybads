"""W2-2: the search bounds and the GP geometry of a half-bounded D = 3 run."""
import logging

import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
logging.getLogger("BADS").setLevel(logging.ERROR)
inf = np.inf
LB = np.array([0.0, -inf, 1e-3])
UB = np.array([inf, 5.0, inf])
PLB = np.array([0.5, -5.0, 1e-2])
PUB = np.array([5.0, 3.0, 10.0])
f = lambda x: float(
    (x[0] - 1) ** 2 + (x[1] + 2) ** 2 + (np.log10(x[2]) + 1) ** 2
)
b = BADS(
    lambda x: f(np.ravel(x)),
    np.array([2.0, 0.0, 1.0]),
    LB,
    UB,
    PLB,
    PUB,
    options={"display": "off", "random_seed": 0, "max_fun_evals": 200},
)
s = b.optim_state
print("log", b.var_transf.apply_log_t, "u lb", s["lb"], "u ub", s["ub"])
print("lb_search", s["lb_search"], "ub_search", s["ub_search"])
r = b.optimize()
gp = b.iteration_history["gp"][-1] if "gp" in b.iteration_history else None
if gp is not None:
    td = gp.temporary_data
    print(
        "poll_scale",
        td.get("poll_scale"),
        "len_scale",
        td.get("len_scale"),
        "effective_radius",
        td.get("effective_radius"),
    )
print(
    "lb_search",
    b.optim_state["lb_search"],
    "ub_search",
    b.optim_state["ub_search"],
)
print("x", r["x"], "fval", r["fval"], "evals", r["func_count"])
