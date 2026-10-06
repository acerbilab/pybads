"""Integer-typed bounds and the log transform, at the revision on PYTHONPATH."""
import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__, gpyreg.__file__)
f = lambda x: float(np.sum((np.log10(np.atleast_2d(x)) - 1.0) ** 2))
for kind, cast in [
    ("int", lambda v: np.array(v, dtype=int)),
    ("float", lambda v: np.array(v, dtype=float)),
]:
    b = BADS(
        f,
        cast([5, 5]),
        cast([1, 1]),
        cast([1000, 1000]),
        cast([2, 2]),
        cast([500, 500]),
        options={"display": "off", "random_seed": 0, "max_fun_evals": 100},
    )
    vt = b.var_transf
    print(
        kind,
        "log",
        vt.apply_log_t.ravel(),
        "plb/pub in u:",
        b.optim_state["plb"].ravel(),
        b.optim_state["pub"].ravel(),
        "lb/ub in u:",
        b.optim_state["lb"].ravel(),
        b.optim_state["ub"].ravel(),
    )
    r = b.optimize()
    print(
        kind,
        "x",
        np.round(r["x"], 4),
        "fval",
        r["fval"],
        "evals",
        r["func_count"],
    )
