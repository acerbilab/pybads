"""K6: in the thin-band run at D=2, are the poll candidates generated and all
removed by non_box_cons?"""
import common
import numpy as np

import pybads.bads.bads as bmod
from pybads import BADS
from pybads.function_logger import contraints_check as orig_cc

log = []


def cc(U, lb, ub, tol_mesh, fl, proj=True, nbc=None):
    out = orig_cc(U, lb, ub, tol_mesh, fl, proj, None)
    out2 = orig_cc(U, lb, ub, tol_mesh, fl, proj, nbc)
    X = (
        fl.variable_transformer.inverse_transf(np.atleast_2d(out))
        if len(out)
        else np.zeros((0, 2))
    )
    log.append(
        (
            len(U),
            len(out),
            len(out2),
            float(np.min(np.abs(X[:, 0] - X[:, 1]))) if len(X) else None,
        )
    )
    return out2


bmod.contraints_check = cc
f = lambda x: float(np.sum((np.ravel(x) - 0.3) ** 2))
nbc = lambda X: np.abs(np.atleast_2d(X)[:, 0] - np.atleast_2d(X)[:, 1]) - 0.005
b = BADS(
    f,
    np.full(2, 0.5),
    np.full(2, -5.0),
    np.full(2, 5.0),
    np.full(2, -2.0),
    np.full(2, 2.0),
    nbc,
    options=dict(display="off", random_seed=0, max_fun_evals=200),
)
r = b.optimize()
print(
    "calls (candidates, after bounds/duplicates, after non_box_cons, min |x1-x2| among candidates):"
)
for e in log:
    print("  ", e[0], e[1], e[2], None if e[3] is None else round(e[3], 4))
print(
    r["message"],
    r["func_count"],
    "mesh sizes",
    b.iteration_history.get("mesh_size"),
)
