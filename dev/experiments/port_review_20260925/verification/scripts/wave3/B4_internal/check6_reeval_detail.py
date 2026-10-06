"""Detail of the poll's re-evaluation of an already evaluated point
(rosen D=3, seed 1)."""
import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
import pybads.bads.bads as bm
from pybads import BADS
from pybads.function_logger import FunctionLogger

ctx = {"poll": False, "bads": None}
orig_poll = bm.BADS._poll_step_


def poll(self, gp):
    ctx["poll"] = True
    ctx["bads"] = self
    ctx["u0"] = self.u.copy()
    ctx["prev"] = ctx.get("last_inc")
    try:
        return orig_poll(self, gp)
    finally:
        ctx["poll"] = False
        ctx["last_inc"] = ctx["u0"]


bm.BADS._poll_step_ = poll

orig_call = FunctionLogger.__call__


def call(self, x, record_duplicate_data=True):
    u = np.atleast_2d(x)
    n = self.X_max_idx + 1
    if ctx["poll"] and n > 0:
        m = np.all(np.abs(self.X[:n] - u) < 2.0**-21, axis=1)
        if np.any(m):
            b = ctx["bads"]
            print(
                "re-evaluation in poll, iter",
                b.optim_state["iter"] + 1,
                "mesh",
                b.mesh_size,
                "earlier rows",
                np.nonzero(m)[0],
                "of",
                n,
                "y earlier",
                self.Y[:n][m],
                "u",
                u,
                "incumbent",
                ctx["u0"],
                "is incumbent of an earlier iteration:",
                any(np.allclose(u, h) for h in b.iteration_history.get("u")),
            )
    return orig_call(self, x, record_duplicate_data)


FunctionLogger.__call__ = call


def rosen(x):
    x = np.atleast_2d(x)
    return float(
        np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2)
    )


D = 3
lb = -5 * np.ones((1, D))
ub = 5 * np.ones((1, D))
plb = -2 * np.ones((1, D))
pub = 2 * np.ones((1, D))
r = BADS(
    rosen,
    np.full((1, D), 1.5),
    lb,
    ub,
    plb,
    pub,
    options={"random_seed": 1, "display": "off", "max_fun_evals": 200},
).optimize()
print("fval", r["fval"])
