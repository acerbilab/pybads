import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
import pybads.bads.bads as bm
from pybads import BADS

src = {"search": 0, "poll": 0, "search_dup": 0, "poll_dup": 0}
stage = {"s": None}
orig_s, orig_p = bm.BADS._search_step_, bm.BADS._poll_step_


def s(self, gp):
    stage["s"] = "search"
    out = orig_s(self, gp)
    stage["s"] = None
    return out


def p(self, gp):
    stage["s"] = "poll"
    out = orig_p(self, gp)
    stage["s"] = None
    return out


bm.BADS._search_step_ = s
bm.BADS._poll_step_ = p
import pybads.function_logger.function_logger as flm

orig_call = flm.FunctionLogger.__call__


def call(self, x, *a, **k):
    st = stage["s"]
    if st is not None:
        xx = np.atleast_2d(x).reshape(1, -1)
        prev = self.X[: self.Xn + 1]
        dup = bool(len(prev)) and bool(
            np.any(np.all(np.abs(prev - xx) <= 1e-12, axis=1))
        )
        src[st] += 1
        src[st + "_dup"] += int(dup)
    return orig_call(self, x, *a, **k)


flm.FunctionLogger.__call__ = call
for D, name in [(2, "corner2"), (1, "bound1")]:
    for k in src:
        src[k] = 0
    fun = lambda x: float(np.sum((np.asarray(x).ravel() + 1.0) ** 2))
    b = BADS(
        fun,
        np.full(D, 2.0),
        np.zeros(D),
        np.full(D, 5.0),
        np.full(D, 0.5),
        np.full(D, 4.0),
        options={"random_seed": 1, "max_fun_evals": 150, "display": "off"},
    )
    r = b.optimize()
    print(name, "x=", np.round(r["x"], 6), "evals", r["func_count"], src)
