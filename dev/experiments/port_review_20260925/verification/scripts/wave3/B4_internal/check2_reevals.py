"""Count evaluations, in default deterministic runs, at points already in the
function log (within tol_mesh/2), by stage (poll or search)."""
import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
import pybads.bads.bads as bm
from pybads import BADS
from pybads.function_logger import FunctionLogger

stage = {"s": "init"}
orig_poll = bm.BADS._poll_step_
orig_search = bm.BADS._search_step_


def poll(self, gp):
    stage["s"] = "poll"
    try:
        return orig_poll(self, gp)
    finally:
        stage["s"] = "other"


def search(self, gp):
    stage["s"] = "search"
    try:
        return orig_search(self, gp)
    finally:
        stage["s"] = "other"


bm.BADS._poll_step_ = poll
bm.BADS._search_step_ = search

counts = {}
orig_call = FunctionLogger.__call__


def call(self, x, record_duplicate_data=True):
    u = np.atleast_2d(x)
    n = self.X_max_idx + 1
    tol = 2.0**-21  # about tol_mesh/2 for the default tol_mesh
    dup = n > 0 and np.any(np.all(np.abs(self.X[:n] - u) < tol, axis=1))
    key = (stage["s"], bool(dup))
    counts[key] = counts.get(key, 0) + 1
    return orig_call(self, x, record_duplicate_data)


FunctionLogger.__call__ = call


def rosen(x):
    x = np.atleast_2d(x)
    return float(
        np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2)
    )


def ellip(x):
    x = np.ravel(x)
    D = x.size
    return float(np.sum((10 ** (3 * np.arange(D) / (D - 1)) * x) ** 2))


def sphere(x):
    return float(np.sum(np.ravel(x) ** 2))


for name, f, D in [
    ("rosen", rosen, 3),
    ("ellip", ellip, 4),
    ("sphere", sphere, 2),
    ("rosen2", rosen, 2),
]:
    for seed in [1, 2]:
        counts.clear()
        lb = -5 * np.ones((1, D))
        ub = 5 * np.ones((1, D))
        plb = -2 * np.ones((1, D))
        pub = 2 * np.ones((1, D))
        x0 = np.full((1, D), 1.5)
        b = BADS(
            f,
            x0,
            lb,
            ub,
            plb,
            pub,
            options={
                "random_seed": seed,
                "display": "off",
                "max_fun_evals": 200,
            },
        )
        r = b.optimize()
        print(
            name,
            seed,
            dict(sorted(counts.items())),
            "fval %.3g" % r["fval"],
            "evals",
            r["func_count"],
        )
