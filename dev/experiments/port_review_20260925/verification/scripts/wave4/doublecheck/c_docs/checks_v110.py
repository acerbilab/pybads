"""Spot checks at 1.1.0 of what the extended entries "Checks of
max_fun_evals, improvement_quantile, accelerate_mesh_steps and n_search_iter"
and "Search without a candidate" say 1.1.0 did."""

import traceback

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__, gpyreg.__file__, flush=True)

D = 2


def sphere(x):
    return float(np.sum(np.ravel(x) ** 2))


def run(opts, non_box_cons=None, max_fun_evals=200):
    options = {
        "display": "off",
        "random_seed": 0,
        "max_fun_evals": max_fun_evals,
    }
    options.update(opts)
    try:
        b = BADS(
            sphere,
            np.array([1.0, 1.5]),
            -5 * np.ones(D),
            5 * np.ones(D),
            -3 * np.ones(D),
            3 * np.ones(D),
            non_box_cons=non_box_cons,
            options=options,
        )
        r = b.optimize()
        return (
            f"completed: func_count={r['func_count']} fval={r['fval']:.3g} "
            f"iterations={r['iterations']}"
        )
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        msg = " ".join(" ".join(str(a) for a in e.args).split())[:90]
        return f"{type(e).__name__}: {msg} [{tb.filename.split('/pybads/')[-1]}:{tb.lineno}]"


for name, value in [
    ("max_fun_evals", 30.5),
    ("accelerate_mesh_steps", 0),
    ("accelerate_mesh_steps", -1),
    ("accelerate_mesh_steps", 2.5),
    ("accelerate_mesh_steps", True),
]:
    extra = {} if name == "max_fun_evals" else {}
    mfe = 30.5 if name == "max_fun_evals" else 200
    opts = {name: value}
    if name == "max_fun_evals":
        print(f"{name}={value!r}: {run({}, max_fun_evals=value)}", flush=True)
    else:
        print(f"{name}={value!r}: {run(opts)}", flush=True)


# A constraint that accepts every point at its first two calls (x0's check
# and the design's; FREE calls) and none after, so that the searches leave no candidate
class Flaky:
    def __init__(self):
        self.calls = 0
        self.free = FREE

    def __call__(self, X):
        self.calls += 1
        return np.full(np.atleast_2d(X).shape[0], self.calls > self.free)


for FREE in (3, 4, 5):
    print(
        f"non_box_cons flaky after {FREE} calls: {run({}, non_box_cons=Flaky())}",
        flush=True,
    )
