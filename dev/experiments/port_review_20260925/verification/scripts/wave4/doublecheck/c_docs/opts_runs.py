"""Short seeded runs with the option values that wave 4's changelog entries
describe (the hedge's parameters, n_search_iter, sqrt_beta, periodic_vars),
at whichever pybads PYTHONPATH selects. Usage: opts_runs.py <group>..."""

import sys
import traceback

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__, gpyreg.__file__, flush=True)

D = 2


def sphere(x):
    return float(np.sum(np.ravel(x) ** 2))


def run(opts, x0=None, max_fun_evals=100, fun=sphere):
    options = {
        "display": "off",
        "random_seed": 0,
        "max_fun_evals": max_fun_evals,
    }
    options.update(opts)
    if x0 is None:
        x0 = np.array([1.0, 1.5])
    stage = "create"
    try:
        b = BADS(
            fun,
            x0,
            -5 * np.ones(D),
            5 * np.ones(D),
            -3 * np.ones(D),
            3 * np.ones(D),
            options=options,
        )
        stage = "optimize"
        r = b.optimize()
        return (
            f"completed: func_count={r['func_count']} fval={r['fval']:.3g} "
            f"iterations={r['iterations']}"
        )
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        msg = " ".join(" ".join(str(a) for a in e.args).split())[:100]
        where = f"{tb.filename.split('/pybads/')[-1]}:{tb.lineno}"
        return f"{stage}: {type(e).__name__}: {msg} [{where}]"


class Counter:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __call__(self, t, d):
        self.calls += 1
        return self.value


groups = {
    "hedge": [
        ("hedge_gamma", 0.75),
        ("hedge_gamma", 1.25),
        ("hedge_gamma", -0.5),
        ("hedge_gamma", "0.1"),
        ("hedge_gamma", True),
        ("hedge_beta", -1.0),
        ("hedge_beta", -1000.0),
        ("hedge_beta", np.inf),
        ("hedge_beta", np.nan),
        ("hedge_beta", "1"),
        ("hedge_decay", -0.5),
        ("hedge_decay", 2.0),
        ("hedge_decay", 1e10),
        ("hedge_decay", np.nan),
    ],
    "nsi": [
        ("n_search_iter", 0),
        ("n_search_iter", -1),
        ("n_search_iter", -2.5),
        ("n_search_iter", 0.5),
        ("n_search_iter", 2.0),
        ("n_search_iter", 2.5),
        ("n_search_iter", np.inf),
        ("n_search_iter", np.nan),
        ("n_search_iter", 1e-300),
        ("n_search_iter", True),
        ("n_search_iter", False),
        ("n_search_iter", "2"),
        ("n_search_iter", np.int64(3)),
    ],
    "sqrt_beta": [
        ("search_acq_fcn", ("acq_LCB", 2.0)),
        ("search_acq_fcn", ("acq_LCB", np.float64(0.0))),
        ("search_acq_fcn", ("acq_LCB", np.float64(-1.0))),
        ("search_acq_fcn", ("acq_LCB", np.array([-1.0]))),
        ("search_acq_fcn", ("acq_LCB", np.bool_(True))),
        ("search_acq_fcn", ("acq_LCB", np.complex128(2.0))),
        ("search_acq_fcn", ("acq_LCB", np.float64(np.nan))),
        ("search_acq_fcn", ("acq_LCB", np.float64(np.inf))),
        ("search_acq_fcn", ("acq_LCB", "2")),
        ("search_acq_fcn", ("acq_LCB", lambda t, d: -1.0)),
        ("search_acq_fcn", ("acq_LCB", lambda t, d: np.nan)),
        ("search_acq_fcn", ("acq_LCB", lambda t, d: np.array([1.0, 2.0]))),
        ("search_acq_fcn", ("acq_LCB", lambda t, d: "2")),
        ("search_acq_fcn", ("acq_LCB", lambda t, d: 2.0)),
    ],
    "periodic": [
        ("periodic_vars", []),
        ("periodic_vars", np.array([])),
        ("periodic_vars", [0]),
        ("periodic_vars", [5]),
    ],
}

for group in sys.argv[1:]:
    print(f"-- {group}", flush=True)
    for name, value in groups[group]:
        label = f"{name}={value!r}"
        print(f"{label[:52]:54s} {run({name: value})}", flush=True)
        if group == "periodic":
            label = f"{name}={value!r}, random x0"
            print(
                f"{label[:52]:54s} {run({name: value}, x0=np.full(D, np.nan))}",
                flush=True,
            )
