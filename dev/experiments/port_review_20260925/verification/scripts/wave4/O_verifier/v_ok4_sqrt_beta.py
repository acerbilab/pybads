"""O-K4: when a bad sqrt_beta is refused, and what a callable's value does.
(1) search_acq_fcn = ('acq_LCB', v) for v in -1.0, 0.0, 'ucb': is BADS
created, and after how many target evaluations does optimize() stop?
(2) a callable sqrt_beta returning -1, nan, an array of two: does the run
stop, warn, or go on?"""
import logging
import warnings

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__, flush=True)
print("gpyreg:", gpyreg.__file__, flush=True)
from pybads import BADS

CNT = {"n": 0}


def f(x):
    CNT["n"] += 1
    x = np.asarray(x)
    return float(np.sum(x**2 * np.array([1.0, 10.0, 0.5])))


def run(sb, label):
    CNT["n"] = 0
    try:
        b = BADS(
            f,
            np.array([2.0, -1.0, 1.5]),
            -5 * np.ones(3),
            5 * np.ones(3),
            -3 * np.ones(3),
            3 * np.ones(3),
            options={
                "display": "off",
                "random_seed": 0,
                "max_fun_evals": 60,
                "search_acq_fcn": ("acq_LCB", sb),
            },
        )
    except Exception as e:
        print(f"{label}: BADS() raised {type(e).__name__}: {e}", flush=True)
        return
    print(
        f"{label}: BADS() created, {CNT['n']} evaluations so far", flush=True
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            r = b.optimize()
            print(
                f"    optimize() finished: {CNT['n']} evaluations, fval {r['fval']:.3g}, "
                f"{len(w)} Python warnings",
                flush=True,
            )
        except Exception as e:
            print(
                f"    optimize() raised {type(e).__name__} after {CNT['n']} evaluations: "
                f"{str(e)[:110]}",
                flush=True,
            )


logging.getLogger("BADS").setLevel(logging.ERROR)
for v in [-1.0, 0.0, "ucb"]:
    run(v, f"sqrt_beta={v!r}")
run(lambda t, n: -1.0, "callable -> -1.0")
run(lambda t, n: float("nan"), "callable -> nan")
run(lambda t, n: np.array([1.0, 2.0]), "callable -> array of 2")
run(lambda t, n: "x", "callable -> 'x'")
