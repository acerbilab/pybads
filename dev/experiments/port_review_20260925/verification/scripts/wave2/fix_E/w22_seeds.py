"""W2-2 check (not a gate): half-bounded problems of D = 3, 5 seeds, 200
evaluations at most. Variable 0 bounded below only (linear), variable 1
bounded above only (linear), variable 2 bounded below only (log)."""
import logging
import warnings

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__, gpyreg.__file__)
logging.getLogger("BADS").setLevel(logging.ERROR)
warnings.simplefilter("always")
import collections

WARN = collections.Counter()
_show = warnings.showwarning


def _record(message, category, filename, lineno, file=None, line=None):
    WARN[
        (category.__name__, str(message)[:60], filename.split("/")[-1], lineno)
    ] += 1


warnings.showwarning = _record

inf = np.inf
LB = np.array([0.0, -inf, 1e-3])
UB = np.array([inf, 5.0, inf])
PLB = np.array([0.5, -5.0, 1e-2])
PUB = np.array([5.0, 3.0, 10.0])


def interior(x):
    x = np.ravel(x)
    return float((x[0] - 1) ** 2 + (x[1] + 2) ** 2 + (np.log10(x[2]) + 1) ** 2)


def on_bound(x):
    # minimum at x0 = lb = 0, x1 = ub = 5, x2 = 0.1
    x = np.ravel(x)
    return float((x[0] + 1) ** 2 + (x[1] - 6) ** 2 + (np.log10(x[2]) + 1) ** 2)


def far(x):
    # minimum outside the plausible box, on the unbounded sides
    x = np.ravel(x)
    return float(
        (x[0] - 40) ** 2 + (x[1] + 30) ** 2 + (np.log10(x[2]) - 3) ** 2
    )


cases = [
    ("interior", interior, np.array([2.0, 0.0, 1.0]), {}, [1, -2, 0.1]),
    ("interior x0=None", interior, None, {}, [1, -2, 0.1]),
    (
        "on the finite bounds",
        on_bound,
        np.array([2.0, 0.0, 1.0]),
        {},
        [0, 5, 0.1],
    ),
    (
        "beyond the plausible box",
        far,
        np.array([2.0, 0.0, 1.0]),
        {},
        [40, -30, 1e3],
    ),
    (
        "interior, noisy",
        None,
        np.array([2.0, 0.0, 1.0]),
        {"uncertainty_handling": True},
        [1, -2, 0.1],
    ),
]
for name, fun, x0, opt, xmin in cases:
    for seed in range(5):
        if fun is None:
            g = np.random.default_rng(100 + seed)
            f = lambda x, g=g: interior(x) + 0.1 * g.standard_normal()
        else:
            f = fun
        calls = []

        def wrapped(x, f=f):
            x = np.ravel(x)
            calls.append(x.copy())
            return f(x)

        try:
            b = BADS(
                wrapped,
                x0,
                LB,
                UB,
                PLB,
                PUB,
                options={
                    "display": "off",
                    "random_seed": seed,
                    "max_fun_evals": 200,
                    **opt,
                },
            )
            r = b.optimize()
            X = np.array(calls)
            inside = np.all((X >= LB) & (X <= UB)) and np.all(np.isfinite(X))
            err = np.abs(np.ravel(r["x"]) - xmin)
            if name.startswith("beyond"):
                err[2] = abs(np.log10(r["x"][2]) - 3)
            print(
                f"{name:26s} seed {seed}: x {np.array2string(np.ravel(r['x']), precision=4)} "
                f"fval {r['fval']:.3g} evals {r['func_count']} iters {r['iterations']} "
                f"max|err| {err.max():.2g} all evals within bounds {inside} | {r['message'][:45]}"
            )
        except Exception as e:
            import traceback

            print(f"{name:26s} seed {seed}: RAISED {type(e).__name__}: {e}")
            traceback.print_exc()

print("warnings (category, message, file, line): count")
for k, v in sorted(WARN.items()):
    print(" ", k, v)
