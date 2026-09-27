"""B3-K9: np.argsort (unstable) at es_search.py:190 and :246 vs a stable sort."""
import sys

import numpy as np
import vhdr  # noqa

import pybads.search.es_search as es_mod
from pybads import BADS

stats = {
    190: [0, 0],
    246: [0, 0],
}  # line: [calls, calls where the permutation differs]


class NP:
    def __init__(self, stable):
        self.stable = stable

    def __getattr__(self, k):
        return getattr(np, k)

    def argsort(self, a, *args, **kw):
        line = sys._getframe(1).f_lineno
        d = np.argsort(a, *args, **kw)
        s = np.argsort(a, kind="stable")
        if line in stats:
            stats[line][0] += 1
            stats[line][1] += not np.array_equal(d, s)
        return s if self.stable else d


def run(fun, stable, seed, D=3, evals=100):
    es_mod.np = NP(stable)
    try:
        b = BADS(
            fun,
            np.full(D, 1.5),
            np.full(D, -5.0),
            np.full(D, 5.0),
            np.full(D, -2.0),
            np.full(D, 2.0),
            options={
                "display": "off",
                "random_seed": seed,
                "max_fun_evals": evals,
            },
        )
        r = b.optimize()
    finally:
        es_mod.np = np
    fl = b.function_logger
    return r["fval"], fl.X[: fl.X_max_idx + 1].copy()


funs = {
    "sphere": lambda x: float(np.sum(np.atleast_2d(x) ** 2)),
    "quantized sphere (step 0.5)": lambda x: float(
        np.floor(2 * np.sum(np.atleast_2d(x) ** 2)) / 2
    ),
    "rosenbrock": lambda x: float(
        np.sum(
            100
            * (np.atleast_2d(x)[:, 1:] - np.atleast_2d(x)[:, :-1] ** 2) ** 2
            + (1 - np.atleast_2d(x)[:, :-1]) ** 2
        )
    ),
}
for name, f in funs.items():
    for seed in (0, 1):
        for k in stats:
            stats[k] = [0, 0]
        f1, X1 = run(f, False, seed)
        s_port = {k: tuple(v) for k, v in stats.items()}
        f2, X2 = run(f, True, seed)
        same = X1.shape == X2.shape and np.array_equal(X1, X2)
        print(
            f"{name:28s} seed {seed}: argsort calls (line: calls, permutation differs) {s_port}; "
            f"runs identical with a stable sort: {same} (fval {f1:.4g} vs {f2:.4g})"
        )
