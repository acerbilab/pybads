"""The fingerprint's six runs, counting the calls of contraints_check whose
bins, or the ES split, differ between np.round and round_half_away."""
import numpy as np

import pybads
import pybads.function_logger.constraints_check as cc
from pybads import BADS
from pybads.rounding import round_half_away

stats = {"calls": 0, "differ": 0, "min_mesh": np.inf}


def counting(q):
    r = round_half_away(q)
    if q.ndim == 2 and q.shape[0] > 0:
        stats["calls"] += 0.5  # two calls per contraints_check
        if not np.array_equal(r, np.round(q)):
            stats["differ"] += 1
    return r


cc.round_half_away = counting


def f(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


g = np.random.default_rng(0)


def fn(x):
    return float(np.sum(np.atleast_2d(x) ** 2) + g.standard_normal())


for fun, noisy in [(f, False), (fn, True)]:
    for seed in range(3):
        o = {"display": "off", "max_fun_evals": 80, "random_seed": seed}
        if noisy:
            o["uncertainty_handling"] = True
        b = BADS(
            fun,
            np.ones(3) * 4,
            -100 * np.ones(3),
            100 * np.ones(3),
            -8 * np.ones(3),
            12 * np.ones(3),
            options=o,
        )
        r = b.optimize()
        print(
            noisy,
            seed,
            "mesh_size",
            b.optim_state["mesh_size"],
            "search_mesh_size",
            b.optim_state["search_mesh_size"],
            flush=True,
        )
print(pybads.__file__, stats)
