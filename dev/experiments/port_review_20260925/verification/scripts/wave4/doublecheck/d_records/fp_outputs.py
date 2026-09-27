"""The fingerprint's six runs, counting the calls of contraints_check whose
output differs between the bins of np.round and those of round_half_away."""
import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__, flush=True)
import pybads.bads.bads as bads_module
import pybads.function_logger.constraints_check as cc
import pybads.search.es_search as es_module
from pybads import BADS
from pybads.rounding import round_half_away

stats = {"calls": 0, "output_differs": 0}
original = cc.contraints_check


def wrapped(*args, **kwargs):
    cc.round_half_away = np.round
    old = original(*args, **kwargs)
    cc.round_half_away = round_half_away
    new = original(*args, **kwargs)
    stats["calls"] += 1
    stats["output_differs"] += not (
        old.shape == new.shape and np.array_equal(old, new)
    )
    return new


bads_module.contraints_check = wrapped
es_module.contraints_check = wrapped


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
        before = dict(stats)
        BADS(
            fun,
            np.ones(3) * 4,
            -100 * np.ones(3),
            100 * np.ones(3),
            -8 * np.ones(3),
            12 * np.ones(3),
            options=o,
        ).optimize()
        print(
            noisy,
            seed,
            {k: stats[k] - before[k] for k in stats},
            flush=True,
        )
print(pybads.__file__, stats)
