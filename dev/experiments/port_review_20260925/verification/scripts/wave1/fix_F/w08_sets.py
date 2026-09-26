"""On the fingerprint's six runs (with the stable sort), compare at every
call of get_grid_search_neighbors the stable and the default argsort: the
set of rows selected, their order, and ties across the cut-off."""
import numpy as np

import pybads
import pybads.bads.gaussian_process_train as gpt
from pybads import BADS
from pybads.search.grid_functions import udist

print(pybads.__file__)
orig = gpt.get_grid_search_neighbors
st = dict(calls=0, order=0, set=0, cut_ties=0, ties=0)


def spy(function_logger, u, gp, options, optim_state):
    out = orig(function_logger, u, gp, options, optim_state)
    n = function_logger.X_max_idx + 1
    U = function_logger.X[:n]
    d = udist(
        U,
        u,
        gp.temporary_data["len_scale"],
        optim_state["lb"],
        optim_state["ub"],
        optim_state["scale"],
        optim_state["periodic_vars"],
    )
    if d.ndim > 1:
        d = np.min(d, axis=1)
    k = optim_state["ntrain"]
    a = np.argsort(d, kind="stable")[:k]
    b = np.argsort(d)[:k]
    st["calls"] += 1
    st["order"] += not np.array_equal(a, b)
    st["set"] += set(a) != set(b)
    st["ties"] += len(np.unique(d)) < len(d)
    if k < n:
        ds = np.sort(d)
        st["cut_ties"] += ds[k - 1] == ds[k]
    return out


gpt.get_grid_search_neighbors = spy


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
        BADS(
            fun,
            np.ones(3) * 4,
            -100 * np.ones(3),
            100 * np.ones(3),
            -8 * np.ones(3),
            12 * np.ones(3),
            options=o,
        ).optimize()
        print(noisy, seed, st)
