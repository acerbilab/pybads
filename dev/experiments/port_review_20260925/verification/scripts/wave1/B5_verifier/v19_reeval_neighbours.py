"""B5-R3 part 2: neighbours chosen in _re_evaluate_history_ with each stored GP's geometry vs the current GP's (MATLAB)."""
import copy
import logging

import common  # noqa
import numpy as np

from pybads import BADS
from pybads.bads import gaussian_process_train as gpt

logging.getLogger("BADS").setLevel(logging.ERROR)
r_ = np.random.default_rng(90)


def fun(x):
    x = np.ravel(x)
    return float(
        np.sum(x**2 * np.array([1.0, 10.0, 50.0]))
        + 0.3 * r_.standard_normal()
    )


D = 3
b = BADS(
    fun,
    0.5 * np.ones((1, D)),
    -5 * np.ones((1, D)),
    5 * np.ones((1, D)),
    -2 * np.ones((1, D)),
    2 * np.ones((1, D)),
    options={
        "random_seed": 90,
        "display": "off",
        "max_fun_evals": 150,
        "uncertainty_handling": True,
    },
)
b.optimize()
gps = b.iteration_history["gp"]
us = b.iteration_history["u"]
cur = gps[b.optim_state["iter"]]
os_ = copy.deepcopy(b.optim_state)
n_diff = 0
n = 0
sizes = []
for i in range(len(gps)):
    if gps[i] is None:
        continue
    g_own = gps[i]
    g_cur = copy.copy(g_own)
    g_cur.temporary_data = dict(g_own.temporary_data)
    g_cur.temporary_data["len_scale"] = cur.temporary_data["len_scale"]
    g_cur.temporary_data["effective_radius"] = cur.temporary_data[
        "effective_radius"
    ]
    X1, _, _ = gpt.get_grid_search_neighbors(
        b.function_logger, us[i], g_own, b.options, os_
    )
    X2, _, _ = gpt.get_grid_search_neighbors(
        b.function_logger, us[i], g_cur, b.options, os_
    )
    s1 = {tuple(np.round(x, 12)) for x in X1}
    s2 = {tuple(np.round(x, 12)) for x in X2}
    n += 1
    n_diff += s1 != s2
    sizes.append(len(s1 ^ s2))
print(
    f"stored GPs {n}; neighbour sets that differ {n_diff}; symmetric-difference sizes {sizes}"
)
print(
    "len_scale of first stored GP",
    np.round(gps[0].temporary_data["len_scale"], 3),
    " current",
    np.round(cur.temporary_data["len_scale"], 3),
)
