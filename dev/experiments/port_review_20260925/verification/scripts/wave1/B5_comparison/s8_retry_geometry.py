"""local_gp_fitting with a refit whose posterior update fails once: the
retry with the previous hyperparameters succeeds; is the geometry in
temporary_data (len_scale, poll_scale, effective_radius) that of the
hyperparameters the GP holds?"""
import copy

import gpyreg
import gpyreg as gpr
import numpy as np

import pybads
import pybads.bads.gaussian_process_train as gpt
from pybads import BADS

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
cap = {}
orig_lgf = gpt.local_gp_fitting
import pybads.bads.bads as bm


def spy(gp, u, fl, options, optim_state, ih, refit_flag, rng=None):
    if refit_flag and optim_state["iter"] >= 3 and not cap:
        cap["a"] = copy.deepcopy((gp, u, fl, options, optim_state)) + (ih,)
    return orig_lgf(gp, u, fl, options, optim_state, ih, refit_flag, rng=rng)


bm.local_gp_fitting = spy
D = 2
b = BADS(
    lambda x: float(np.sum((np.ravel(x) * [1, 10]) ** 2)),
    np.full((1, D), 1.0),
    np.full((1, D), -5.0),
    np.full((1, D), 5.0),
    np.full((1, D), -2.0),
    np.full((1, D), 2.0),
    options={"display": "off", "random_seed": 0, "max_fun_evals": 80},
)
b.optimize()
bm.local_gp_fitting = orig_lgf
gp, u, fl, options, optim_state, ih = cap["a"]
old_hyp = gp.get_hyperparameters(as_array=True).copy()
old_geo = {
    k: copy.deepcopy(gp.temporary_data[k])
    for k in ("len_scale", "poll_scale", "effective_radius")
}

orig_update = gpr.GP.update
state = {"n": 0}


def update(self, *a, **k):
    import sys

    if (
        sys._getframe(1).f_code.co_name == "local_gp_fitting"
        and "hyp" in k
        and state["n"] == 0
    ):
        state["n"] += 1
        raise np.linalg.LinAlgError("injected")
    return orig_update(self, *a, **k)


gpr.GP.update = update
g, flag = gpt.local_gp_fitting(
    gp, u, fl, options, optim_state, ih, True, rng=np.random.default_rng(0)
)
gpr.GP.update = orig_update
held = g.get_hyperparameters(as_array=True)
print(
    "exit flag",
    flag,
    "; GP holds the previous hyperparameters:",
    np.allclose(held, old_hyp),
)
print(
    "markers:",
    {k: g.temporary_data.get(k) for k in ("needs_rebuild", "needs_refit")},
)
ll = held[0, :D]
print(
    "len_scale from held hyp  :",
    np.exp(ll),
    " temporary_data:",
    g.temporary_data["len_scale"],
    " (before the call:",
    old_geo["len_scale"],
    ")",
)
llc = ll - ll.mean()
print(
    "poll_scale from held hyp (before clipping):",
    np.exp(llc),
    " temporary_data:",
    g.temporary_data["poll_scale"],
    " (before:",
    old_geo["poll_scale"],
    ")",
)
alpha = np.exp(held[0, D + 1])
print(
    "effective_radius from held hyp:",
    np.sqrt(alpha * (np.exp(1 / alpha) - 1)),
    " temporary_data:",
    g.temporary_data["effective_radius"],
    " (before:",
    old_geo["effective_radius"],
    ")",
)
print(
    "lastfitgp was set by the refit decision; the next refit waits min_refit_time"
)
