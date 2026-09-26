"""get_grid_search_neighbors against a transcription of gpupdate.m 'nearest'
(lines 85-111) on the same data, at every rebuild of default runs."""
import gpyreg
import numpy as np
import s4_runs as R

import pybads
import pybads.bads.gaussian_process_train as gpt
from pybads.bads.bads import BADS

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)

orig = gpt.get_grid_search_neighbors
stats = {
    "calls": 0,
    "set_diff": 0,
    "order_diff": 0,
    "n_diff": 0,
    "ties_at_cut": 0,
}


def matlab_nearest(U, Y, uc, lenscale, eff_radius, opts, Xmax):
    d = np.sum(((U - uc) / lenscale) ** 2, axis=1)  # udist.m
    ord_ = np.argsort(d, kind="stable")  # MATLAB sort is stable
    distord = d[ord_]
    radius = opts["gp_radius"] * eff_radius
    ntrain = min(opts["n_train_max"], int(np.sum(distord <= radius**2)))
    ntrain = max(
        opts["n_train_min"],
        opts["n_train_max"] - opts["buffer_ntrain"],
        ntrain,
    )
    ntrain = min(ntrain, Xmax)
    return ord_[:ntrain], d, distord, ntrain


def spy(function_logger, u, gp, options, optim_state):
    Xp, Yp, Sp = orig(function_logger, u, gp, options, optim_state)
    n = function_logger.X_max_idx + 1
    U = function_logger.X[:n]
    Y = function_logger.Y[:n]
    idx, d, distord, ntrain = matlab_nearest(
        U,
        Y,
        np.atleast_2d(u),
        gp.temporary_data["len_scale"],
        np.ravel(gp.temporary_data["effective_radius"])[0],
        options,
        n,
    )
    stats["calls"] += 1
    Um = U[idx]
    if Um.shape[0] != Xp.shape[0]:
        stats["n_diff"] += 1
    else:
        key = lambda A: set(map(tuple, np.round(A, 14)))
        if key(Um) != key(Xp):
            stats["set_diff"] += 1
        elif not np.array_equal(Um, Xp):
            stats["order_diff"] += 1
        if ntrain < n and np.isclose(
            distord[ntrain - 1], distord[ntrain], rtol=1e-12, atol=0
        ):
            stats["ties_at_cut"] += 1
    return Xp, Yp, Sp


gpt.get_grid_search_neighbors = spy
D = 3
for seed in range(2):
    b = BADS(
        R.rosen,
        np.full((1, D), -1.5),
        np.full((1, D), -5.0),
        np.full((1, D), 5.0),
        np.full((1, D), -3.0),
        np.full((1, D), 3.0),
        options={"display": "off", "random_seed": seed, "max_fun_evals": 200},
    )
    b.optimize()
    print("rosen3 seed", seed, stats)
    for k in stats:
        stats[k] = 0
b = BADS(
    R.NoisySphere(7),
    np.full((1, D), 2.0),
    np.full((1, D), -5.0),
    np.full((1, D), 5.0),
    np.full((1, D), -3.0),
    np.full((1, D), 3.0),
    options={
        "display": "off",
        "random_seed": 0,
        "max_fun_evals": 200,
        "uncertainty_handling": True,
    },
)
b.optimize()
print("noisy sphere3", stats)
