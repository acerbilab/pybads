"""F1: how much the constant-mean bounds, fixed on the initial design, cost the GP.
At each refit of a default run, redo the rebuild on a copy of the GP (same data, same
generator state) with the mean bounds widened to cover the prior centre and the local
targets, and compare the log marginal likelihood (data fit) and the log posterior
reached, and the mean fitted. Also: is the first rebuild a refit?"""
import copy
import warnings

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
import pybads.bads.bads as bmod
from pybads import BADS

rows = []
first = {}
orig = bmod.local_gp_fitting


def w(gp, u, fl, options, optim_state, ih, refit_flag, rng=None):
    if "refit" not in first:
        first["refit"] = bool(refit_flag)
    if not refit_flag:
        return orig(gp, u, fl, options, optim_state, ih, refit_flag, rng=rng)
    gp_c = copy.deepcopy(gp)
    os_c = copy.deepcopy(optim_state)
    rng_c = copy.deepcopy(rng)
    out = orig(gp, u, fl, options, optim_state, ih, refit_flag, rng=rng)
    g = out[0]
    i = g.D + 3
    y = g.y
    h = np.max(y) - np.min(y)
    mu_p = g.hyper_priors["mu"][i]
    b = gp_c.get_bounds()
    b["mean_const"] = (
        np.array(
            [min(np.min(y) - 0.5 * h, mu_p - 5 * g.hyper_priors["sigma"][i])]
        ),
        np.array(
            [max(np.max(y) + 0.5 * h, mu_p + 5 * g.hyper_priors["sigma"][i])]
        ),
    )
    gp_c.set_bounds(b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cf, _ = orig(gp_c, u, fl, options, os_c, ih, refit_flag, rng=rng_c)
    h1 = g.get_hyperparameters(as_array=True)[0]
    h2 = cf.get_hyperparameters(as_array=True)[0]
    outside = (mu_p < g.lower_bounds[i]) or (mu_p > g.upper_bounds[i])
    rows.append(
        (
            len(y),
            outside,
            g.log_likelihood(h1),
            cf.log_likelihood(h2),
            h1[i],
            h2[i],
            mu_p,
            g.lower_bounds[i],
            g.upper_bounds[i],
            float(np.min(y)),
            float(np.max(y)),
        )
    )
    return out


bmod.local_gp_fitting = w


def rosen(z):
    z = np.atleast_1d(z)
    return float(np.sum(100 * (z[1:] - z[:-1] ** 2) ** 2 + (z[:-1] - 1) ** 2))


cases = [
    (
        "sphere D=2",
        lambda x: float(np.sum(np.atleast_1d(x) ** 2)),
        2,
        np.full(2, 0.5),
        -5,
        5,
        -2,
        2,
    ),
    (
        "quadratic at 20 D=2",
        lambda x: float(np.sum((np.atleast_1d(x) - 20) ** 2) / 100),
        2,
        np.zeros(2),
        -50,
        50,
        -1,
        1,
    ),
    (
        "rosen shifted by 4 D=3",
        lambda x: rosen(np.atleast_1d(x) - 4),
        3,
        np.zeros(3),
        -10,
        10,
        -1,
        1,
    ),
]
for name, f, D, x0, lb, ub, plb, pub in cases:
    rows.clear()
    first.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bb = BADS(
            f,
            x0,
            np.full(D, float(lb)),
            np.full(D, float(ub)),
            np.full(D, float(plb)),
            np.full(D, float(pub)),
            options=dict(display="off", random_seed=0, max_fun_evals=200),
        )
        r = bb.optimize()
    print(
        f"\n{name}: fval {r['fval']:.4g} evals {r['func_count']}; first rebuild is a refit: {first['refit']}"
    )
    print(
        "   N  prior-outside  logML(frozen bounds)  logML(widened)   m frozen   m widened   prior centre   mean bounds          local y range"
    )
    for n, o, l1, l2, m1, m2, mp, blo, bhi, ylo, yhi in rows:
        print(
            f"  {n:3d}  {str(o):5s}  {l1:14.4g}  {l2:14.4g}  {m1:10.4g}  {m2:10.4g}  {mp:10.4g}   [{blo:.4g}, {bhi:.4g}]   [{ylo:.3g}, {yhi:.3g}]"
        )
