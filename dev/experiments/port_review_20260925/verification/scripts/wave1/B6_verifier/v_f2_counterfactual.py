"""F2 in a default run: at each refit whose mean-prior constant is 0, redo the
rebuild on copies (same GP, optim_state, generator state) with gpyreg's
normalization constants computed as usual except that a 0 becomes 1. A
constant cannot move the MAP, so any difference is the underflow's."""

import copy
import time
import warnings

import gpyreg as gpr
import numpy as np
from common import bads_mod, rosenbrock

from pybads import BADS

GP = gpr.GP
orig_recompute = GP._GP__recompute_normalization_constants


def recompute_no_zero(self):
    orig_recompute(self)
    self.normalization_constants[self.normalization_constants == 0] = 1.0


def neg_log_post_no_zero(gp, h):
    g = copy.deepcopy(gp)
    GP._GP__recompute_normalization_constants = recompute_no_zero
    try:
        g._GP__recompute_normalization_constants()
        g._prior_cache = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return -g.log_posterior(h)
    finally:
        GP._GP__recompute_normalization_constants = orig_recompute


orig = bads_mod.local_gp_fitting
rows = []


def wrapped(gp, u, fl, options, optim_state, ih, refit_flag, rng=None):
    if not refit_flag:
        return orig(gp, u, fl, options, optim_state, ih, refit_flag, rng=rng)
    c = (copy.deepcopy(gp), copy.deepcopy(optim_state), copy.deepcopy(rng))
    t = time.time()
    out = orig(gp, u, fl, options, optim_state, ih, refit_flag, rng=rng)
    t_port = time.time() - t
    g = out[0]
    i = g.D + 3
    if g.normalization_constants[i] != 0:
        return out
    GP._GP__recompute_normalization_constants = recompute_no_zero
    try:
        t = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cf, _ = orig(c[0], u, fl, options, c[1], ih, refit_flag, rng=c[2])
        t_cf = time.time() - t
    finally:
        GP._GP__recompute_normalization_constants = orig_recompute
    h1 = g.get_hyperparameters(as_array=True)[0]
    h2 = cf.get_hyperparameters(as_array=True)[0]
    Xs = g.X[:5]
    m1, _ = g.predict(Xs)
    m2, _ = cf.predict(Xs)
    rows.append(
        (
            len(rows),
            t_port,
            t_cf,
            neg_log_post_no_zero(g, h1),
            neg_log_post_no_zero(cf, h2),
            np.max(np.abs(h1 - h2)),
            float(np.max(np.abs(m1 - m2))),
        )
    )
    return out


bads_mod.local_gp_fitting = wrapped
D = 3
x0 = np.random.default_rng(100).uniform(-1, 1, D)
t = time.time()
res = BADS(
    lambda x: rosenbrock(np.atleast_1d(x) - 4),
    x0,
    np.full(D, -10.0),
    np.full(D, 10.0),
    np.full(D, -1.0),
    np.full(D, 1.0),
    options=dict(display="off", random_seed=0, max_fun_evals=200),
).optimize()
print(
    f"run: fval={res['fval']:.3g} n={res['func_count']} "
    f"({time.time()-t:.1f}s including the counterfactuals)"
)
print(
    "refit | time port | time C:0->1 | -logpost port | -logpost C:0->1 |"
    " max|dhyp| | max|dmean| at 5 training pts"
)
for r in rows:
    print(
        f"{r[0]:5d} | {r[1]:8.2f}s | {r[2]:10.2f}s | {r[3]:13.2f} | "
        f"{r[4]:15.2f} | {r[5]:9.3f} | {r[6]:.3g}"
    )
