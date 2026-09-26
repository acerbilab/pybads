"""F4 (internal) / F3 (comparison): what an inflated posterior is. Default
Rosenbrock D=2 seed 1 run (as in v_reach.py); at the first three GP states
handed on by local_gp_fitting with sn2_mult >= 100, compare the fitted noise
with the effective one, try the uninflated factorization, and look at the
residuals of the posterior mean at the training points."""

import copy

import numpy as np
from common import bads_mod, rosenbrock

from pybads import BADS

orig = bads_mod.local_gp_fitting
out_states = []


def w(*a, **k):
    out = orig(*a, **k)
    g = out[0]
    m = max((p.sn2_mult or 1) for p in g.posteriors)
    if m >= 100 and len(out_states) < 3:
        out_states.append((copy.deepcopy(g), m))
    return out


bads_mod.local_gp_fitting = w
x0 = np.random.default_rng(101).uniform(-2, 2, 2)
try:
    BADS(
        rosenbrock,
        x0,
        np.full(2, -5.0),
        np.full(2, 5.0),
        np.full(2, -2.0),
        np.full(2, 2.0),
        options=dict(display="off", random_seed=1, max_fun_evals=200),
    ).optimize()
finally:
    bads_mod.local_gp_fitting = orig

for g, m in out_states:
    h = g.get_hyperparameters()[0]
    sn = float(np.exp(np.ravel(h["noise_log_scale"])[0]))
    sf = float(np.exp(np.ravel(h["covariance_log_outputscale"])[0]))
    hyp = g.get_hyperparameters(as_array=True)[0]
    K = g.covariance.compute(hyp[: g.D + 2], g.X)
    try:
        np.linalg.cholesky(K + sn**2 * np.eye(len(K)))
        pd = "factorizes"
    except np.linalg.LinAlgError:
        pd = "fails"
    mu, _ = g.predict(g.X)
    res = np.abs(mu - g.y).ravel()
    order = np.argsort(g.y.ravel())[:10]
    print(
        f"sn2_mult={m:g}: N={len(g.y)} fitted sn={sn:.3g}, effective "
        f"sn={sn*np.sqrt(m):.3g}; sf={sf:.3g}, std(y)={np.std(g.y):.3g}; "
        f"uninflated K+sn2 I {pd}; |mean-y| at the 10 best points: max "
        f"{res[order].max():.3g} (their range {np.ptp(g.y.ravel()[order]):.3g})"
    )
