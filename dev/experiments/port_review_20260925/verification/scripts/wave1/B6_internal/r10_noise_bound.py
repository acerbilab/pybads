"""The noise upper bound (log SD 5, i.e. SD 148) against a target whose noise SD is 500,
with the default noise_size and with noise_size=500."""
import warnings

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
import pybads.bads.bads as bmod
from pybads import BADS

rec = []
orig = bmod.local_gp_fitting


def w(gp, *a, **k):
    out = orig(gp, *a, **k)
    g = out[0]
    cov_N = g.covariance.hyperparameter_count(g.D)
    h = g.get_hyperparameters(as_array=True)[0]
    rec.append(
        (
            bool(a[5]),
            h[cov_N],
            g.hyper_priors["mu"][cov_N],
            g.upper_bounds[cov_N],
            h[g.D],
        )
    )
    return out


bmod.local_gp_fitting = w
D = 2
for ns in (None, 500.0):
    rec.clear()
    rn = np.random.default_rng(3)
    f = lambda x: float(
        1e4 * np.sum(np.atleast_1d(x) ** 2) + 500 * rn.normal()
    )
    opts = dict(
        display="off",
        random_seed=2,
        max_fun_evals=200,
        uncertainty_handling=True,
    )
    if ns is not None:
        opts["noise_size"] = ns
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = BADS(
            f,
            np.full(D, 1.0),
            np.full(D, -5.0),
            np.full(D, 5.0),
            np.full(D, -2.0),
            np.full(D, 2.0),
            options=opts,
        )
        r = b.optimize()
    ref = [x for x in rec if x[0]]
    print(
        f"noise_size={ns}: x {np.round(r['x'], 3)} fval {r['fval']:.4g} fsd {r['fsd']:.4g}; refits {len(ref)}"
    )
    print(
        "   fitted log noise SD at refits:",
        np.round([x[1] for x in ref], 2).tolist(),
    )
    print(
        "   prior centre / upper bound:",
        round(ref[-1][2], 2),
        ref[-1][3],
        "| true log SD:",
        round(np.log(500), 2),
        "| fitted log sf:",
        np.round([x[4] for x in ref], 2).tolist(),
    )
