"""How often a refit's gp.fit raises in _robust_gp_fit_ at default options, and where the noise lands."""
import logging

import common  # noqa
import gpyreg as gpr
import numpy as np

from pybads import BADS
from pybads.bads import gaussian_process_train as gpt

logging.getLogger("BADS").setLevel(logging.ERROR)

orig_fit = gpr.GP.fit
orig_rf = gpt._robust_gp_fit_
STATE = {"in_rf": False, "fails": 0}
REFITS = []


def fit(self, *a, **k):
    try:
        return orig_fit(self, *a, **k)
    except np.linalg.LinAlgError:
        if STATE["in_rf"]:
            STATE["fails"] += 1
        raise


def rf(gp, *a, **k):
    STATE["in_rf"] = True
    STATE["fails"] = 0
    try:
        out = orig_rf(gp, *a, **k)
    finally:
        STATE["in_rf"] = False
    g, hyp, res, flag = out
    cN = g.covariance.hyperparameter_count(g.D)
    noise = hyp[0, cN]
    lb0 = g.get_bounds()["noise_log_scale"][0][0]
    k_ = STATE["fails"]
    nudged_lb = lb0 + k_ * (k_ + 1) / 2
    REFITS.append((k_, noise, nudged_lb))
    return out


gpr.GP.fit = fit
gpt._robust_gp_fit_ = rf


def rosen(x):
    x = np.ravel(x)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def ell(x):
    x = np.ravel(x)
    D = x.size
    return float(np.sum((10 ** (np.arange(D) / (D - 1) * 2) * x) ** 2))


def sph_noisy_factory(seed):
    r = np.random.default_rng(1000 + seed)
    return lambda x: float(np.sum(np.ravel(x) ** 2) + r.standard_normal())


runs = (
    [("rosen3", rosen, 3, s) for s in (40, 41)]
    + [("ell4", ell, 4, s) for s in (40, 41)]
    + [("ell2", ell, 2, 40)]
    + [("noisy sphere3", None, 3, s) for s in (40, 41)]
)
tot = []
for name, fun, D, seed in runs:
    REFITS.clear()
    opts = {"random_seed": seed, "display": "off", "max_fun_evals": 200}
    if fun is None:
        fun = sph_noisy_factory(seed)
        opts["uncertainty_handling"] = True
    b = BADS(
        fun,
        0.5 * np.ones((1, D)) + 0.1 * np.arange(D),
        -5 * np.ones((1, D)),
        5 * np.ones((1, D)),
        -2 * np.ones((1, D)),
        2 * np.ones((1, D)),
        options=opts,
    )
    r = b.optimize()
    k = np.array([x[0] for x in REFITS])
    at_b = np.array([abs(x[1] - x[2]) < 1e-9 and x[0] > 0 for x in REFITS])
    print(
        f"{name:14s} seed {seed}: refits {len(k)}, with >=1 failure {int(np.sum(k>0))}, max consecutive {k.max() if k.size else 0}, "
        f"noise at nudged bound {int(at_b.sum())}/{int(np.sum(k>0))}, fval {r['fval']:.3g}"
    )
    tot += REFITS
k = np.array([x[0] for x in tot])
print(
    "total refits",
    len(k),
    "with failure",
    int(np.sum(k > 0)),
    "histogram of failures:",
    np.bincount(k),
)
