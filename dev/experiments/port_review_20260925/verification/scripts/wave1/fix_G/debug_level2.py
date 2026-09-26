import sys

import numpy as np

import pybads.bads.bads as bads_module
from pybads import BADS

level = int(sys.argv[1])
rng = np.random.default_rng(1000)


def fun(x):
    y = np.sum(np.atleast_2d(x) ** 2)
    if level == 0:
        return y
    sd = 2 + np.sqrt(y)
    return y + sd * rng.standard_normal(), sd


predictions, rows = [], []
orig_acq = bads_module.acq_fcn_lcb
orig_save = BADS._save_gp_stats_


def spy_acq(xi, fc, gp, *a, **k):
    out = orig_acq(xi, fc, gp, *a, **k)
    predictions.append((xi, gp, out))
    return out


def spy_save(self, fval, ymu, ys):
    xi, gp, (z, f_mu, fs) = predictions[-1]
    i = np.argmin(z)
    _, f2 = gp.predict(xi[i : i + 1])
    nl = gp.get_hyperparameters()[0]["noise_log_scale"].item()
    m = gp.posteriors[0].sn2_mult or 1
    rows.append((ys, np.sqrt(f2.item()), np.exp(nl), m))
    return orig_save(self, fval, ymu, ys)


bads_module.acq_fcn_lcb = spy_acq
BADS._save_gp_stats_ = spy_save
opts = {"display": "off", "max_fun_evals": 60, "random_seed": 0}
if level == 2:
    opts.update(uncertainty_handling=True, specify_target_noise=True)
D = 3
b = BADS(
    fun,
    np.ones(D) * 4,
    -100 * np.ones(D),
    100 * np.ones(D),
    -8 * np.ones(D),
    12 * np.ones(D),
    options=opts,
)
b.optimize()
r = np.array(rows)
print(
    "n stats",
    len(r),
    "ratio ys/latent pct 10/50/90/max",
    np.percentile(r[:, 0] / r[:, 1], [10, 50, 90, 100]),
)
print(
    "noise sd range", r[:, 2].min(), r[:, 2].max(), "mult", np.unique(r[:, 3])
)
print("n with ys > 1.1 latent", np.sum(r[:, 0] > 1.1 * r[:, 1]))
