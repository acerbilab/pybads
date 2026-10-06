import collections

import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__)
print("gpyreg", gpyreg.__file__)
from scipy.stats import shapiro

import pybads.bads.bads as bb
from pybads import BADS

last = {}
orig_acq = bb.acq_fcn_lcb


def acq(xi, fc, gp, *a, **k):
    z, fmu, fs = orig_acq(xi, fc, gp, *a, **k)
    _, ys2 = gp.predict(xi, add_noise=True)
    last["fmu"], last["ys"] = np.ravel(fmu), np.sqrt(np.ravel(ys2))
    return z, fmu, fs


bb.acq_fcn_lcb = acq
pred_sd = []
orig_save = bb.BADS._save_gp_stats_


def save(self, fval, ymu, ys):
    i = int(np.argmin(np.abs(last["fmu"] - ymu)))
    pred_sd.append((fval, ymu, ys, last["ys"][i]))
    return orig_save(self, fval, ymu, ys)


bb.BADS._save_gp_stats_ = save
dec = collections.Counter()
orig_rec = bb.BADS._record_gp_refit_


def rec(self):
    pred_sd.clear()
    return orig_rec(self)


bb.BADS._record_gp_refit_ = rec
orig_irt = bb.BADS._is_gp_refit_time_


def irt(self, alpha):
    out = orig_irt(self, alpha)
    if len(pred_sd) >= 3 or out[0]:
        pass
    n = len(pred_sd)
    if n >= 3 and not out[0]:
        a = np.array(pred_sd)
        z_lat = (a[:, 0] - a[:, 1]) / np.where(
            np.isclose(a[:, 2], 0), 1e-6, a[:, 2]
        )
        z_pred = (a[:, 0] - a[:, 1]) / a[:, 3]
        dec["checks n>=3"] += 1
        dec["flag, latent SD (code)"] += shapiro(z_lat).pvalue < alpha
        dec["flag, predictive SD"] += shapiro(z_pred).pvalue < alpha
        dec["sum ratio latent/pred |z|"] += float(
            np.median(np.abs(z_lat) / np.maximum(np.abs(z_pred), 1e-300))
        )
    return out


bb.BADS._is_gp_refit_time_ = irt


def rosen(x):
    x = np.atleast_2d(x)
    return float(
        np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2)
    )


for name, D, noisy in [("rosen2", 2, False), ("noisy sphere3", 3, True)]:
    for seed in [0, 1]:
        dec.clear()
        pred_sd.clear()
        if noisy:
            r_ = np.random.default_rng(100 + seed)
            f = lambda x: float(np.sum(np.atleast_2d(x) ** 2) + r_.normal())
        else:
            f = rosen
        b = BADS(
            f,
            np.full((1, D), 1.5),
            np.full((1, D), -10.0),
            np.full((1, D), 10.0),
            np.full((1, D), -3.0),
            np.full((1, D), 3.0),
            options={
                "random_seed": seed,
                "display": "off",
                "max_fun_evals": 200,
            },
        )
        b.optimize()
        d = dict(dec)
        c = d.get("checks n>=3", 1)
        d["median |z_lat|/|z_pred| (mean over checks)"] = (
            d.pop("sum ratio latent/pred |z|", 0) / c
        )
        print(name, seed, d)
