"""F9: the GP calibration statistics compare noisy observations with the latent SD.
Count the calibration failures and refits of level-1 runs as they are, and with the SD
recorded as that of an observation, sqrt(fs^2 + sn^2) (counterfactual)."""
import warnings

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
import pybads.bads.bads as bmod
from pybads import BADS

LAST = {}
orig_lcb = bmod.acq_fcn_lcb


def lcb(xi, fc, gp, *a, **k):
    LAST["gp"] = gp
    return orig_lcb(xi, fc, gp, *a, **k)


bmod.acq_fcn_lcb = lcb
orig_save = BADS._save_gp_stats_
MODE = {"obs": False}


def save(self, fval, ymu, ys):
    if MODE["obs"]:
        g = LAST["gp"]
        cov_N = g.covariance.hyperparameter_count(g.D)
        sn2 = np.exp(2 * g.get_hyperparameters(as_array=True)[0][cov_N])
        ys = float(np.sqrt(ys**2 + sn2))
    return orig_save(self, fval, ymu, ys)


BADS._save_gp_stats_ = save
orig_rt = BADS._is_gp_refit_time_
CNT = {}


def rt(self, alpha):
    out = orig_rt(self, alpha)
    CNT["checks"] += 1
    CNT["refit"] += int(out[0])
    CNT["calib"] += int(out[1] or out[0])
    return out


BADS._is_gp_refit_time_ = rt
for D in (2, 3):
    for obs in (False, True):
        MODE["obs"] = obs
        CNT.update(checks=0, refit=0, calib=0)
        errs = []
        for seed in range(3):
            rn = np.random.default_rng(100 + seed)
            f = lambda x: float(np.sum(np.atleast_1d(x) ** 2) + rn.normal())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                b = BADS(
                    f,
                    np.full(D, 1.5),
                    np.full(D, -5.0),
                    np.full(D, 5.0),
                    np.full(D, -2.0),
                    np.full(D, 2.0),
                    options=dict(
                        display="off",
                        random_seed=seed,
                        max_fun_evals=200,
                        uncertainty_handling=True,
                    ),
                )
                r = b.optimize()
            errs.append(np.sum(r["x"] ** 2))
        print(
            f"D={D} SD as {'observation' if obs else 'latent (as is)'}: refit checks {CNT['checks']}, refits {CNT['refit']}; true f at result per seed {np.round(errs, 3).tolist()}"
        )
