"""Q3: the GP's noise at uncertainty levels 0, 1 and 2 in real runs: the noise function's
flags and hyperparameters, and at level 2 whether gp.s2 holds, row by row, the squares
of the SDs the target returned for the training inputs."""
import warnings

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
import pybads.bads.bads as bmod
import pybads.bads.gaussian_process_train as gpt
from pybads import BADS

checks = []
FL = {}
orig_lgf = bmod.local_gp_fitting


def wrapped(gp, u, fl, *a, **k):
    FL["fl"] = fl
    out = orig_lgf(gp, u, fl, *a, **k)
    g = out[0]
    row = dict(
        params=tuple(g.noise.parameters),
        noise_N=g.noise.hyperparameter_count(),
        n=len(g.y),
    )
    if g.s2 is not None and fl.he_noise_flag:
        # match each training row to the logger's row
        Xl = fl.X[: fl.Xn + 1]
        Sl = fl.S[: fl.Xn + 1]
        Yl = fl.Y[: fl.Xn + 1]
        errs = []
        yerrs = []
        for x, yv, s2 in zip(g.X, g.y, g.s2):
            idx = np.flatnonzero(np.all(Xl == x, axis=1))
            errs.append(np.min(np.abs(Sl[idx, 0] ** 2 - s2[0])))
            yerrs.append(np.min(np.abs(Yl[idx, 0] - yv[0])))
        row["s2_err"] = max(errs)
        row["y_err"] = max(yerrs)
        h = g.get_hyperparameters(as_array=True)[0]
        cov_N = g.covariance.hyperparameter_count(g.D)
        sn2 = g.noise.compute(h[cov_N : cov_N + 1], g.X, g.y, g.s2)
        row["sn2_err"] = np.max(
            np.abs(np.ravel(sn2) - (np.exp(2 * h[cov_N]) + np.ravel(g.s2)))
        )
        row["s2_nan"] = bool(np.any(np.isnan(g.s2)))
    elif g.s2 is not None:
        row["s2_nan"] = bool(np.all(np.isnan(g.s2)))
    checks.append(row)
    return out


bmod.local_gp_fitting = wrapped

add_checks = []
orig_add = bmod.add_and_update_gp


def wrapped_add(fl, gp, x_new, y_new, sd_new=None, options=None):
    n0 = len(gp.y)
    out = orig_add(fl, gp, x_new, y_new, sd_new, options)
    if options["specify_target_noise"] and len(out.y) > n0:
        add_checks.append(abs(out.s2[-1, 0] - sd_new**2))
    return out


bmod.add_and_update_gp = wrapped_add

D = 2
for level, opts, fun in [
    (0, {}, lambda x: float(np.sum(np.atleast_1d(x) ** 2))),
    (1, {"uncertainty_handling": True}, None),
    (2, {"specify_target_noise": True}, None),
]:
    checks.clear()
    add_checks.clear()
    rng_noise = np.random.default_rng(7)
    if level == 1:
        fun = lambda x: float(
            np.sum(np.atleast_1d(x) ** 2) + rng_noise.normal()
        )
    if level == 2:

        def fun(x):
            f = np.sum(np.atleast_1d(x) ** 2)
            sd = 0.5 + 0.5 * np.sqrt(f)
            return float(f + sd * rng_noise.normal()), float(sd)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = BADS(
            fun,
            np.full(D, 1.0),
            np.full(D, -5.0),
            np.full(D, 5.0),
            np.full(D, -2.0),
            np.full(D, 2.0),
            options=dict(
                display="off", random_seed=5, max_fun_evals=150, **opts
            ),
        )
        r = b.optimize()
    print(
        f"\nlevel {b.optim_state['uncertainty_handling_level']}: fval {r['fval']:.4g} evals {r['func_count']}; rebuilds {len(checks)}"
    )
    print(
        "  noise flags (const, user, rlod):",
        sorted({c["params"] for c in checks}),
        "noise hyp count:",
        sorted({c["noise_N"] for c in checks}),
    )
    print(
        "  gp.s2 all-NaN (levels 0/1) or any NaN (level 2):",
        sorted({c.get("s2_nan") for c in checks}),
    )
    if level == 2:
        print(
            "  max |gp.s2 - S_logger^2| over rebuilds:",
            max(c["s2_err"] for c in checks),
            " max |gp.y - Y_logger|:",
            max(c["y_err"] for c in checks),
        )
        print(
            "  max |noise.compute - (exp(2h)+s2)|:",
            max(c["sn2_err"] for c in checks),
        )
        print(
            "  add_and_update_gp: points added",
            len(add_checks),
            "max |s2_new - sd^2|",
            max(add_checks) if add_checks else None,
        )
