"""F2 (internal) / F1(c) (comparison): the normalization of a Gaussian prior
far outside its bounds, and a fit under it, against the same fit with the
constant set to 1 (the MAP cannot depend on a constant)."""

import time
import warnings

import gpyreg as gpr
import numpy as np
import scipy.stats as st
from common import banner  # noqa: F401

# 1. Where does the mass underflow?  lb > mu: gpyreg uses sf(lb) - sf(ub).
for z in (30, 35, 37, 37.5, 38, 38.5, 39, 40):
    c = st.norm.sf(z) - st.norm.sf(z + 1e4)
    print(f"z={z:5.1f}: sf-difference {c:.3e}   logsf {st.norm.logsf(z):.2f}")

# 2. A GP as PyBADS builds it, on a quadratic in D=2.
rng = np.random.default_rng(0)
D = 2
X = rng.uniform(-1, 1, (30, D))
y = np.sum(X**2, axis=1, keepdims=True) * 5 + 0.05 * rng.standard_normal(
    (30, 1)
)


def make_gp(mean_lb, prior_mu, prior_sd):
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.RationalQuadraticARD(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    b = gp.get_bounds()
    b["covariance_log_lengthscale"] = (
        np.full(D, np.log(1e-6)),
        np.full(D, np.log(20.0)),
    )
    b["covariance_log_outputscale"] = (np.log(1e-3), np.log(1e6 * 1e-3 / 1e-6))
    b["covariance_log_shape"] = (np.array([-5.0]), np.array([5.0]))
    b["noise_log_scale"] = (np.log(1e-3) - 1, 5)
    b["mean_const"] = (np.array([mean_lb]), np.array([mean_lb + 100.0]))
    gp.set_bounds(b)
    p = gp.get_priors()
    p["covariance_log_lengthscale"] = ("gaussian", (-1.0, 2.0))
    p["covariance_log_outputscale"] = ("gaussian", (np.log(np.std(y)), 2.0))
    p["covariance_log_shape"] = ("gaussian", (1.0, 1.0))
    p["noise_log_scale"] = ("gaussian", (np.log(np.sqrt(1e-3)), 1.0))
    p["mean_const"] = ("gaussian", (prior_mu, prior_sd))
    gp.set_priors(p)
    return gp


opts = {
    "init_method": "sobol",
    "tol_opt": 1e-5,
    "widths": None,
    "sampler": "slicesample",
    "init_N": 64,
    "opts_N": 1,
    "n_samples": 0,
}
hyp0 = np.array([[0.0, 0.0, np.log(np.std(y)), 0.0, -3.45, 15.0]])

for label, patch in (
    ("constant as computed", False),
    ("constant set to 1", True),
):
    gp = make_gp(mean_lb=15.0, prior_mu=0.0, prior_sd=0.2)  # z = 75
    i = D + 3
    print(
        f"\n{label}: normalization constant of the mean "
        f"{gp.normalization_constants[i]:.3g}"
    )
    if patch:
        gp.normalization_constants[i] = 1.0
        gp._prior_cache = None
    gp.X, gp.y = X, y
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = hyp0[0]
        gp.set_hyperparameters(hyp0, compute_posterior=False)
        lp = gp.log_posterior(h) if hasattr(gp, "log_posterior") else None
        print(
            "   log posterior at hyp0:",
            lp,
            "  log likelihood:",
            gp.log_likelihood(h),
        )
        t = time.time()
        try:
            hyp, opt_res, _ = gp.fit(
                X,
                y,
                None,
                hyp0=hyp0,
                options=opts,
                rng=np.random.default_rng(1),
            )
        except np.linalg.LinAlgError as e:
            print(f"   fit raised LinAlgError after {time.time()-t:.2f}s: {e}")
            continue
        dt = time.time() - t
    print(f"   fit time {dt:.2f}s; hyp = {np.round(hyp[0], 3)}")
    print(f"   log likelihood at the fit {gp.log_likelihood(hyp[0]):.3f}")
    if patch:
        ref = hyp[0]
    else:
        bad = hyp[0]
    try:
        print(
            "   optimizer result nit/nfev:",
            getattr(opt_res, "nit", None),
            getattr(opt_res, "nfev", None),
        )
    except Exception:
        pass
