"""Q1/Q3: the fit objective of the GP as PyBADS builds it: log marginal likelihood
with the noise exp(2*h_noise) (+ s2 at level 2) and the Gaussian log priors,
against an independent computation, and its gradient against finite differences."""
import gpyreg
import numpy as np
import scipy.stats as st

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from gpyreg.covariance_functions import RationalQuadraticARD


def indep_nlZ(hyp, X, y, s2, D):
    ell = np.exp(hyp[:D])
    sf2 = np.exp(2 * hyp[D])
    a = np.exp(hyp[D + 1])
    sn2 = np.exp(2 * hyp[D + 2])
    m = hyp[D + 3]
    d2 = (((X[:, None, :] - X[None, :, :]) / ell) ** 2).sum(-1)
    K = sf2 * (1 + d2 / (2 * a)) ** (-a)
    C = K + np.diag(sn2 + (np.zeros(len(y)) if s2 is None else s2.ravel()))
    return -st.multivariate_normal(mean=np.full(len(y), m), cov=C).logpdf(
        y.ravel()
    )


rng = np.random.default_rng(1)
for level in (0, 2):
    D = 3
    N = 20
    X = rng.uniform(-1, 1, (N, D))
    y = (np.sum(X**2, 1) + 0.1 * rng.normal(size=N))[:, None]
    s2 = rng.uniform(0.01, 0.2, (N, 1)) if level == 2 else None
    noise = gpyreg.noise_functions.GaussianNoise(
        constant_add=True, user_provided_add=(level == 2)
    )
    gp = gpyreg.GP(
        D=D,
        covariance=RationalQuadraticARD(),
        mean=gpyreg.mean_functions.ConstantMean(),
        noise=noise,
    )
    gp.X, gp.y, gp.s2 = X, y, s2
    lb = np.array(
        [np.log(1e-6)] * D
        + [np.log(1e-3), -5, np.log(1e-3) - 1, np.min(y) - 1]
    )
    ub = np.array([np.log(20)] * D + [np.log(1e9), 5, 5, np.max(y) + 1])
    b = gp.bounds_to_dict(lb, ub)
    gp.set_bounds(b)
    pri = {
        "covariance_log_lengthscale": ("gaussian", (-0.5, 1.2)),
        "covariance_log_outputscale": ("gaussian", (np.log(np.std(y)), 2.0)),
        "covariance_log_shape": ("gaussian", (1.0, 1.0)),
        "noise_log_scale": (
            "gaussian",
            (np.log(1e-3) if level == 2 else -3.45, 1.0),
        ),
        "mean_const": (
            "gaussian",
            (np.percentile(y, 90, method="hazen"), 0.2),
        ),
    }
    gp.set_priors(pri)
    hyp = np.concatenate(
        [rng.normal(-0.5, 0.5, D), [0.3], [0.5], [-2.0], [np.median(y)]]
    )
    gp.update(hyp=hyp[None, :])
    nlZ_nop = -gp.log_likelihood(hyp)
    print(
        f"level {level}: nlZ gpyreg {nlZ_nop:.10f}  independent {indep_nlZ(hyp, X, y, s2, D):.10f}"
    )
    # log prior: gaussian densities truncated by the bounds, normalized
    mu = gp.hyper_priors["mu"]
    sd = gp.hyper_priors["sigma"]
    lp_ind = np.sum(
        st.norm.logpdf(hyp, mu, sd)
        - np.log(st.norm.cdf(ub, mu, sd) - st.norm.cdf(lb, mu, sd))
    )
    lpost = gp.log_posterior(hyp)
    print(
        f"   log prior gpyreg {lpost + nlZ_nop:.10f}  independent (truncated normal) {lp_ind:.10f}"
    )
    f, g = gp._GP__gp_obj_fun(hyp, True, False)
    fd = np.array(
        [
            (
                gp._GP__gp_obj_fun(hyp + e, False, False)
                - gp._GP__gp_obj_fun(hyp - e, False, False)
            )
            / 2e-6
            for e in np.eye(len(hyp)) * 1e-6
        ]
    )
    print(
        "   obj grad max |analytic - FD| / max|grad|:",
        np.max(np.abs(g - fd)) / np.max(np.abs(g)),
    )
    fs = gp._GP__gp_obj_fun(hyp, False, True)
    print(
        "   slice-sampler objective (swap_sign) == log posterior:",
        np.isclose(fs, lpost),
    )
