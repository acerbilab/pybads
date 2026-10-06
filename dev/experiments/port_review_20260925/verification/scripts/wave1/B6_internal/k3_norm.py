"""Q2: gpyreg's log prior and normalization when a Gaussian prior's centre lies outside
its bounds, down to where the mass inside underflows."""
import warnings

import gpyreg
import numpy as np
import scipy.stats as st

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from gpyreg.covariance_functions import RationalQuadraticARD

warnings.simplefilter("always")
rng = np.random.default_rng(0)
D, N = 2, 25
X = rng.uniform(-1, 1, (N, D))
y = (np.sum(X**2, 1))[:, None]


def make(mean_mu, mean_sd, mlb, mub):
    gp = gpyreg.GP(
        D=D,
        covariance=RationalQuadraticARD(),
        mean=gpyreg.mean_functions.ConstantMean(),
        noise=gpyreg.noise_functions.GaussianNoise(constant_add=True),
    )
    gp.X, gp.y = X, y
    lb = np.array(
        [np.log(1e-6)] * D + [np.log(1e-3), -5, np.log(1e-3) - 1, mlb]
    )
    ub = np.array([np.log(20)] * D + [np.log(1e9), 5, 5, mub])
    gp.set_bounds(gp.bounds_to_dict(lb, ub))
    gp.set_priors(
        {
            "covariance_log_lengthscale": ("gaussian", (-0.5, 1.2)),
            "covariance_log_outputscale": (
                "gaussian",
                (np.log(np.std(y)), 2.0),
            ),
            "covariance_log_shape": ("gaussian", (1.0, 1.0)),
            "noise_log_scale": ("gaussian", (-3.45, 1.0)),
            "mean_const": ("gaussian", (mean_mu, mean_sd)),
        }
    )
    return gp


hyp = np.array([0.0, 0.0, 0.5, 1.0, -4.0, 0.5])
print(
    " z (sd beyond ub)   norm const        exact log mass       gpyreg log posterior - log posterior at z=0 prior"
)
ref = make(0.5, 1.0, 0.0, 1.0)
for z in [0, 5, 20, 37, 38, 39, 60]:
    gp = make(
        1.0 + z * 1.0, 1.0, 0.0, 1.0
    )  # centre z SDs above the upper bound 1
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        lp = gp.log_posterior(hyp)
    exact = (
        st.norm.logsf(0.0, 1.0 + z, 1.0)
        if False
        else np.logaddexp(st.norm.logcdf(1.0, 1.0 + z, 1.0), -np.inf)
    )  # log(cdf(ub)-cdf(lb)) ~ log cdf(ub)
    print(
        f" {z:4d}  {gp.normalization_constants[-1]:14.4g}  {exact:18.6g}   {lp:14.6g}   warn={[str(x.message)[:40] for x in w]}"
    )
# a fit with the prior 60 SDs outside
gp = make(61.0, 1.0, 0.0, 1.0)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try:
        h, res, _ = gp.fit(
            X,
            y,
            hyp0=hyp[None, :],
            options={
                "opts_N": 1,
                "init_N": 8,
                "init_method": "rand",
                "n_samples": 0,
                "tol_opt": 1e-5,
            },
            rng=np.random.default_rng(1),
        )
        print(
            "fit with nc=0: hyp",
            np.round(h, 3),
            "fun",
            res.fun,
            "success",
            res.success,
            res.message,
            "nit",
            res.nit,
        )
    except Exception as e:
        print("fit raised", type(e).__name__, e)
    print("  warnings:", sorted({str(x.message)[:60] for x in w}))
gp2 = make(1.0 + 30, 1.0, 0.0, 1.0)  # finite normalization
h2, res2, _ = gp2.fit(
    X,
    y,
    hyp0=hyp[None, :],
    options={
        "opts_N": 1,
        "init_N": 8,
        "init_method": "rand",
        "n_samples": 0,
        "tol_opt": 1e-5,
    },
    rng=np.random.default_rng(1),
)
print(
    "fit with z=30 (finite nc):  hyp",
    np.round(h2, 3),
    "fun",
    res2.fun,
    "nit",
    res2.nit,
)
