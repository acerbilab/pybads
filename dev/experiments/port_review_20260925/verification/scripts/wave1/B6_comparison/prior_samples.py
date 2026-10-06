"""_get_random_samples_from_priors_ vs MATLAB gppriorrnd (priorGauss: mu + sqrt(s2)*randn in the
hyperparameter's own units)."""

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
import gpyreg as gpr

from pybads.bads.gaussian_process_train import _get_random_samples_from_priors_

D = 3
gp = gpr.GP(
    D=D,
    covariance=gpr.covariance_functions.RationalQuadraticARD(),
    mean=gpr.mean_functions.ConstantMean(),
    noise=gpr.noise_functions.GaussianNoise(constant_add=True),
)
# priors as local_gp_fitting leaves them (iso length scale, output scale, shape, noise at level 0, mean)
pri = {
    "covariance_log_lengthscale": (
        "gaussian",
        (np.full(D, -0.8), np.full(D, 1.5)),
    ),
    "covariance_log_outputscale": (
        "gaussian",
        (np.array([np.log(3.0)]), np.array([2.0])),
    ),
    "covariance_log_shape": ("gaussian", (np.array([1.0]), np.array([1.0]))),
    "noise_log_scale": (
        "gaussian",
        (np.array([np.log(np.sqrt(1e-3))]), np.array([1.0])),
    ),
    "mean_const": ("gaussian", (np.array([5.0]), np.array([0.4]))),
}
gp.set_priors(pri)
X = np.random.default_rng(0).normal(size=(10, D))
y = np.sum(X**2, 1, keepdims=True)
gp.update(X_new=X, y_new=y, hyp=np.zeros((1, D + 4)), compute_posterior=False)
rng = np.random.default_rng(0)
S = np.array(
    [_get_random_samples_from_priors_(gp, rng)[0] for _ in range(20000)]
)
names = ["log ell"] * D + ["log sf", "log alpha", "log sn", "mean"]
mus = np.concatenate(
    [
        pri[k][1][0]
        for k in [
            "covariance_log_lengthscale",
            "covariance_log_outputscale",
            "covariance_log_shape",
            "noise_log_scale",
            "mean_const",
        ]
    ]
)
sds = np.concatenate(
    [
        pri[k][1][1]
        for k in [
            "covariance_log_lengthscale",
            "covariance_log_outputscale",
            "covariance_log_shape",
            "noise_log_scale",
            "mean_const",
        ]
    ]
)
print(
    f"{'hyp':10s} {'prior mu':>9s} {'prior sd':>9s} | {'sample mean':>11s} {'sample sd':>9s}"
)
for i, n in enumerate(names):
    print(
        f"{n:10s} {mus[i]:9.3f} {sds[i]:9.3f} | {S[:, i].mean():11.3f} {S[:, i].std():9.3f}"
    )
