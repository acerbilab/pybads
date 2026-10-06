"""_get_random_samples_from_priors_: draws against the priors they claim to sample,
on a GP whose priors are the ones local_gp_fitting sets."""
import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from gpyreg.covariance_functions import RationalQuadraticARD

from pybads.bads.gaussian_process_train import _get_random_samples_from_priors_

D = 2
gp = gpyreg.GP(
    D=D,
    covariance=RationalQuadraticARD(),
    mean=gpyreg.mean_functions.ConstantMean(),
    noise=gpyreg.noise_functions.GaussianNoise(constant_add=True),
)
pri = {
    "covariance_log_lengthscale": (
        "gaussian",
        (np.log(0.3), 2.5),
    ),  # iso empirical prior: centre log geomean distance, SD 0.5*log(dmax/dmin)
    "covariance_log_outputscale": (
        "gaussian",
        (np.log(50.0), 2.0),
    ),  # log std(y) = log 50
    "covariance_log_shape": ("gaussian", (1.0, 1.0)),
    "noise_log_scale": (
        "gaussian",
        (np.log(np.sqrt(1e-3)) + 0.5 * np.log(2.0**-6), 1.0),
    ),
    "mean_const": ("gaussian", (12.0, 0.8)),
}
gp.set_priors(pri)
gp.update(hyp=np.zeros((1, D + 4)))
rng = np.random.default_rng(0)
S = np.array(
    [_get_random_samples_from_priors_(gp, rng)[0] for _ in range(20000)]
)
names = ["log ell0", "log ell1", "log sf", "log alpha", "log sn", "m"]
mu = gp.hyper_priors["mu"]
sd = gp.hyper_priors["sigma"]
print(
    f"{'hyp':10s} {'prior mean':>11s} {'prior sd':>9s} {'sample mean':>12s} {'sample sd':>10s}"
)
for i, n in enumerate(names):
    print(
        f"{n:10s} {mu[i]:11.4g} {sd[i]:9.4g} {S[:, i].mean():12.4g} {S[:, i].std():10.4g}"
    )
print(
    "fraction of log sf draws above log(1e9) (the upper bound):",
    np.mean(S[:, 2] > np.log(1e9)),
)
