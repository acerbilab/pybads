"""F3 (internal) / F2 (comparison): _get_random_samples_from_priors_ against
MATLAB's gppriorrnd -> priorGauss (lp = sqrt(s2)*randn + mu, in log units)."""

import gpyreg as gpr
import numpy as np
from common import gpt

D = 2


def gp_with_priors(mean):
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.RationalQuadraticARD(),
        mean=mean,
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    p = gp.get_priors()
    p["covariance_log_lengthscale"] = ("gaussian", (-1.2, 2.5))
    p["covariance_log_outputscale"] = ("gaussian", (3.91, 2.0))
    p["covariance_log_shape"] = ("gaussian", (1.0, 1.0))
    p["noise_log_scale"] = ("gaussian", (-3.45, 1.0))
    if isinstance(mean, gpr.mean_functions.ConstantMean):
        p["mean_const"] = ("gaussian", (5.0, 0.5))
    gp.set_priors(p)
    X = np.random.default_rng(0).uniform(-1, 1, (10, D))
    gp.update(
        X_new=X,
        y_new=np.sum(X**2, 1, keepdims=True),
        hyp=np.zeros((1, np.size(gp.hyper_priors["mu"]))),
        compute_posterior=False,
    )
    return gp


gp = gp_with_priors(gpr.mean_functions.ConstantMean())
names = ["log ell_1", "log ell_2", "log sf", "log alpha", "log sn", "mean"]
prior = [
    (-1.2, 2.5),
    (-1.2, 2.5),
    (3.91, 2.0),
    (1.0, 1.0),
    (-3.45, 1.0),
    (5.0, 0.5),
]
rng = np.random.default_rng(0)
S = np.array(
    [gpt._get_random_samples_from_priors_(gp, rng)[0] for _ in range(20000)]
)
print(
    f"{'hyperparameter':>14} | prior (mu, sd) | PyBADS draws (mean, sd)"
    " | N(exp mu, exp sd)"
)
for j, n in enumerate(names):
    mu, sd = prior[j]
    em = (np.exp(mu), np.exp(sd)) if "log" in n else (mu, sd)
    print(
        f"{n:>14} | ({mu:5.2f}, {sd:4.2f})  | ({S[:,j].mean():7.3f}, "
        f"{S[:,j].std():6.3f})       | ({em[0]:.3f}, {em[1]:.3f})"
    )
ub_sf = np.log(1e6 * 1e-3 / 1e-6)
print(
    f"share of log sf draws above its upper bound {ub_sf:.2f}: "
    f"{np.mean(S[:,2] > ub_sf):.5f}"
)

# A block without a prior: gp_mean_fun = 'negquad', for which _gp_hyp sets no
# prior on the mean's blocks.
gq = gp_with_priors(gpr.mean_functions.NegativeQuadratic())
print(
    "\npriors of the negquad GP's mean blocks:",
    {k: v for k, v in gq.get_priors().items() if k.startswith("mean")},
)
try:
    gpt._get_random_samples_from_priors_(gq, rng)
    print("no error")
except Exception as e:
    print(
        "negquad: _get_random_samples_from_priors_ raises",
        type(e).__name__,
        "-",
        e,
    )
