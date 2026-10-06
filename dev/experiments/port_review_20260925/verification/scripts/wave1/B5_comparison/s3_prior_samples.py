"""_get_random_samples_from_priors_ against the priors the GP holds
(MATLAB gppriorrnd.m draws each hyperparameter, on its own log scale, from
its Gaussian prior). GP captured at the first refit of a short run."""
import copy

import gpyreg
import numpy as np

import pybads
import pybads.bads.gaussian_process_train as gpt
from pybads import BADS

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)

captured = {}
_orig = gpt._robust_gp_fit_


def spy(*args, **kwargs):
    if not captured:
        captured["gp"] = copy.deepcopy(args[0])
    return _orig(*args, **kwargs)


gpt._robust_gp_fit_ = spy
D = 2
b = BADS(
    lambda x: float(np.sum(np.ravel(x) ** 2)),
    np.full((1, D), 1.0),
    np.full((1, D), -5.0),
    np.full((1, D), 5.0),
    np.full((1, D), -2.0),
    np.full((1, D), 2.0),
    options={"display": "off", "random_seed": 0, "max_fun_evals": 60},
)
b.optimize()
gpt._robust_gp_fit_ = _orig
gp = captured["gp"]

rng = np.random.default_rng(0)
S = np.vstack(
    [gpt._get_random_samples_from_priors_(gp, rng) for _ in range(4000)]
)
mu, sigma = gp.hyper_priors["mu"], gp.hyper_priors["sigma"]
names = []
for key, val in gp.get_priors().items():
    k = np.size(val[1][0]) if val[0] == "gaussian" else 1
    names += [key] * max(1, k)
info = gp.get_hyperparameters()[0]
labels = []
for key in info:
    labels += [key] * np.size(info[key])
print(
    f"{'hyperparameter':30s} {'prior mu':>9s} {'prior sd':>9s} {'draw mean':>10s} {'draw sd':>9s}"
)
for i, lab in enumerate(labels):
    print(
        f"{lab:30s} {mu[i]:9.3f} {sigma[i]:9.3f} {S[:, i].mean():10.3f} {S[:, i].std():9.3f}"
    )
