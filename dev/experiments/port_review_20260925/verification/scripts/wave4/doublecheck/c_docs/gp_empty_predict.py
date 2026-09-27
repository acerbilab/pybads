"""Does 1.1.0's acq_fcn_lcb reach the ES search's "random search" warning
for a generation that leaves no candidate, that is, does the GP predict on
zero rows? (The changelog's "Search without a candidate", W4-27.)"""

import gpyreg
import numpy as np

import pybads
from pybads.acquisition_functions import acq_fcn_lcb

print(pybads.__file__, gpyreg.__file__, flush=True)

rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, (10, 2))
y = np.sum(X**2, axis=1, keepdims=True)
gp = gpyreg.GP(
    D=2,
    covariance=gpyreg.covariance_functions.SquaredExponential(),
    mean=gpyreg.mean_functions.ConstantMean(),
    noise=gpyreg.noise_functions.GaussianNoise(constant_add=True),
)
gp.fit(
    X=X, y=y, options={"n_samples": 0}, rng=rng
) if "rng" in gp.fit.__code__.co_varnames else gp.fit(
    X=X, y=y, options={"n_samples": 0}
)
z, f_mu, f_s = acq_fcn_lcb(np.empty((0, 2)), 10, gp, None)
print("z", z.shape, "size", z.size)
