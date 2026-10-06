import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__, flush=True)
print("gpyreg:", gpyreg.__file__, flush=True)
from pybads.acquisition_functions.acq_fcn_lcb import acq_fcn_lcb

rng = np.random.default_rng(0)
D = 3
X = rng.normal(size=(20, D))
y = np.sum(X**2, 1, keepdims=True)
gp2 = gpyreg.GP(
    D=D,
    covariance=gpyreg.covariance_functions.RationalQuadraticARD(),
    mean=gpyreg.mean_functions.ConstantMean(),
    noise=gpyreg.noise_functions.GaussianNoise(constant_add=True),
)
gp2.fit(
    X, y, options={"n_samples": 0}, rng=rng
) if "rng" in gpyreg.GP.fit.__code__.co_varnames else gp2.fit(
    X, y, options={"n_samples": 0}
)
try:
    z, fm, fs = acq_fcn_lcb(np.empty((0, D)), 10, gp2)
    print("empty LCB ok:", z.shape, fm.shape, fs.shape)
except Exception as e:
    print("empty LCB raised:", type(e).__name__, e)
z, fm, fs = acq_fcn_lcb(X[:4] + 0.1, 10, gp2)
print("LCB shapes:", z.shape, fm.shape, fs.shape)
