import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
from pybads.acquisition_functions.acq_fcn_lcb import acq_fcn_lcb


class FakeGP:
    def predict(self, x):
        return np.zeros((len(x), 1)), np.ones((len(x), 1))


X = np.zeros((3, 2))
for sb in [
    None,
    np.float64(2.0),
    2.0,
    2,
    np.array([1.0, 2.0]),
    np.inf,
    np.float64(np.inf),
]:
    try:
        z, _, _ = acq_fcn_lcb(X, 10, FakeGP(), sb)
        print(repr(sb), "->", z.ravel()[:1])
    except Exception as e:
        print(repr(sb), "-> raises", type(e).__name__, str(e)[:70])
