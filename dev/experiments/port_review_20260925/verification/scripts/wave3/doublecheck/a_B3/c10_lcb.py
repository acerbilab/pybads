"""W3-10: which values of sqrt_beta acq_fcn_lcb accepts."""
from fractions import Fraction

import hdr  # noqa: F401
import numpy as np

from pybads.acquisition_functions.acq_fcn_lcb import acq_fcn_lcb


class G:
    def predict(self, x):
        return np.ones((len(x), 1)), 4 * np.ones((len(x), 1))


vals = {
    "None": None,
    "2.0": 2.0,
    "2": 2,
    "np.float32(2)": np.float32(2),
    "np.uint8(2)": np.uint8(2),
    "np.float16(2)": np.float16(2),
    "[2.0]": [2.0],
    "(2.0,)": (2.0,),
    "np.array([[[2.0]]])": np.array([[[2.0]]]),
    "1e-300": 1e-300,
    "1e308": 1e308,
    "0.0": 0.0,
    "-0.0": -0.0,
    "np.nan": np.nan,
    "np.inf": np.inf,
    "True": True,
    "np.True_": np.True_,
    "2+0j": 2 + 0j,
    "'2'": "2",
    "Fraction(2)": Fraction(2),
    "lambda": (lambda t, n: 1.5),
    "[]": [],
    "np.array([])": np.array([]),
}
for k, v in vals.items():
    try:
        z, _, _ = acq_fcn_lcb(np.zeros((3, 2)), 5, G(), v)
        print(
            f"{k:22s} accepted, z shape {z.shape}, z[0] {float(np.ravel(z)[0]):.4g}"
        )
    except ValueError as e:
        print(f"{k:22s} ValueError")
    except Exception as e:
        print(f"{k:22s} {type(e).__name__}: {e}")
