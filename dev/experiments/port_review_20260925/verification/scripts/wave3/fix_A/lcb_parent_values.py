import warnings

import numpy as np

import pybads
from pybads.acquisition_functions import acq_fcn_lcb

print(pybads.__file__)


class G:
    def predict(self, x):
        return np.array([[1.0], [2.0]]), np.array([[4.0], [0.25]])


vals = {
    "2.0": 2.0,
    "2": 2,
    "0.0": 0.0,
    "-1.0": -1.0,
    "np.float64(0)": np.float64(0.0),
    "np.float64(-1)": np.float64(-1.0),
    "inf": np.inf,
    "nan": np.nan,
    "'name'": "acq_schedule",
    "array([1,2])": np.array([1.0, 2.0]),
    "array([])": np.array([]),
    "True": True,
    "np.True_": np.True_,
    "np.complex128(2)": np.complex128(2.0),
    "array([[[2.]]])": np.array([[[2.0]]]),
}
for k, v in vals.items():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            z, _, _ = acq_fcn_lcb(np.zeros((2, 3)), 9, G(), v)
            print(
                f"{k:20s} ran: z={z.ravel()} shape={z.shape} dtype={z.dtype}",
                [str(x.category.__name__) for x in w],
            )
        except Exception as e:
            print(f"{k:20s} {type(e).__name__}: {str(e)[:70]}")
