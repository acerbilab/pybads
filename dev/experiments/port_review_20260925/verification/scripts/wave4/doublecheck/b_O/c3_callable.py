"""acq_fcn_lcb with a callable sqrt_beta returning various values (W4-19)."""
from decimal import Decimal

import hdr  # noqa
import numpy as np

from pybads.acquisition_functions import acq_fcn_lcb, check_sqrt_beta


class G:
    def predict(self, x):
        return np.zeros((len(x), 1)), np.ones((len(x), 1))


xi = np.zeros((3, 2))
vals = [
    -1,
    0,
    np.nan,
    np.inf,
    2.0,
    np.float64(2),
    np.int64(2),
    np.array([2.0]),
    np.array([[2.0]]),
    [2.0],
    True,
    np.True_,
    "2",
    None,
    2 + 0j,
    np.complex128(2),
    [1, 2],
    Decimal(2),
    1e-300,
]
for v in vals:
    try:
        z, fmu, fs = acq_fcn_lcb(xi, 5, G(), lambda t, d, v=v: v)
        print(f"{v!r:28s} accepted: z={np.ravel(z)[0]!r}")
    except ValueError as e:
        print(f"{v!r:28s} ValueError: {e}")
    except Exception as e:
        print(f"{v!r:28s} {type(e).__name__}: {e}")
