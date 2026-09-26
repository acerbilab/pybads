"""upper_gp_length_factor at 0, 0.05, 5: the length-scale bounds of the initial GP."""
import numpy as np

from pybads import BADS

for fac in (0, 0.05, 5.0):
    b = BADS(
        lambda x: float(np.sum(np.ravel(x) ** 2)),
        np.ones(2),
        -5 * np.ones(2),
        5 * np.ones(2),
        -2 * np.ones(2),
        2 * np.ones(2),
        options={
            "display": "off",
            "random_seed": 0,
            "upper_gp_length_factor": fac,
        },
    )
    gp, *_ = b._init_optimization_()
    lo, hi = gp.get_bounds()["covariance_log_lengthscale"]
    print(
        fac,
        np.round(lo, 3),
        np.round(hi, 3),
        np.round(gp.get_hyperparameters(as_array=True), 6),
    )
