"""The descriptions that Options reads for the options the pass changed."""
import gpyreg
import numpy as np

import pybads
from pybads import BADS

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
b = BADS(
    lambda x: float(np.sum(x**2)),
    np.zeros(2),
    lower_bounds=-5 * np.ones(2),
    upper_bounds=5 * np.ones(2),
    plausible_lower_bounds=-4 * np.ones(2),
    plausible_upper_bounds=4 * np.ones(2),
    options={"display": "off"},
)
d = b.options.descriptions
for k in (
    "hedge_gamma",
    "gp_rescale_poll",
    "tol_poi",
    "sloppy_improvement",
    "accelerate_mesh_steps",
    "improvement_quantile",
    "search_acq_fcn",
    "n_search_iter",
    "search_method",
    "poll_training",
    "uncertain_incumbent",
    "skip_poll",
    "poll_method",
    "poll_acq_fcn",
    "search_improve_frac",
    "search_optimize",
    "acq_hedge",
):
    print(f"{k}: {d.get(k)!r}", flush=True)
