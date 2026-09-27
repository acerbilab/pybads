"""np.geterr() and the root logger's handlers around a run, and around the
construction of an ES search."""
import logging

import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
from pybads import BADS
from pybads.bads.options import Options  # noqa

root = logging.getLogger()
print("root handlers at start", root.handlers, "level", root.level, flush=True)
from pybads.search import ESSearchWM

opts = {
    "poll_mesh_multiplier": 2.0,
    "es_start": 0.25,
    "n_search_iter": 2,
    "search_acq_fcn": ("acq_LCB", None),
    "es_beta": 1,
}
ESSearchWM(8, 8, opts, rng=np.random.default_rng(0))
print("root handlers after ESSearchWM()", root.handlers, flush=True)
before = np.geterr()
b = BADS(
    lambda x: float(np.sum(x**2)),
    np.array([2.0, 2.0]),
    lower_bounds=-5 * np.ones(2),
    upper_bounds=5 * np.ones(2),
    plausible_lower_bounds=-4 * np.ones(2),
    plausible_upper_bounds=4 * np.ones(2),
    options={"display": "off", "random_seed": 0, "max_fun_evals": 60},
)
print("root handlers after BADS()", root.handlers, flush=True)
b.optimize()
print("geterr before", before, flush=True)
print("geterr after ", np.geterr(), flush=True)
