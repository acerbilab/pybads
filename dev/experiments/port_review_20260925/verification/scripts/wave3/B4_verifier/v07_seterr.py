"""I-F10 / K5: a default run changes NumPy's global error state."""
import logging
import warnings

import numpy as np
from vhdr import box, sphere

from pybads import BADS

print(
    "root logger level:",
    logging.getLogger().level,
    "| geterr before:",
    np.geterr(),
)
D = 2
lb, ub, plb, pub = box(D)
BADS(
    sphere,
    np.ones(D),
    lb,
    ub,
    plb,
    pub,
    options=dict(random_seed=0, display="off", max_fun_evals=60),
).optimize()
print("geterr after:", np.geterr())
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    np.array([1.0]) / np.array([0.0])
    print("user 1/0 after the run warns:", len(w) > 0)
