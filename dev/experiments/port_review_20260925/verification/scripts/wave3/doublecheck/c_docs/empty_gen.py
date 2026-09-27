"""A non_box_cons that rejects every candidate of the second generation of
the ES search (every second call with more than 100 points), all else
feasible: the warning and what the run does."""
import logging
import traceback

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)


class Count(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, record):
        if "es_search" in record.getMessage():
            self.msgs.append(record.getMessage())


h = Count()
logging.getLogger("BADS").addHandler(h)
big = [0]


def nbc(X):
    X = np.atleast_2d(X)
    if X.shape[0] > 100:
        big[0] += 1
        if big[0] % 2 == 0:
            return np.ones(X.shape[0], dtype=bool)
    return np.zeros(X.shape[0], dtype=bool)


b = BADS(
    lambda x: float(np.sum(np.ravel(x) ** 2)),
    np.array([2.0, 2.0]),
    lower_bounds=-5 * np.ones(2),
    upper_bounds=5 * np.ones(2),
    plausible_lower_bounds=-4 * np.ones(2),
    plausible_upper_bounds=4 * np.ones(2),
    non_box_cons=nbc,
    options={"display": "off", "random_seed": 0, "max_fun_evals": 60},
)
try:
    r = b.optimize()
    print("ran:", r["fval"], r["func_count"], flush=True)
except Exception as e:
    tb = traceback.extract_tb(e.__traceback__)[-1]
    print(f"{type(e).__name__}: {e} at {tb.name}:{tb.lineno}", flush=True)
print("ES warnings:", len(h.msgs), set(h.msgs), flush=True)
