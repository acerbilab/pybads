import logging
import time

import gpyreg
import numpy as np

import pybads
from pybads import BADS
from pybads.utils.iteration_history import IterationHistory

print(pybads.__file__)
print(gpyreg.__file__)
logging.getLogger("BADS").setLevel(logging.WARNING)
acc = {"t": 0.0, "n": 0}
orig = IterationHistory.record


def rec(self, key, value, iteration):
    t = time.perf_counter()
    orig(self, key, value, iteration)
    acc["t"] += time.perf_counter() - t
    acc["n"] += 1


IterationHistory.record = rec
D = 6
f = lambda x: float(np.sum(x**2) + np.sum(np.cos(3 * x)))
t0 = time.perf_counter()
b = BADS(
    f,
    np.full(D, 1.0),
    np.full(D, -5.0),
    np.full(D, 5.0),
    np.full(D, -2.0),
    np.full(D, 2.0),
    options=dict(display="off", random_seed=0, max_fun_evals=200),
)
r = b.optimize()
print(
    "iterations",
    r["iterations"],
    "total %.1fs" % (time.perf_counter() - t0),
    "time in IterationHistory.record %.1fs over %d calls"
    % (acc["t"], acc["n"]),
)
