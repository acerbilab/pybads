import logging
import threading

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from pybads import BADS

logging.getLogger("BADS").setLevel(logging.ERROR)


class Model:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = np.zeros(10)
        self.ncalls = 0

    def nll(self, x):
        with self.lock:
            self.ncalls += 1
        return float(np.sum(np.ravel(x) ** 2))


m = Model()
b = BADS(
    m.nll,
    x0=np.ones(2),
    lower_bounds=-5 * np.ones(2),
    upper_bounds=5 * np.ones(2),
    plausible_lower_bounds=-2 * np.ones(2),
    plausible_upper_bounds=2 * np.ones(2),
    options={"random_seed": 0, "display": "off", "max_fun_evals": 30},
)
try:
    r = b.optimize()
    print("ok; fun is user's:", r["fun"] is m.nll, r["fun"].__self__ is m)
except Exception as e:
    print(
        "optimize() raised at the end:",
        type(e).__name__,
        e,
        "| target calls made:",
        m.ncalls,
        "| b.x =",
        getattr(b, "x", None),
    )
m2 = Model()
del m2.lock
m2.nll = Model.nll.__get__(m2)


class M3:
    def __init__(self):
        self.ncalls = 0

    def __call__(self, x):
        self.ncalls += 1
        return float(np.sum(np.ravel(x) ** 2))


m3 = M3()
b = BADS(
    m3,
    x0=np.ones(2),
    lower_bounds=-5 * np.ones(2),
    upper_bounds=5 * np.ones(2),
    plausible_lower_bounds=-2 * np.ones(2),
    plausible_upper_bounds=2 * np.ones(2),
    options={"random_seed": 0, "display": "off", "max_fun_evals": 30},
)
r = b.optimize()
print(
    "callable object: result fun is user's object:",
    r["fun"] is m3,
    "; copy ncalls:",
    r["fun"].ncalls,
)
