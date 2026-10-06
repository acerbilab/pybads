"""O-K4: at which evaluation the first search runs, and where the refusal is raised."""
import traceback

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__, flush=True)
print("gpyreg:", gpyreg.__file__, flush=True)
from pybads import BADS

orig_s, orig_p = BADS._search_step_, BADS._poll_step_
EV = []


def s(self, gp):
    EV.append(("search", self.function_logger.func_count))
    return orig_s(self, gp)


def p(self, gp):
    EV.append(("poll", self.function_logger.func_count))
    return orig_p(self, gp)


BADS._search_step_, BADS._poll_step_ = s, p
b = BADS(
    lambda x: float(np.sum(np.asarray(x) ** 2 * np.array([1.0, 10.0, 0.5]))),
    np.array([2.0, -1.0, 1.5]),
    -5 * np.ones(3),
    5 * np.ones(3),
    -3 * np.ones(3),
    3 * np.ones(3),
    options={
        "display": "off",
        "random_seed": 0,
        "max_fun_evals": 60,
        "search_acq_fcn": ("acq_LCB", -1.0),
    },
)
try:
    b.optimize()
except ValueError:
    print("steps before the error:", EV, flush=True)
    print("".join(traceback.format_exc().splitlines(True)[-6:]), flush=True)
