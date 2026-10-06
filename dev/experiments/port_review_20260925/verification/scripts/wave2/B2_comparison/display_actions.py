import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
print(gpyreg.__file__)
D = 3
f = lambda x: float(np.sum(x**2) + 0.5 * np.prod(np.cos(2 * x)))
b = BADS(
    f,
    np.array([1.3, -0.7, 0.4]),
    np.full(D, -5.0),
    np.full(D, 5.0),
    np.full(D, -2.0),
    np.full(D, 2.0),
    options=dict(display="iter", random_seed=3, max_fun_evals=120),
)
refits = []
orig = BADS._record_gp_refit_
import pybads.bads.bads as bm

orig_poll = BADS._poll_step_


def poll(self, gp):
    out = orig_poll(self, gp)
    refits.append(
        (
            self.optim_state["iter"] + 1,
            self.gp_refitted_flag,
            self.last_skipped == self.optim_state["iter"],
        )
    )
    return out


BADS._poll_step_ = poll
r = b.optimize()
print("per-iteration (MATLAB iter, refitted, skipped):", refits)
