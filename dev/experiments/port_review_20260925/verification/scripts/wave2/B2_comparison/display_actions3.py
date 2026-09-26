import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
print(gpyreg.__file__)
# Refit only at the first call, so that later polls do not train
state = {"n": 0}
orig = BADS._is_gp_refit_time_


def refit_once(self, alpha, refit_allowed=True):
    state["n"] += 1
    r = orig(self, alpha, refit_allowed)
    return (r[0] and state["n"] <= 3, r[1])


BADS._is_gp_refit_time_ = refit_once
rows = []
orig_poll = BADS._poll_step_


def poll(self, gp):
    out = orig_poll(self, gp)
    rows.append(
        (
            self.optim_state["iter"] + 1,
            bool(self.gp_refitted_flag),
            self.last_skipped == self.optim_state["iter"],
            self.logging_action[-1],
        )
    )
    return out


BADS._poll_step_ = poll
D = 3
f = lambda x: float(np.sum(x**2) + 0.5 * np.prod(np.cos(2 * x)))
b = BADS(
    f,
    np.array([1.3, -0.7, 0.4]),
    np.full(D, -5.0),
    np.full(D, 5.0),
    np.full(D, -2.0),
    np.full(D, 2.0),
    options=dict(display="off", random_seed=3, max_fun_evals=100),
)
b.optimize()
print("(MATLAB iter, trained this pass, skipped, Actions shown):")
for r in rows:
    print("  ", r)
