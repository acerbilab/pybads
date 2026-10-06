import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
print(gpyreg.__file__)
rows = []
orig_poll = BADS._poll_step_


def poll(self, gp):
    out = orig_poll(self, gp)
    rows.append(
        (
            self.optim_state["iter"] + 1,
            self.gp_refitted_flag,
            self.last_skipped == self.optim_state["iter"],
            self.logging_action[-1],
        )
    )
    return out


BADS._poll_step_ = poll
for seed in range(4):
    rows.clear()
    D = 2
    f = lambda x: float(np.sum(x**2) + 0.5 * np.prod(np.cos(2 * x)))
    b = BADS(
        f,
        np.array([1.3, -0.7]),
        np.full(D, -5.0),
        np.full(D, 5.0),
        np.full(D, -2.0),
        np.full(D, 2.0),
        options=dict(
            display="off",
            random_seed=seed,
            max_fun_evals=200,
            poll_training=False,
        ),
    )
    b.optimize()
    bad = [
        r
        for r in rows
        if (not r[1] and not r[2] and r[3] != "") or (r[1] and r[2])
    ]
    print(
        f"seed {seed}: polls {len(rows)}; polls whose Actions column does not match MATLAB's (iter, refitted, skipped, shown):",
        bad[:6],
    )
