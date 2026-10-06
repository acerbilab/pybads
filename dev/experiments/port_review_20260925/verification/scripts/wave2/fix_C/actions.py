import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
state = {"n": 0}
orig = BADS._is_gp_refit_time_


def refit_once(self, alpha, refit_allowed=True):
    state["n"] += 1
    r = orig(self, alpha, refit_allowed)
    return (r[0] and state["n"] <= NREF, r[1])


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
for NREF in [3, 1000]:
    for sn in [0, None]:
        for seed in [3, 0]:
            rows.clear()
            state["n"] = 0
            opts = dict(display="off", random_seed=seed, max_fun_evals=100)
            if sn is not None:
                opts["search_n_try"] = sn
            b = BADS(
                f,
                np.array([1.3, -0.7, 0.4]),
                np.full(D, -5.0),
                np.full(D, 5.0),
                np.full(D, -2.0),
                np.full(D, 2.0),
                options=opts,
            )
            b.optimize()
            stale = sum(
                ("Train" in a) != t or (("kip" in a) != s)
                for _, t, s, a in rows
            )
            both = sum(t and s for _, t, s, a in rows)
            print(
                f"NREF={NREF} search_n_try={sn} seed={seed}: polls={len(rows)} mismatched={stale} train&skip={both}",
                [r for r in rows if r[1] or r[2]][:6],
            )
