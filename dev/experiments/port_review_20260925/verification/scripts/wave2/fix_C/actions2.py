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


def sphere(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


for NREF in [3, 1000]:
    for sn in [0, None]:
        for tp in [0.5, 0.9]:
            rows.clear()
            state["n"] = 0
            opts = dict(
                display="off",
                random_seed=3,
                max_fun_evals=100,
                min_failed_poll_steps=1,
                tol_poi=tp,
            )
            if sn is not None:
                opts["search_n_try"] = sn
            b = BADS(
                sphere,
                np.ones(D) * 4,
                -100 * np.ones(D),
                100 * np.ones(D),
                -8 * np.ones(D),
                12 * np.ones(D),
                options=opts,
            )
            r = b.optimize()
            stale = sum(
                ("Train" in a) != t or (("kip" in a) != s)
                for _, t, s, a in rows
            )
            both = sum(t and s for _, t, s, a in rows)
            print(
                f"NREF={NREF} sn={sn} tol_poi={tp}: fc={r['func_count']} polls={len(rows)} mismatched={stale} train&skip={both}",
                [x for x in rows if x[1] or x[2]][:8],
            )
