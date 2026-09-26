"""F10 (internal): with poll_training=False, refits recorded by _is_gp_refit_time_ and then cancelled."""
import logging

import common  # noqa
import numpy as np

import pybads.bads.bads as bb
from pybads import BADS

logging.getLogger("BADS").setLevel(logging.ERROR)
orig_lgf = bb.local_gp_fitting
C = {"rec": 0, "done": 0}


def lgf(gp, u, fl, o, os_, ih, refit_flag, rng=None):
    C["done"] += bool(refit_flag)
    return orig_lgf(gp, u, fl, o, os_, ih, refit_flag, rng=rng)


bb.local_gp_fitting = lgf


class B(BADS):
    def _record_gp_refit_(self):
        C["rec"] += 1
        return super()._record_gp_refit_()


def ell(x):
    x = np.ravel(x)
    D = x.size
    return float(np.sum((10 ** (np.arange(D) / (D - 1) * 2) * x) ** 2))


D = 3
for pt in (True, False):
    C.update(rec=0, done=0)
    b = B(
        ell,
        0.5 * np.ones((1, D)),
        -5 * np.ones((1, D)),
        5 * np.ones((1, D)),
        -2 * np.ones((1, D)),
        2 * np.ones((1, D)),
        options={
            "random_seed": 70,
            "display": "off",
            "max_fun_evals": 200,
            "poll_training": pt,
        },
    )
    r = b.optimize()
    print(
        f"poll_training={pt}: refits recorded {C['rec']} (the initial reset of gp_stats is not counted), performed {C['done']}, fval {r['fval']:.3e}"
    )
