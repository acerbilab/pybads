"""W4-28: does _poll_step_ return the GP object it was given? The six runs
of dev/scripts/fingerprint.py."""
import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__, gpyreg.__file__, flush=True)
orig = BADS._poll_step_
same = []


def poll(self, gp):
    out = orig(self, gp)
    same.append(out[-1] is gp)
    return out


BADS._poll_step_ = poll
g = np.random.default_rng(0)
for noisy in (False, True):
    for seed in range(3):
        o = {"display": "off", "max_fun_evals": 80, "random_seed": seed}
        if noisy:
            o["uncertainty_handling"] = True
        fun = (
            (
                lambda x: float(
                    np.sum(np.atleast_2d(x) ** 2) + g.standard_normal()
                )
            )
            if noisy
            else (lambda x: float(np.sum(np.atleast_2d(x) ** 2)))
        )
        BADS(
            fun,
            np.ones(3) * 4,
            -100 * np.ones(3),
            100 * np.ones(3),
            -8 * np.ones(3),
            12 * np.ones(3),
            options=o,
        ).optimize()
print(f"polls {len(same)}, the returned GP is the given one in {sum(same)}")
