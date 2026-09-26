"""F14 (internal): init_and_train_gp retries without a cap."""
import logging

import common  # noqa
import gpyreg as gpr
import numpy as np

from pybads import BADS

logging.getLogger("BADS").setLevel(logging.CRITICAL)
C = {"n": 0}
orig = gpr.GP.fit


class Stop(Exception):
    pass


def fit(self, *a, **k):
    C["n"] += 1
    if C["n"] >= 50:
        raise Stop(f"stopped by the check after {C['n']} calls")
    raise np.linalg.LinAlgError("injected")


gpr.GP.fit = fit
try:
    BADS(
        lambda x: float(np.sum(np.ravel(x) ** 2)),
        0.5 * np.ones((1, 2)),
        -5 * np.ones((1, 2)),
        5 * np.ones((1, 2)),
        -2 * np.ones((1, 2)),
        2 * np.ones((1, 2)),
        options={"random_seed": 0, "display": "off", "max_fun_evals": 30},
    ).optimize()
except Stop as e:
    print("init_and_train_gp kept retrying:", e)
finally:
    gpr.GP.fit = orig
