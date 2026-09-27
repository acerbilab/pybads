import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__, "gpyreg", gpyreg.__file__, flush=True)
from pybads import BADS

f = lambda x: float(np.sum(x**2))
for pv, x0 in [
    ([0], None),
    ([5], None),
    ([5], np.zeros(2)),
    ([], np.zeros(2)),
    (np.array([], dtype=int), None),
]:
    try:
        BADS(
            f,
            x0,
            -np.ones(2),
            np.ones(2),
            -0.5 * np.ones(2),
            0.5 * np.ones(2),
            options={"periodic_vars": pv, "display": "off", "random_seed": 0},
        )
        print(pv, x0 is None, "accepted")
    except Exception as e:
        print(
            pv,
            "x0 None" if x0 is None else "x0 given",
            "->",
            type(e).__name__,
            str(e)[:90],
        )
