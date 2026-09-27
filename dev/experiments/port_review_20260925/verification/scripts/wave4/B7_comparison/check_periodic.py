import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
from pybads import BADS
from pybads.utils import period_check

x = np.array([[3.5, -7.0]])
print(
    "stub returns its input object:",
    period_check(x, -np.ones((1, 2)), np.ones((1, 2)), None) is x,
    flush=True,
)
for pv in ([], [0]):
    try:
        BADS(
            lambda x: float(np.sum(x**2)),
            np.zeros(2),
            -np.ones(2) * 5,
            np.ones(2) * 5,
            -np.ones(2),
            np.ones(2),
            options={"periodic_vars": pv, "display": "off", "random_seed": 0},
        )
        print(pv, "accepted", flush=True)
    except ValueError as e:
        print(pv, "ValueError:", str(e)[:80], flush=True)
