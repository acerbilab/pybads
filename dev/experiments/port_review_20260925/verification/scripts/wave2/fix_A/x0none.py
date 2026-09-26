import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
f = lambda x: float(np.sum(np.asarray(x) ** 2))
O = {"display": "off", "random_seed": 0}
cases = {
    "lb, ub": dict(lower_bounds=-5 * np.ones(2), upper_bounds=5 * np.ones(2)),
    "plb, pub": dict(
        plausible_lower_bounds=-2 * np.ones(2),
        plausible_upper_bounds=2 * np.ones(2),
    ),
    "lb, pub": dict(
        lower_bounds=-5 * np.ones(2), plausible_upper_bounds=2 * np.ones(2)
    ),
    "lb only": dict(lower_bounds=-5 * np.ones(2)),
    "plb, ub": dict(
        plausible_lower_bounds=-2 * np.ones(2), upper_bounds=5 * np.ones(2)
    ),
    "nothing": dict(),
}
for k, v in cases.items():
    try:
        b = BADS(f, None, options=O, **v)
        print(k, "accepted", b.x0)
    except Exception as e:
        print(k, type(e).__name__, str(e).split("\n")[0][:70])
