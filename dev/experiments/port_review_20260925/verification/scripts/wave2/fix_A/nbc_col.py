import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
f = lambda x: float(np.sum(np.asarray(x) ** 2))
for label, nbc in [
    ("(N,)", lambda x: np.sum(np.atleast_2d(x) ** 2, 1) > 1),
    ("(N,1)", lambda x: (np.sum(np.atleast_2d(x) ** 2, 1) > 1)[:, None]),
    ("(1,N)", lambda x: (np.sum(np.atleast_2d(x) ** 2, 1) > 1)[None, :]),
]:
    try:
        b = BADS(
            f,
            np.array([0.3, 0.2]),
            -2 * np.ones(2),
            2 * np.ones(2),
            -1 * np.ones(2),
            np.ones(2),
            non_box_cons=nbc,
            options={"display": "off", "random_seed": 0, "max_fun_evals": 60},
        )
        r = b.optimize()
        print(label, "ran", r["x"], r["func_count"])
    except Exception as e:
        print(label, type(e).__name__, str(e)[:120])
