import logging

import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)


class H(logging.Handler):
    msgs = []

    def emit(self, r):
        H.msgs.append(r.getMessage())


logging.getLogger("BADS").addHandler(H())
for D in (1, 2):
    for seed in (0, 1):
        H.msgs.clear()
        b = BADS(
            lambda x: float(np.sum((np.atleast_2d(x) + 1) ** 2)),
            np.full(D, 2.5),
            np.zeros(D),
            np.full(D, 5.0),
            np.full(D, 0.5),
            np.full(D, 4.5),
            options={"display": "off", "random_seed": seed},
        )
        r = b.optimize()
        w = [m for m in H.msgs if "No candidate left" in m]
        print(D, seed, "warnings", len(w), sorted(set(w))[:2])
