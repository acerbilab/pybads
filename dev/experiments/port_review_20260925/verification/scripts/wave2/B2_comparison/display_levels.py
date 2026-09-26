import logging

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
print(gpyreg.__file__)


class Count(logging.Handler):
    def __init__(self):
        super().__init__()
        self.n = 0
        self.first = []

    def emit(self, rec):
        self.n += 1
        if len(self.first) < 3:
            self.first.append(rec.getMessage().strip()[:60])


D = 2
f = lambda x: float(np.sum(x**2))
for disp in ["off", "notify", "final", "iter"]:
    h = Count()
    lg = logging.getLogger("BADS")
    lg.addHandler(h)
    lg.propagate = False
    b = BADS(
        f,
        np.full(D, 1.0),
        np.full(D, -5.0),
        np.full(D, 5.0),
        np.full(D, -2.0),
        np.full(D, 2.0),
        options=dict(display=disp, random_seed=0, max_fun_evals=60),
    )
    b.optimize()
    lg.removeHandler(h)
    print(
        f"display={disp!r}: {h.n} messages on the BADS logger at level >= its level ({logging.getLevelName(lg.level)})"
    )
