import logging

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from pybads import BADS


class H(logging.Handler):
    def __init__(self):
        super().__init__()
        self.n = {}

    def emit(self, rec):
        self.n[rec.levelname] = self.n.get(rec.levelname, 0) + 1


logging.getLogger("BADS").propagate = False
for disp in ["iter", "notify", "final", "off", "full"]:
    h = H()
    lg = logging.getLogger("BADS")
    lg.handlers = [h]
    b = BADS(
        lambda x: float(np.sum(np.ravel(x) ** 2)),
        np.array([0.3, -0.2]),
        np.array([-2.0, -2.0]),
        np.array([2.0, 2.0]),
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
        options={"display": disp, "random_seed": 0, "max_fun_evals": 30},
    )
    b.optimize()
    print(
        disp,
        "-> logger level",
        logging.getLevelName(lg.level),
        "records:",
        h.n,
    )
