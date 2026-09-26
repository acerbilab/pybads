import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
D = 3
log = []
orig = BADS._poll_step_


def poll(self, gp):
    it = self.optim_state["iter"]
    m0 = self.mesh_size_integer
    out = orig(self, gp)
    log.append((it, m0 - self.mesh_size_integer, float(self.fval)))
    return out


BADS._poll_step_ = poll
b = BADS(
    lambda x: float(np.sum(np.atleast_2d(x) ** 2)),
    np.zeros(D),
    -100 * np.ones(D),
    100 * np.ones(D),
    -8 * np.ones(D),
    8 * np.ones(D),
    options={"display": "off", "max_fun_evals": 200, "random_seed": 3},
)
r = b.optimize()
print(log)
print(r["iterations"], r["func_count"], r["message"], r["fval"], r["x"])
