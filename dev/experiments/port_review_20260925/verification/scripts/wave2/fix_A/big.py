import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
b = BADS(
    lambda x: float(np.sum((np.asarray(x) + 1.4) ** 2)),
    None,
    np.array([-9.53e10]),
    np.array([9.53e10]),
    np.array([-2.06]),
    np.array([-0.74]),
    options={"display": "off", "random_seed": 0, "max_fun_evals": 40},
)
r = b.optimize()
print(r["x"], r["fval"], r["func_count"])
