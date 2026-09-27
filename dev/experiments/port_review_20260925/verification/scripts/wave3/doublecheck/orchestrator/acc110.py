import traceback

import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
from pybads import BADS

f = lambda x: float(np.sum(np.asarray(x) ** 2))
for v in [0, -1, 2.5]:
    try:
        r = BADS(
            f,
            np.ones(2) * 3,
            -5 * np.ones(2),
            5 * np.ones(2),
            options={
                "display": "off",
                "max_fun_evals": 150,
                "random_seed": 0,
                "accelerate_mesh_steps": v,
            },
        ).optimize()
        print(v, "completed", r["func_count"])
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(v, type(e).__name__, str(e)[:80], tb.lineno)
