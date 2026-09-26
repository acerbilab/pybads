"""Wave 0's thin feasible region: |x1 - x2| <= 0.005 as non_box_cons."""
import sys
import traceback
import warnings

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
D = int(sys.argv[1]) if len(sys.argv) > 1 else 2
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
noisy = len(sys.argv) > 3 and sys.argv[3] == "noisy"


def nbc(x):
    x = np.atleast_2d(x)
    return np.abs(x[:, 0] - x[:, 1]) > 0.005


g = np.random.default_rng(seed)
if noisy:
    fun = lambda x: float(
        np.sum((np.ravel(x) - 1.0) ** 2) + g.standard_normal()
    )
else:
    fun = lambda x: float(np.sum((np.ravel(x) - 1.0) ** 2))
x0 = np.zeros(D)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    bads = BADS(
        fun,
        x0,
        np.full(D, -5.0),
        np.full(D, 5.0),
        np.full(D, -2.0),
        np.full(D, 2.0),
        non_box_cons=nbc,
        options={
            "random_seed": seed,
            "max_fun_evals": 100,
            "display": "off",
            **({"uncertainty_handling": True} if noisy else {}),
        },
    )
    try:
        res = bads.optimize()
        print(
            "finished: x",
            res["x"],
            "fval",
            res["fval"],
            "func_count",
            res["func_count"],
            res["message"],
        )
    except Exception as e:
        print("RAISED:", type(e).__name__, str(e)[:300])
        for fr in traceback.extract_tb(e.__traceback__)[-5:]:
            print("   ", fr.filename.split("/")[-1], fr.lineno, fr.name)
        fl = bads.function_logger
        print("evaluated:", fl.X_flag.sum(), "Y:", fl.Y[fl.X_flag].ravel())
