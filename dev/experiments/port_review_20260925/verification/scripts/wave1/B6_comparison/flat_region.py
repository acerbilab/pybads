"""A target with a flat minimum region: once every local training target is equal,
local_gp_fitting's output-scale prior is log(std(y)) = -inf."""
import sys
import traceback
import warnings

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from pybads import BADS

D = int(sys.argv[1])
seed = int(sys.argv[2])
fun = lambda x: float(max(0.0, np.sum(np.asarray(x) ** 2) - 1.0))
x0 = np.full(D, 0.3)
lb = np.full(D, -5.0)
ub = np.full(D, 5.0)
plb = np.full(D, -2.0)
pub = np.full(D, 2.0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    bads = BADS(
        fun,
        x0,
        lb,
        ub,
        plb,
        pub,
        options={"random_seed": seed, "max_fun_evals": 200, "display": "off"},
    )
    try:
        res = bads.optimize()
        print(
            "finished: fval",
            res["fval"],
            "func_count",
            res["func_count"],
            "msg",
            res["message"],
        )
    except Exception as e:
        print("RAISED:", type(e).__name__, str(e)[:300])
        tb = traceback.extract_tb(e.__traceback__)
        for fr in tb[-4:]:
            print("   ", fr.filename.split("/")[-1], fr.lineno, fr.name)
        print("func_count at failure:", bads.function_logger.func_count)
