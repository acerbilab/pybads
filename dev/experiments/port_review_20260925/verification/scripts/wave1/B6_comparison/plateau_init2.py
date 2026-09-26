"""_gp_hyp on an initial design whose lowest 80% of targets are equal."""
import traceback
import warnings

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
import pybads.bads.bads as bb
from pybads import BADS

orig = bb.init_and_train_gp


def spy(hyp_dict, optim_state, fl, *a, **k):
    Y = fl.Y[fl.X_flag].ravel()
    print("initial targets:", np.round(Y, 3))
    return orig(hyp_dict, optim_state, fl, *a, **k)


bb.init_and_train_gp = spy
D = 3


def fun(x):
    x = np.asarray(x)
    r2 = np.sum((x - 1.9) ** 2)
    return float(r2) if r2 < 0.05 else 1e3


x0 = np.zeros(D)
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
        options={"random_seed": 0, "max_fun_evals": 100, "display": "off"},
    )
    try:
        res = bads.optimize()
        print("finished", res["fval"], res["func_count"])
    except Exception as e:
        print("RAISED:", type(e).__name__, str(e)[:250])
        for fr in traceback.extract_tb(e.__traceback__)[-4:]:
            print("   ", fr.filename.split("/")[-1], fr.lineno, fr.name)
