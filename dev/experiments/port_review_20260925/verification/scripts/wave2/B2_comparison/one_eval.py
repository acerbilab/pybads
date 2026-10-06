import logging

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
print(gpyreg.__file__)
logging.getLogger("BADS").setLevel(logging.WARNING)
D = 2
f = lambda x: float(np.sum(x**2))
for opts in [
    dict(max_fun_evals=1),
    dict(max_fun_evals=1, uncertainty_handling=False),
    dict(output_fcn=lambda x, s, st: st == "init"),
]:
    b = BADS(
        f,
        np.full(D, 0.5),
        np.full(D, -5.0),
        np.full(D, 5.0),
        np.full(D, -2.0),
        np.full(D, 2.0),
        options=dict(display="off", random_seed=0, **opts),
    )
    r = b.optimize()
    print(
        {k: opts[k] for k in opts if k != "output_fcn"},
        "output_fcn" in opts,
        "-> func_count",
        r["func_count"],
        "iterations",
        r["iterations"],
        "mesh_size",
        r["mesh_size"],
        "|",
        r["message"],
    )
