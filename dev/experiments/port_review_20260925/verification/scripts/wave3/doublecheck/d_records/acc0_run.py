import gpyreg

import pybads

print("pybads", pybads.__file__, "gpyreg", gpyreg.__file__, flush=True)
"""accelerate_mesh_steps=0: does a run fail at its first failed poll? (wave 2's doublecheck, for wave 3's ledger)"""
import traceback

import numpy as np

from pybads import BADS

for seed in (0, 1):
    try:
        r = BADS(
            lambda x: float(np.sum(np.atleast_2d(x) ** 2)),
            np.array([1.0, -2.0]),
            np.full(2, -5.0),
            np.full(2, 5.0),
            np.full(2, -3.0),
            np.full(2, 3.0),
            options={
                "display": "off",
                "random_seed": seed,
                "accelerate_mesh_steps": 0,
                "max_fun_evals": 150,
            },
        ).optimize()
        print(seed, "completed", r["func_count"], r["fval"])
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(
            seed,
            type(e).__name__,
            str(e)[:120],
            "at",
            tb.filename.split("pybads/")[-1],
            tb.lineno,
        )
