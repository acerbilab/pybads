"""accelerate_mesh_steps = inf (and a large float) at the revision on
PYTHONPATH: does the run complete?"""
import logging

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
logging.getLogger("BADS").setLevel(logging.ERROR)
for v in (np.inf, 1e9, 3):
    try:
        b = BADS(
            lambda x: float(np.sum((np.ravel(x) - 0.3) ** 2)),
            np.array([1.5, -1.0]),
            -5 * np.ones(2),
            5 * np.ones(2),
            -3 * np.ones(2),
            3 * np.ones(2),
            options={
                "display": "off",
                "accelerate_mesh_steps": v,
                "max_fun_evals": 100,
                "random_seed": 0,
            },
        )
        r = b.optimize()
        print(
            f"{v!r}: completed, {r['func_count']} evaluations, fval {r['fval']:.3g}, "
            f"iterations {r['iterations']}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"{v!r}: {type(e).__name__}: {str(e)[:80]}", flush=True)
