"""What 1.1.0 (or the revision on PYTHONPATH) does with accelerate_mesh_steps
0, -1, 2.5, 3.0 and improvement_quantile 0, 1: the exception and when it
comes (the iteration and whether an earlier poll failed)."""
import logging
import traceback

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
logging.getLogger("BADS").setLevel(logging.ERROR)

polls = []
orig_poll = BADS._poll_step_


def poll(self, gp):
    mi = self.mesh_size_integer
    try:
        return orig_poll(self, gp)
    finally:
        polls.append((self.optim_state["iter"], mi, self.mesh_size_integer))


BADS._poll_step_ = poll


def run(opt, val, seed):
    polls.clear()
    b = BADS(
        lambda x: float(np.sum((np.ravel(x) - 0.3) ** 2)),
        np.array([1.5, -1.0]),
        -5 * np.ones(2),
        5 * np.ones(2),
        -3 * np.ones(2),
        3 * np.ones(2),
        options={
            "display": "off",
            opt: val,
            "max_fun_evals": 100,
            "random_seed": seed,
        },
    )
    try:
        r = b.optimize()
        moved = not np.allclose(r["x"], [1.5, -1.0])
        out = (
            f"completed, fval {r['fval']:.3g}, x {np.round(r['x'], 4)}, "
            f"func_count {r['func_count']}"
        )
    except Exception as e:  # noqa: BLE001
        tb = traceback.extract_tb(e.__traceback__)[-1]
        out = f"{type(e).__name__}: {str(e)[:70]} at line {tb.lineno}"
    # polls completed before (iter, mesh int before, after)
    failed = [p for p in polls if p[2] < p[1]]
    print(
        f"{opt}={val!r} seed {seed}: {out}; polls done {len(polls)}, "
        f"first failed poll iter "
        f"{failed[0][0] if failed else None}",
        flush=True,
    )


for seed in (0, 1):
    for v in (0, -1, 2.5, 3.0):
        run("accelerate_mesh_steps", v, seed)
    for v in (0, 1, 0.5):
        run("improvement_quantile", v, seed)
