"""Does a hedge_decay above 1 make the gains overflow within a run, and what
does the run then do (1.1.0: IndexError, or a random choice)? Seeded runs of
at most 200 evaluations on a 2-D Rosenbrock function."""

import traceback

import gpyreg
import numpy as np

import pybads
from pybads import BADS
from pybads.search.search_hedge import ESSearchHedge

print(pybads.__file__, gpyreg.__file__, flush=True)

max_g = []
orig_update = ESSearchHedge.update_hedge


def update_hedge(self, *args, **kwargs):
    out = orig_update(self, *args, **kwargs)
    max_g.append(np.max(np.abs(self.g)))
    return out


ESSearchHedge.update_hedge = update_hedge


def rosen(x):
    x = np.ravel(x)
    return float(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)


for decay in (1e10, 1e30, 1e100):
    max_g.clear()
    try:
        b = BADS(
            rosen,
            np.array([-1.5, 2.0]),
            -5 * np.ones(2),
            5 * np.ones(2),
            -3 * np.ones(2),
            3 * np.ones(2),
            options={
                "display": "off",
                "random_seed": 0,
                "max_fun_evals": 200,
                "hedge_decay": decay,
            },
        )
        r = b.optimize()
        res = f"completed func_count={r['func_count']} fval={r['fval']:.3g}"
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        res = f"{type(e).__name__}: {e} [{tb.filename.split('/pybads/')[-1]}:{tb.lineno}]"
    print(
        f"hedge_decay={decay:g}: {res}; hedge updates={len(max_g)}, "
        f"largest |g| {max(max_g) if max_g else None}, "
        f"first inf at update {next((i for i, g in enumerate(max_g) if not np.isfinite(g)), None)}",
        flush=True,
    )
