"""Exact halves among the ES candidates per search call, by the log2 of the
search mesh, on 2-D and 3-D ellipsoids run to convergence (at most 200
evaluations)."""
import logging

import gpyreg
import numpy as np

import pybads
import pybads.search.es_search as es
from pybads import BADS
from pybads.search import grid_functions as gf

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
logging.getLogger("BADS").setLevel(logging.ERROR)
orig = gf.force_to_grid
rows = []


def counting(x, search_mesh_size, tol=None):
    t = search_mesh_size if tol is None else tol
    q = np.asarray(x) / t
    frac, _ = np.modf(q)
    rows.append(
        (
            int(round(np.log2(search_mesh_size))),
            q.size,
            int((np.abs(frac) == 0.5).sum()),
            float(np.log2(np.max(np.abs(q)) + 1e-300)),
        )
    )
    return orig(x, search_mesh_size, tol)


es.force_to_grid = counting
for D, seed in ((2, 0), (2, 1), (3, 0)):
    rows.clear()
    sc = np.arange(1, D + 1, dtype=float)
    b = BADS(
        lambda x: float(np.sum(sc * (np.atleast_2d(x) - 0.3) ** 2)),
        np.full(D, 2.0),
        lower_bounds=np.full(D, -5.0),
        upper_bounds=np.full(D, 5.0),
        plausible_lower_bounds=np.full(D, -4.0),
        plausible_upper_bounds=np.full(D, 4.0),
        options={"display": "off", "random_seed": seed, "max_fun_evals": 200},
    )
    r = b.optimize()
    agg = {}
    for lg, n, h, mq in rows:
        a = agg.setdefault(lg, [0, 0, 0, 0.0])
        a[0] += 1
        a[1] += n
        a[2] += h
        a[3] = max(a[3], mq)
    print(f"D={D} seed={seed}: status {r['message'][:40]!r}", flush=True)
    for lg in sorted(agg):
        if lg <= -30:
            c, n, h, mq = agg[lg]
            print(
                f"  log2 mesh {lg}: calls {c}, coordinates {n}, exact halves {h}, max log2|x/tol| {mq:.1f}",
                flush=True,
            )
