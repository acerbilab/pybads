"""How often the arguments of force_to_grid lie exactly halfway between two
grid points, in the ES search of a 6-D run (at most 200 evaluations)."""
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
stats = []
orig = gf.force_to_grid


def counting(x, search_mesh_size, tol=None):
    t = search_mesh_size if tol is None else tol
    frac, _ = np.modf(np.asarray(x) / t)
    halves = np.abs(frac) == 0.5
    stats.append(
        (
            np.log2(search_mesh_size),
            halves.size,
            int(halves.sum()),
            int(np.any(halves, axis=-1).sum())
            if np.ndim(x) == 2
            else int(halves.any()),
            np.shape(x)[0] if np.ndim(x) == 2 else 1,
        )
    )
    return orig(x, search_mesh_size, tol)


es.force_to_grid = counting
D = 6
scales = np.array([1, 2, 4, 8, 16, 32.0])
b = BADS(
    lambda x: float(np.sum(scales * (np.atleast_2d(x) - 0.3) ** 2)),
    np.full(D, 2.0),
    lower_bounds=np.full(D, -5.0),
    upper_bounds=np.full(D, 5.0),
    plausible_lower_bounds=np.full(D, -4.0),
    plausible_upper_bounds=np.full(D, 4.0),
    options={"display": "off", "random_seed": 3, "max_fun_evals": 200},
)
b.optimize()
s = np.array(stats)
print("ES calls of force_to_grid:", len(s), flush=True)
for lo, hi in ((-100, -30), (-30, -20), (-20, 0)):
    m = (s[:, 0] >= lo) & (s[:, 0] < hi)
    if m.any():
        print(
            f"log2 search mesh in [{lo},{hi}): calls {m.sum()}, coordinates {int(s[m,1].sum())}, "
            f"exact halves {int(s[m,2].sum())}, candidates with a half {int(s[m,3].sum())} of {int(s[m,4].sum())}",
            flush=True,
        )
