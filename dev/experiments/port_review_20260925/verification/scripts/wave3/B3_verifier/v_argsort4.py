"""B3-K9: one example of two distinct candidates with exactly equal LCB."""
import sys

import numpy as np
import vhdr  # noqa

import pybads.search.es_search as es_mod
from pybads import BADS

ex = []


class NP:
    def __getattr__(self, k):
        return getattr(np, k)

    def argsort(self, a, *args, **kw):
        fr = sys._getframe(1)
        d = np.argsort(a, *args, **kw)
        if fr.f_lineno == 190 and not ex:
            L = fr.f_locals
            C = L["us_candidates"]
            zs = a[d]
            for j in range(len(zs) - 1):
                if zs[j] == zs[j + 1] and not np.array_equal(
                    C[d[j]], C[d[j + 1]]
                ):
                    ex.append(
                        (
                            j,
                            C[d[j]],
                            C[d[j + 1]],
                            zs[j],
                            L["u"],
                            L["self"].search_mesh_size,
                        )
                    )
                    break
        return d


es_mod.np = NP()
D = 3
BADS(
    lambda x: float(np.sum(np.atleast_2d(x) ** 2)),
    np.full(D, 1.5),
    np.full(D, -5.0),
    np.full(D, 5.0),
    np.full(D, -2.0),
    np.full(D, 2.0),
    options={"display": "off", "random_seed": 1, "max_fun_evals": 100},
).optimize()
j, a, b, z, u, h = ex[0]
print(
    f"rank {j}: z = {z!r}\n  point A - u = {((a - u) / h).tolist()} (grid units)\n  point B - u = {((b - u) / h).tolist()} (grid units)"
)
