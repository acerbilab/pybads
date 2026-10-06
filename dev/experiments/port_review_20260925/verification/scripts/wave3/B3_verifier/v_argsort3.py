"""B3-K9: where in the LCB ranking the ties at es_search.py:190 fall."""
import sys

import numpy as np
import vhdr  # noqa

import pybads.search.es_search as es_mod
from pybads import BADS

out = []


class NP:
    def __getattr__(self, k):
        return getattr(np, k)

    def argsort(self, a, *args, **kw):
        fr = sys._getframe(1)
        line = fr.f_lineno
        d = np.argsort(a, *args, **kw)
        if line == 190:
            zs = a[d]
            eq = np.flatnonzero(zs[1:] == zs[:-1])
            if eq.size:
                vals, cnt = np.unique(zs[eq], return_counts=True)
                top = vals[np.argmax(cnt)]
                out.append(
                    (
                        eq.min() / len(zs),
                        eq.size,
                        len(zs),
                        top == zs.max(),
                        float(top),
                        float(np.median(zs)),
                    )
                )
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
for r in out[:8]:
    print(
        f"first tie at rank fraction {r[0]:.3f}; tied adjacent pairs {r[1]} of {r[2]}; most common tied value is the max z: {r[3]}; tied z {r[4]:.6g}, median z {r[5]:.6g}"
    )
print(
    f"calls with ties: {len(out)}; first tie in the top 1%: {sum(r[0] < 0.01 for r in out)}; median first-tie rank fraction {np.median([r[0] for r in out]):.3f}"
)
