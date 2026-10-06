"""B3-K10: a thin feasible band at D = 3; what the ES's 'Something went wrong' warning means."""
import logging

import numpy as np
import vhdr  # noqa

import pybads.search.es_search as es_mod
from pybads import BADS

gen_sizes = []  # per ES call: sizes of each generation after the check
cur = {"sizes": None}
orig_cc = es_mod.contraints_check
orig_call = es_mod.ESSearch.__call__


def cc(U, *a, **k):
    out = orig_cc(U, *a, **k)
    cur["sizes"].append((len(np.atleast_2d(U)), len(out)))
    return out


def call(self, *a, **k):
    cur["sizes"] = []
    out = orig_call(self, *a, **k)
    gen_sizes.append((cur["sizes"], np.shape(out[0])))
    return out


es_mod.contraints_check = cc
es_mod.ESSearch.__call__ = call
msgs = []


class H(logging.Handler):
    def emit(self, r):
        msgs.append((len(gen_sizes), r.getMessage()))


logging.getLogger("BADS").addHandler(H())

for eps in (0.05, 0.01):
    gen_sizes.clear()
    msgs.clear()
    D = 3
    cons = lambda X, e=eps: (
        np.abs(np.atleast_2d(X)[:, 0] - np.atleast_2d(X)[:, 1]) > e
    ).astype(float)
    res = BADS(
        lambda x: float(np.sum((np.atleast_2d(x) - 1.0) ** 2)),
        np.zeros(D),
        np.full(D, -5.0),
        np.full(D, 5.0),
        np.full(D, -2.0),
        np.full(D, 2.0),
        non_box_cons=cons,
        options={"display": "off", "random_seed": 0, "max_fun_evals": 150},
    ).optimize()
    es_warn = [m for m in msgs if "es_search" in m[1]]
    empty_ret = sum(1 for s, shp in gen_sizes if len(shp) == 2 and shp[0] == 0)
    which = [
        tuple(i for i, (a, b) in enumerate(s) if b == 0)
        for s, shp in gen_sizes
        if any(b == 0 for a, b in s)
    ]
    print(
        f"band |x1-x2|<={eps}: fval={res['fval']:.4g} evals={res['func_count']} ES calls={len(gen_sizes)} "
        f"ES warnings={len(es_warn)} ES calls returning an empty set={empty_ret}"
    )
    print("   emptied generations (0-based) per affected call:", which[:10])
    print(
        "   first 3 calls, (in, out) per generation:",
        [s for s, _ in gen_sizes[:3]],
    )
