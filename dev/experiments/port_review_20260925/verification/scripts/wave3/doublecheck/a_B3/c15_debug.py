"""F1: the ES's returned points over a whole seeded run, with contraints_check
as at 0d866e8 and with MATLAB's round in its bins."""
import hdr  # noqa: F401
import numpy as np
from cc_variants import port_mround

import pybads.bads.bads as bads_module
import pybads.search.es_search as es_module
from pybads import BADS
from pybads.search.search_hedge import ESSearchHedge

port_cc = es_module.contraints_check
orig = ESSearchHedge.__call__
rec = []


def call(self, *a, **k):
    out = orig(self, *a, **k)
    rec.append(np.array(out[0]).copy())
    return out


ESSearchHedge.__call__ = call


def edge_sphere(x):
    return float(np.sum((np.atleast_2d(x) + 1.0) ** 2))


seqs = []
evals = []
for fn in (port_cc, port_mround):
    es_module.contraints_check = fn
    bads_module.contraints_check = fn
    rec.clear()
    b = BADS(
        edge_sphere,
        2.5 * np.ones(2),
        np.zeros(2),
        5 * np.ones(2),
        np.zeros(2),
        5 * np.ones(2),
        options={"display": "off", "max_fun_evals": 200, "random_seed": 0},
    )
    b.optimize()
    seqs.append([r for r in rec])
    fl = b.function_logger
    evals.append(fl.X[: fl.Xn + 1].copy())
a, m = seqs
print("ES calls", len(a), len(m))
for i, (p, q) in enumerate(zip(a, m)):
    if not np.array_equal(p, q):
        print(
            f"search {i}: port {p.tolist()} MATLAB's round {q.tolist()} diff/2^-20 {((np.array(q) - np.array(p)) / 2.0**-20).tolist()}"
        )
print("evaluated points identical:", np.array_equal(evals[0], evals[1]))
