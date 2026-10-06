"""O-K2: the names that acq_fcn_lcb assigns and never reads (AST), and its
docstring's summary; update_hedge's docstring against the quantity it
updates (the gains g)."""
import ast
import inspect

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__, flush=True)
print("gpyreg:", gpyreg.__file__, flush=True)
from pybads.acquisition_functions.acq_fcn_lcb import acq_fcn_lcb
from pybads.search.search_hedge import ESSearchHedge

tree = ast.parse(inspect.getsource(acq_fcn_lcb))
stored = {
    n.id
    for n in ast.walk(tree)
    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
}
loaded = {
    n.id
    for n in ast.walk(tree)
    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
}
print("acq_fcn_lcb: assigned, never read:", sorted(stored - loaded))
print("acq_fcn_lcb summary:", inspect.getdoc(acq_fcn_lcb).splitlines()[:2])
print(
    "update_hedge summary:",
    inspect.getdoc(ESSearchHedge.update_hedge).splitlines()[0],
)
opts = {
    "hedge_gamma": 0.125,
    "hedge_beta": 1.0,
    "hedge_decay": 0.5,
    "n_search_iter": 2,
    "n_search": 4096,
}
h = ESSearchHedge(options_dict=opts, rng=np.random.default_rng(0))
h.g = np.array([10.0, 0.0])
h.chosen_hedge = np.array([1])
h.phat = np.array([np.inf, 0.125])
attrs0 = {
    k: (v.copy() if isinstance(v, np.ndarray) else v)
    for k, v in vars(h).items()
    if k != "rng"
}
h.update_hedge(np.zeros((1, 2)), 1.0, 0.0, 0.0, None, 0.5)
changed = [
    k
    for k, v in vars(h).items()
    if k != "rng"
    and not np.array_equal(
        np.asarray(v, dtype=object), np.asarray(attrs0[k], dtype=object)
    )
]
print(
    "update_hedge changed attributes:",
    changed,
    "g ->",
    h.g,
    "(0.5*10 + 0/inf, 0.5*0 + 1/0.125/0.5)",
)
