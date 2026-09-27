import time

import numpy as np

import pybads
import pybads.search.es_search as es_mod
from pybads import BADS
from pybads.function_examples import rosenbrocks_fcn
from pybads.search.es_search import ESSearchELL, ESSearchWM

print(pybads.__file__)
t0 = time.time()
x0 = np.array([[0, 0, 0]])
lb = np.array([[-20, -20, -20]])
ub = -lb
plb = np.array([[-5, -5, -5]])
pub = -plb
bads = BADS(rosenbrocks_fcn, x0, lb, ub, plb, pub, options={"random_seed": 0})
bads.options["fun_eval_start"] = 10
gp, *_ = bads._init_optimization_()
print("init", time.time() - t0, gp.X.shape)
rec = []
orig = es_mod.acq_fcn_lcb


def lcb(u, *a, **k):
    out = orig(u, *a, **k)
    rec.append((u.copy(), np.ravel(out[0]).copy()))
    return out


es_mod.acq_fcn_lcb = lcb
calls = {"n": 0}


def cons(X):
    calls["n"] += 1
    return np.zeros(len(X)) if calls["n"] != 2 else np.ones(len(X))


for n_iter in (2, 3):
    rec.clear()
    calls["n"] = 0
    bads.options["n_search_iter"] = n_iter
    mu = int(bads.options["n_search"] / n_iter)
    es = ESSearchWM(mu, mu, bads.options, rng=np.random.default_rng(0))
    us, z = es(
        bads.u, lb, ub, bads.function_logger, gp, bads.optim_state, True, cons
    )
    print(n_iter, [len(r[0]) for r in rec], np.shape(us), z, es.scale)
print(time.time() - t0)
