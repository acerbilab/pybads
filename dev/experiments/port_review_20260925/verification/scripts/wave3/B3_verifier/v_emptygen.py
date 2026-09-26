"""Internal F9 / comparison F6: the ES when a later generation loses every candidate."""
import logging

import numpy as np
import vhdr  # noqa
from capture import capture_states

import pybads.search.es_search as es_mod
from pybads.search.es_search import ESSearchWM

st = capture_states(D=3, seed=0, max_fun_evals=60, max_states=3)[-1]
rec = []
orig_lcb = es_mod.acq_fcn_lcb


def lcb(u, *a, **k):
    out = orig_lcb(u, *a, **k)
    rec.append((u.copy(), np.ravel(out[0]).copy()))
    return out


es_mod.acq_fcn_lcb = lcb


class Handler(logging.Handler):
    msgs = []

    def emit(self, r):
        Handler.msgs.append(r.getMessage())


logging.getLogger("BADS").addHandler(Handler())


def first_call_only():
    n = {"c": 0}

    def cons(X):
        n["c"] += 1
        return np.zeros(len(X)) if n["c"] == 1 else np.ones(len(X))

    return cons


es = ESSearchWM(2048, 2048, st["options"], rng=np.random.default_rng(5))
us, z = es(
    st["u"],
    None,
    None,
    st["func_logger"],
    st["gp"],
    st["optim_state"],
    True,
    first_call_only(),
)
u1, z1 = rec[0]
best1 = u1[np.argsort(z1, kind="stable")[0]]
print("generation sizes seen by LCB:", [len(r[0]) for r in rec])
print(
    "PyBADS ES returns shape",
    np.shape(us),
    "| MATLAB transcription returns the best of generation 1:",
    best1.round(5).tolist(),
)
print("warnings logged:", Handler.msgs)
# the same search without the constraint returns a point
Handler.msgs.clear()
rec.clear()
es = ESSearchWM(2048, 2048, st["options"], rng=np.random.default_rng(5))
us2, z2 = es(
    st["u"],
    None,
    None,
    st["func_logger"],
    st["gp"],
    st["optim_state"],
    True,
    None,
)
print(
    "without the constraint: shape", np.shape(us2), "warnings:", Handler.msgs
)
