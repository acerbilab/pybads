import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
import pybads.search.es_search as es
from pybads import BADS
from pybads.function_examples import rosenbrocks_fcn
from pybads.search.es_search import ESSearchELL

D = 3
b = BADS(
    rosenbrocks_fcn,
    np.zeros((1, D)),
    -20 * np.ones((1, D)),
    20 * np.ones((1, D)),
    -5 * np.ones((1, D)),
    5 * np.ones((1, D)),
    options={"random_seed": 0, "display": "off"},
)
b.options["fun_eval_start"] = 10
gp, *_ = b._init_optimization_()
orig = es.contraints_check
n = {"c": 0}


def cc(U, *a, **k):
    n["c"] += 1
    out = orig(U, *a, **k)
    return (
        out if n["c"] == 1 else out[:0]
    )  # the second generation loses every candidate


for patched in [False, True]:
    es.contraints_check = cc if patched else orig
    n["c"] = 0
    s = ESSearchELL(2048, 2048, b.options, rng=np.random.default_rng(5))
    us, z = s(
        b.u, None, None, b.function_logger, gp, b.optim_state, True, None
    )
    print(
        "second generation emptied" if patched else "unpatched",
        "-> returned shape",
        np.shape(us),
    )
