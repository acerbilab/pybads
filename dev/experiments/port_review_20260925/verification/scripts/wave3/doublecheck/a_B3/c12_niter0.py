"""'Found while fixing' (C): what n_search_iter = 0 does, in a run and in a
directly built ES search."""
import traceback

import hdr  # noqa: F401
import numpy as np

from pybads import BADS
from pybads.function_examples import rosenbrocks_fcn
from pybads.search.es_search import ESSearchWM

D = 2
for n_iter in (0, 0.5):
    try:
        r = BADS(
            rosenbrocks_fcn,
            np.zeros(D),
            -5 * np.ones(D),
            5 * np.ones(D),
            -2 * np.ones(D),
            2 * np.ones(D),
            options={
                "random_seed": 0,
                "display": "off",
                "max_fun_evals": 40,
                "n_search_iter": n_iter,
            },
        ).optimize()
        print(f"n_search_iter={n_iter}: run completes, fval {r['fval']:.3g}")
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(
            f"n_search_iter={n_iter}: {type(e).__name__}: {e} at {tb.filename.split('pybads-review/')[-1]}:{tb.lineno}"
        )

b = BADS(
    rosenbrocks_fcn,
    np.zeros(D),
    -5 * np.ones(D),
    5 * np.ones(D),
    -2 * np.ones(D),
    2 * np.ones(D),
    options={"random_seed": 0, "display": "off", "n_search_iter": 0},
)
b.options["fun_eval_start"] = 10
gp, _, _, _ = b._init_optimization_()
s = ESSearchWM(2048, 2048, b.options, rng=np.random.default_rng(0))
us, z = s(b.u, None, None, b.function_logger, gp, b.optim_state, True, None)
print(
    "direct ESSearchWM with n_search_iter=0 returns shapes",
    np.shape(us),
    np.shape(z),
)
