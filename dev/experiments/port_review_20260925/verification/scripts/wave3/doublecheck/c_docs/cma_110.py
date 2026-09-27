"""Call ESSearchCMA of 1.1.0 in place of ESSearchWM, and ask the hedge for
'ES-cma+'."""
import logging
import traceback

import gpyreg
import numpy as np

import pybads
import pybads.search.search_hedge as sh
from pybads import BADS
from pybads.search import ESSearchCMA

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
logging.getLogger("BADS").setLevel(logging.ERROR)
sh.ESSearchWM = ESSearchCMA


def f(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


opts = {
    "display": "off",
    "random_seed": 0,
    "max_fun_evals": 60,
    "search_method": [("ES-wcm", 1)],
}
b = BADS(
    f,
    np.array([2.0, 2.0]),
    lower_bounds=-5 * np.ones(2),
    upper_bounds=5 * np.ones(2),
    plausible_lower_bounds=-4 * np.ones(2),
    plausible_upper_bounds=4 * np.ones(2),
    options=opts,
)
try:
    b.optimize()
    print("ESSearchCMA ran", flush=True)
except Exception as e:
    tb = traceback.extract_tb(e.__traceback__)[-1]
    print(
        f"ESSearchCMA: {type(e).__name__}: {e} at {tb.name}:{tb.lineno}",
        flush=True,
    )

opts["search_method"] = [("ES-cma+", 1)]
b = BADS(
    f,
    np.array([2.0, 2.0]),
    lower_bounds=-5 * np.ones(2),
    upper_bounds=5 * np.ones(2),
    plausible_lower_bounds=-4 * np.ones(2),
    plausible_upper_bounds=4 * np.ones(2),
    options=opts,
)
try:
    b.optimize()
    print("ES-cma+ ran", flush=True)
except Exception as e:
    print(f"search_method ES-cma+: {type(e).__name__}: {e}", flush=True)
