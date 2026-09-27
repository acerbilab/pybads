"""The root logger (found while verifying): creating and running the ES
searches and the hedge adds no handler to the root logger; BADS() adds one
(basicConfig), and the BADS logger's level follows `display`."""
import logging

import hdr  # noqa: F401
import numpy as np

from pybads import BADS
from pybads.function_examples import rosenbrocks_fcn
from pybads.search.es_search import ESSearchELL, ESSearchWM
from pybads.search.search_hedge import ESSearchHedge

root = logging.getLogger()
for h in root.handlers[:]:
    root.removeHandler(h)

# Build a state first, then clear the handler BADS added
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
print(
    "handlers after BADS():",
    root.handlers,
    "; BADS logger level",
    logging.getLogger("BADS").level,
)
b.options["fun_eval_start"] = 10
gp, _, _, _ = b._init_optimization_()
for h in root.handlers[:]:
    root.removeHandler(h)
for cls in (ESSearchWM, ESSearchELL):
    s = cls(2048, 2048, b.options, rng=np.random.default_rng(0))
    s(b.u, None, None, b.function_logger, gp, b.optim_state, True, None)
h = ESSearchHedge(
    b.options["search_method"], b.options, rng=np.random.default_rng(0)
)
h(b.u, None, None, b.function_logger, gp, b.optim_state)
print(
    "handlers after creating and running ESSearchWM, ESSearchELL, ESSearchHedge:",
    root.handlers,
)
for disp in ("iter", "final", "full"):
    BADS(
        rosenbrocks_fcn,
        np.zeros((1, D)),
        -20 * np.ones((1, D)),
        20 * np.ones((1, D)),
        -5 * np.ones((1, D)),
        5 * np.ones((1, D)),
        options={"random_seed": 0, "display": disp},
    )
    print(
        f"display={disp!r}: BADS logger level {logging.getLogger('BADS').level}, root handlers {len(root.handlers)}"
    )
