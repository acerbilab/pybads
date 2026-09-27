import logging

import vhdr  # noqa

print("root handlers before:", len(logging.getLogger().handlers))
import numpy as np

from pybads import BADS
from pybads.function_logger.constraints_check import contraints_check

b = BADS(
    lambda x: float(np.sum(np.atleast_2d(x) ** 2)),
    np.full(3, 1.5),
    np.full(3, -5.0),
    np.full(3, 5.0),
    np.full(3, -2.0),
    np.full(3, 2.0),
    options={"display": "off", "random_seed": 0},
)
print("root handlers after BADS():", len(logging.getLogger().handlers))
os_ = b.optim_state
print(
    "tol_mesh =",
    os_["tol_mesh"],
    "= 2^%d" % np.log2(os_["tol_mesh"]),
    "| lb_search shape",
    np.shape(os_["lb_search"]),
)
from pybads.bads.option_configs import get_pybads_option_dir_path
from pybads.search.es_search import ESSearchWM

ESSearchWM(4, 4, b.options)
print(
    "root handlers after ESSearchWM():",
    len(logging.getLogger().handlers),
    [type(h).__name__ for h in logging.getLogger().handlers],
)
from types import SimpleNamespace

fl = SimpleNamespace(X=np.full((1, 3), 9.0), X_max_idx=0)
print(
    "1-D point through contraints_check ->",
    contraints_check(
        np.array([0.1, 0.2, 0.3]),
        os_["lb_search"],
        os_["ub_search"],
        os_["tol_mesh"],
        fl,
        True,
    ).shape,
)
print(
    "empty (0,3) through contraints_check ->",
    contraints_check(
        np.empty((0, 3)),
        os_["lb_search"],
        os_["ub_search"],
        os_["tol_mesh"],
        fl,
        True,
    ).shape,
)
