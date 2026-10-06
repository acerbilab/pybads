import sys
import warnings

import numpy as np

sys.path.insert(0, "pybads/testing/bads")
from test_gaussian_process_train import _initialized_bads

from pybads.bads.gaussian_process_train import local_gp_fitting

for refit in [False, True]:
    bads, gp = _initialized_bads()
    prev = gp.get_priors()["covariance_log_outputscale"]
    logger = bads.function_logger
    print("rows", np.flatnonzero(logger.X_flag), logger.X_max_idx)
    logger.X_flag[1:] = False
    logger.X_max_idx = 0
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        gp, ef = local_gp_fitting(
            gp,
            bads.u,
            logger,
            bads.options,
            bads.optim_state,
            bads.iteration_history,
            refit,
            rng=bads.rng,
        )
    print(
        refit,
        gp.y.shape,
        ef,
        prev[1][0],
        gp.get_priors()["covariance_log_outputscale"][1][0],
    )
    for w in wl:
        print("  ", w.category.__name__, w.message, w.filename, w.lineno)
# ddof test probe
bads, gp = _initialized_bads()
gp, _ = local_gp_fitting(
    gp,
    bads.u,
    bads.function_logger,
    bads.options,
    bads.optim_state,
    bads.iteration_history,
    False,
    rng=bads.rng,
)
print(
    "N",
    gp.y.size,
    "centre",
    gp.get_priors()["covariance_log_outputscale"][1][0],
    "ddof0",
    np.log(np.std(gp.y)),
    "ddof1",
    np.log(np.std(gp.y, ddof=1)),
)
