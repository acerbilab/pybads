import sys
import time
import warnings

import numpy as np

sys.path.insert(0, "pybads/testing/bads")
from test_gaussian_process_train import _initialized_bads

from pybads.bads.gaussian_process_train import local_gp_fitting
from pybads.stats import get_hpd

for shift in [1e2, 1e3]:
    bads, gp = _initialized_bads()
    logger = bads.function_logger
    X = logger.X[logger.X_flag]
    Y = logger.Y[logger.X_flag]
    hpd_X, hpd_y, _, _ = get_hpd(X, Y, bads.options["hpd_frac"])
    old = gp.mean.get_bounds_info(hpd_X, hpd_y)
    print(
        "bounds held",
        gp.get_bounds()["mean_const"],
        "recommended LB",
        old["LB"],
        "y",
        Y.ravel(),
    )
    logger.Y[logger.X_flag] -= shift
    t = time.time()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        gp, ef = local_gp_fitting(
            gp,
            bads.u,
            logger,
            bads.options,
            bads.optim_state,
            bads.iteration_history,
            True,
            rng=bads.rng,
        )
    m = gp.get_hyperparameters()[0]["mean_const"]
    print(
        shift,
        "exit",
        ef,
        "mean",
        m,
        "prior",
        gp.get_priors()["mean_const"],
        "y",
        gp.y.ravel(),
        f"{time.time()-t:.2f}s",
        len(wl),
        [str(w.message)[:50] for w in wl][:3],
    )
