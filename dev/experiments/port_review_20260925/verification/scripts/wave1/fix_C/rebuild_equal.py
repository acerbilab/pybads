import numpy as np

from pybads.bads.gaussian_process_train import local_gp_fitting
from pybads.testing.bads.test_gaussian_process_train import _initialized_bads

for refit in (False, True):
    bads, gp = _initialized_bads()
    logger = bads.function_logger
    prev = gp.get_priors()["covariance_log_outputscale"]
    print("prev", prev)
    logger.Y[logger.X_flag] = 7.0
    try:
        gp, flag = local_gp_fitting(
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
            "flag",
            flag,
            gp.get_priors()["covariance_log_outputscale"],
            gp.get_priors()["mean_const"],
            gp.temporary_data.keys(),
        )
        print(gp.predict(np.zeros((1, 2))))
    except Exception as e:
        print(refit, "RAISED", type(e).__name__, str(e)[:100])
