import numpy as np

import pybads
from pybads import BADS
from pybads.bads.gaussian_process_train import local_gp_fitting

print(pybads.__file__)
D = 3
for seed in (0, 3, 11):
    bads = BADS(
        lambda x: float(
            np.sum((np.array([1.0, 5.0, 30.0]) * np.ravel(x)) ** 2)
        ),
        np.array([1.0, -1.2, 0.8]),
        None,
        None,
        -2 * np.ones(D),
        2 * np.ones(D),
        options={"display": "off", "random_seed": seed},
    )
    gp, _, _, _ = bads._init_optimization_()
    print("plb", bads.optim_state["plb"], "pub", bads.optim_state["pub"])
    print("init poll_scale", gp.temporary_data.get("poll_scale"))
    gp, flag = local_gp_fitting(
        gp,
        bads.u,
        bads.function_logger,
        bads.options,
        bads.optim_state,
        bads.iteration_history,
        True,
        rng=bads.rng,
    )
    log_ls = gp.get_hyperparameters()[0]["covariance_log_lengthscale"]
    ll = np.exp(bads.options["gp_rescale_poll"] * (log_ls - np.mean(log_ls)))
    print(
        seed,
        "flag",
        flag,
        "ls",
        np.exp(log_ls),
        "ll",
        ll,
        "poll_scale",
        gp.temporary_data["poll_scale"],
        "msize",
        bads.optim_state["search_mesh_size"],
        "rescale",
        bads.options["gp_rescale_poll"],
    )
