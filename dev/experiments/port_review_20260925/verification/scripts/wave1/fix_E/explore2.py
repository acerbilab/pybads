import numpy as np

from pybads import BADS

for D in (2, 3):
    bads = BADS(
        lambda x: float(np.sum(np.atleast_2d(x) ** 2)),
        np.ones(D) * 4,
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
        options={"display": "off", "random_seed": 3},
    )
    gp, _, _, _ = bads._init_optimization_()
    print(
        D,
        gp.X.shape,
        bads.optim_state["iter"],
        gp.get_hyperparameters(as_array=True),
    )
    print(gp.y.ravel())
