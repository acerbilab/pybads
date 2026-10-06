import warnings

import numpy as np

warnings.simplefilter("ignore")
import pybads.bads.gaussian_process_train as gpt
from pybads import BADS

print(gpt.__file__)
orig = gpt._get_gp_training_options
for D in (2, 3):
    for mfe in (2, 3, 4, 5, 6, 8):
        seen = []

        def spy(*a, **k):
            g = orig(*a, **k)
            os_ = a[0] if a else k.get("optim_state")
            seen.append((g["init_N"], int(os_["eff_starting_points"])))
            return g

        gpt._get_gp_training_options = spy
        b = BADS(
            lambda x: float(np.sum(np.ravel(x) ** 2)),
            0.5 * np.ones(D),
            -5 * np.ones(D),
            5 * np.ones(D),
            -2 * np.ones(D),
            2 * np.ones(D),
            options={"display": "off", "max_fun_evals": mfe, "random_seed": 0},
        )
        r = b.optimize()
        print(
            D,
            mfe,
            "eff",
            b.optim_state["eff_starting_points"],
            "fc",
            r["func_count"],
            "Xn+1",
            b.function_logger.Xn + 1,
            "init_N",
            seen,
            "n_init",
            b.options["gp_train_n_init"],
            "final",
            b.options["gp_train_n_init_final"],
        )
gpt._get_gp_training_options = orig
