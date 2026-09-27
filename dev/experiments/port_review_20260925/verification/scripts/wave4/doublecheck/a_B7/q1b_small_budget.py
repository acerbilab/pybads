"""The body of test_get_gp_training_options_small_budget, over its 8 cases,
at the package on PYTHONPATH (run at e7bd01d and 46af65a)."""
import gpyreg
import numpy as np

import pybads

print(pybads.__file__)
print(gpyreg.__file__)
import pybads.bads.gaussian_process_train as gpt
from pybads import BADS

original = gpt._get_gp_training_options
for D in (2, 3):
    for mfe in (2, 3, 4, 5):
        seen = []

        def spy(*args, **kwargs):
            g = original(*args, **kwargs)
            seen.append(g["init_N"])
            return g

        gpt._get_gp_training_options = spy
        bads = BADS(
            lambda x: float(np.sum(np.ravel(x) ** 2)),
            0.5 * np.ones(D),
            -5 * np.ones(D),
            5 * np.ones(D),
            -2 * np.ones(D),
            2 * np.ones(D),
            options={"display": "off", "max_fun_evals": mfe, "random_seed": 0},
        )
        r = bads.optimize()
        ok = (
            r["func_count"] == mfe
            and bads.optim_state["eff_starting_points"] == mfe - 1
            and seen
            and all(n == bads.options["gp_train_n_init_final"] for n in seen)
            and np.isfinite(r["fval"])
        )
        print(
            f"D={D} mfe={mfe}: fc={r['func_count']} esp={bads.optim_state['eff_starting_points']} init_N={seen} -> {'pass' if ok else 'FAIL'}"
        )
