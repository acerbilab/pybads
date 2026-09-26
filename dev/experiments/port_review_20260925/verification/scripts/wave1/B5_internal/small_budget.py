import warnings

import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__)
print("gpyreg", gpyreg.__file__)
import pybads.bads.gaussian_process_train as gpt
from pybads import BADS

seen = []
orig = gpt._get_gp_training_options


def wrap(*a, **k):
    out = orig(*a, **k)
    seen.append(out["init_N"])
    return out


gpt._get_gp_training_options = wrap
f = lambda x: float(np.sum(np.atleast_2d(x) ** 2))
for D in [2, 3]:
    for mfe in range(2, 12):
        seen.clear()
        b = BADS(
            f,
            np.full((1, D), 0.7),
            np.full((1, D), -5.0),
            np.full((1, D), 5.0),
            np.full((1, D), -2.0),
            np.full((1, D), 2.0),
            options={"random_seed": 0, "display": "off", "max_fun_evals": mfe},
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = b.optimize()
            msg = f"ok func_count {r['func_count']}"
        except Exception as e:
            msg = f"RAISED {type(e).__name__}: {str(e)[:90]}"
        print(
            f"D={D} max_fun_evals={mfe}: {msg}; eff_starting_points={b.optim_state.get('eff_starting_points')}, init_N seen {seen[:3]}"
        )
