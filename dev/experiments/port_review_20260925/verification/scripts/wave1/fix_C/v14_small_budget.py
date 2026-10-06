"""F13 (internal): the init_N schedule when the budget does not exceed the initial design."""
import logging
import warnings

import common  # noqa
import numpy as np

from pybads import BADS
from pybads.bads import gaussian_process_train as gpt

logging.getLogger("BADS").setLevel(logging.ERROR)
orig = gpt._get_gp_training_options
SEEN = []


def wrap(os_, ih, o, h, s, fl, second_fit=False):
    g = orig(os_, ih, o, h, s, fl, second_fit)
    SEEN.append((g["init_N"], os_["eff_starting_points"]))
    return g


gpt._get_gp_training_options = wrap
import pybads.bads.bads as bb

bb._get_gp_training_options = (
    wrap if hasattr(bb, "_get_gp_training_options") else None
)
warnings.simplefilter("ignore")
for D in (2, 3):
    for mfe in range(2, 9):
        SEEN.clear()
        try:
            r = BADS(
                lambda x: float(np.sum(np.ravel(x) ** 2)),
                0.5 * np.ones((1, D)),
                -5 * np.ones((1, D)),
                5 * np.ones((1, D)),
                -2 * np.ones((1, D)),
                2 * np.ones((1, D)),
                options={
                    "random_seed": 0,
                    "display": "off",
                    "max_fun_evals": mfe,
                },
            ).optimize()
            o = f"ok, func_count {r['func_count']}"
        except Exception as e:
            o = f"RAISED {type(e).__name__}: {str(e)[:60]}"
        print(
            f"D={D} max_fun_evals={mfe}: {o}; (init_N, eff_starting_points) seen {sorted(set(SEEN))}"
        )
