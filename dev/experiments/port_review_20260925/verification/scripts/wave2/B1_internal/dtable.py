import os

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from pybads.bads.options import Options

p = os.path.dirname(pybads.bads.options.__file__) + "/option_configs/"
names = [
    "max_iter",
    "max_fun_evals",
    "tol_stall_iters",
    "fun_eval_start",
    "n_basis",
    "tol_poi",
    "mesh_overflow_warning",
    "search_n_try",
    "n_train_max",
    "min_refit_time",
    "stable_gp_sampling",
    "tol_skl",
    "hedge_decay",
    "min_fun_evals",
    "min_iter",
    "noise_shaping_threshold",
    "out_warp_thresh_base",
    "gp_length_prior_mean",
    "tol_noise",
    "hedge_beta",
]
rows = {}
for D in [1, 2, 6, 20]:
    o = Options(p + "basic_bads_options.ini", {"D": D})
    o.load_options_file(p + "advanced_bads_options.ini", {"D": D})
    for n in names:
        rows.setdefault(n, []).append(o[n])
for n, v in rows.items():
    print(
        f"{n:28s}",
        "  ".join(f"{float(x):.4g}" for x in v),
        type(v[0]).__name__,
    )
