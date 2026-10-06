"""What a noisy run stopped by output_fcn at "init" reports as fsd, at
uncertainty levels 1 and 2 (W4-30)."""

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__, gpyreg.__file__, flush=True)
for specify_target_noise in (False, True):
    for noise_size in (None, 2.5):
        rng = np.random.default_rng(0)

        def fun(x):
            f = float(np.sum(np.ravel(x) ** 2)) + 0.7 * rng.normal()
            return (f, 0.7) if specify_target_noise else f

        options = {
            "display": "off",
            "random_seed": 0,
            "uncertainty_handling": True,
            "specify_target_noise": specify_target_noise,
            "output_fcn": lambda x, s, st: st == "init",
        }
        if noise_size is not None and not specify_target_noise:
            options["noise_size"] = noise_size
        b = BADS(
            fun,
            np.ones(3),
            -10 * np.ones(3),
            10 * np.ones(3),
            -5 * np.ones(3),
            5 * np.ones(3),
            options=options,
        )
        r = b.optimize()
        print(
            f"specify_target_noise={specify_target_noise} noise_size={b.options['noise_size']}: "
            f"iterations={r['iterations']} func_count={r['func_count']} fval={r['fval']:.4f} "
            f"fsd={r['fsd']} yval_vec={r['yval_vec']} ysd_vec={r['ysd_vec']}",
            flush=True,
        )
