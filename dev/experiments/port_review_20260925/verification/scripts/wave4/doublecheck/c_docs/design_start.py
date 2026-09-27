"""Which initial design a run gets, by start and seed (the changelog's
"Initial design" entry), at whichever pybads PYTHONPATH selects. Records the
u0 that BADS passes to init_sobol and the design it returns; each run is cut
at its initial design (max_fun_evals just above it)."""

import hashlib

import gpyreg
import numpy as np

import pybads
import pybads.bads.bads as bads_module
from pybads import BADS

print(pybads.__file__, gpyreg.__file__, flush=True)

orig = bads_module.init_sobol
seen = []


def spy(u0, *args, **kwargs):
    u_init, n = orig(u0, *args, **kwargs)
    seen.append((np.array(u0, dtype=float).copy(), u_init.copy(), n))
    return u_init, n


bads_module.init_sobol = spy

D = 2
lb, ub = -10 * np.ones(D), 10 * np.ones(D)
plb, pub = -3 * np.ones(D), 3 * np.ones(D)
starts = {
    "interior [1, 1.5]": np.array([1.0, 1.5]),
    "interior [-2, 0.5]": np.array([-2.0, 0.5]),
    "on plb [-3, 0]": np.array([-3.0, 0.0]),
    "below plb [-4, 0]": np.array([-4.0, 0.0]),
    "on pub [3, 0]": np.array([3.0, 0.0]),
    "above pub [4, 0]": np.array([4.0, 0.0]),
    "above pub [9, 0]": np.array([9.0, 0.0]),
}
for seed in (0, 1):
    for label, x0 in starts.items():
        seen.clear()
        b = BADS(
            lambda x: float(np.sum(np.ravel(x) ** 2)),
            x0,
            lb,
            ub,
            plb,
            pub,
            options={
                "display": "off",
                "random_seed": seed,
                "max_fun_evals": 12,
                "uncertainty_handling": False,
            },
        )
        try:
            b.optimize()
        except Exception as e:  # the run itself does not matter here
            print("  run:", type(e).__name__, e)
        u0, u_init, n = seen[0]
        h = hashlib.sha1(np.round(u_init, 12).tobytes()).hexdigest()[:10]
        print(
            f"seed={seed} {label:22s} u0={np.round(u0, 4)} n_second_value={n} "
            f"design rows={u_init.shape[0]} hash={h} first={np.round(u_init[0], 4)}",
            flush=True,
        )
