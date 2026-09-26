"""How often the end-of-iteration choice moves in noisy runs of the test's
setup, with and without the injected failures of the re-estimate."""
import sys

import numpy as np

import pybads.bads.bads as bm
from pybads import BADS

D = 3
inject = sys.argv[1] == "1"
seeds = range(int(sys.argv[2])) if len(sys.argv) > 2 else [3]

for seed in seeds:
    rng = np.random.default_rng(0)
    fun = (
        lambda x: float(np.sum(np.atleast_2d(x) ** 2)) + rng.standard_normal()
    )
    stats = {"choices": 0, "moves": 0, "nanchoices": 0, "zmax": []}
    orig_eval = BADS._eval_improvement_
    tol = 1e-3

    def evaluate(self, f_base, f_new, s_base, s_new, q):
        z = orig_eval(self, f_base, f_new, s_base, s_new, q)
        if sys._getframe(1).f_code.co_name == "optimize" and np.ndim(f_new):
            stats["choices"] += 1
            zz = z[1:]
            if np.any(np.isnan(zz)):
                stats["nanchoices"] += 1
            m = np.nanmax(zz)
            stats["zmax"].append(round(float(m), 3))
            if m > tol:
                stats["moves"] += 1
        return z

    BADS._eval_improvement_ = evaluate
    b = BADS(
        fun,
        np.ones(D) * 4,
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
        options={
            "display": "off",
            "max_fun_evals": 150,
            "random_seed": seed,
            "uncertainty_handling": True,
            "noise_final_samples": 0,
        },
    )
    r = b.optimize()
    BADS._eval_improvement_ = orig_eval
    print(
        seed,
        {k: v for k, v in stats.items() if k != "zmax"},
        stats["zmax"][:30],
        r["func_count"],
    )
