"""B7 verifier, K5: a noisy run at D = 2 whose design leaves fewer
evaluations than noise_final_samples. What the run does at 0d866e8, over a
range of budgets, against MATLAB's arithmetic (design of 20, the same
'iter > 1' condition for the final samples)."""
import warnings

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)


class Noisy:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        x = np.atleast_2d(x)
        return float(np.sum((x - 0.1) ** 2)) + 0.5 * self.rng.normal()


def run(mfe, seed=0):
    f = Noisy(seed)
    D = 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = BADS(
            f,
            0.3 * np.ones((1, D)),
            -5 * np.ones((1, D)),
            5 * np.ones((1, D)),
            -2 * np.ones((1, D)),
            2 * np.ones((1, D)),
            options={
                "display": "off",
                "random_seed": seed,
                "max_fun_evals": mfe,
            },
        )
        r = b.optimize()
    return b, r, f


b, r, f = run(38)
print(
    f"max_fun_evals 38: level {b.optim_state['uncertainty_handling_level']}, "
    f"target calls {f.calls}, func_count {r['func_count']}, iterations "
    f"{r['iterations']}, noise_final_samples reserved "
    f"{b.options['noise_final_samples']}, max_fun_evals after the reserve "
    f"{b.options['max_fun_evals']}, yval_vec {b.optim_state['yval_vec']}, "
    f"fval {r['fval']:.4g}, fsd {r['fsd']:.4g}, message: "
    f"{r['message']!r}",
    flush=True,
)

print(
    "\nbudget | calls | iterations | reserved | final samples taken",
    flush=True,
)
for mfe in list(range(34, 57, 2)) + [60, 70]:
    b, r, f = run(mfe)
    taken = b.optim_state["ysd_vec"] is not None
    print(
        f"{mfe:3d} | {f.calls:3d} | {r['iterations']:2d} | "
        f"{b.options['noise_final_samples']:2d} | {taken} "
        f"(unused budget {mfe - f.calls})",
        flush=True,
    )

print(
    "\nMATLAB's arithmetic at D = 2 (design Ninit = min(max(20, D), "
    "MaxFunEvals - 1), noise test counted after the cap):",
    flush=True,
)
for mfe in (30, 32, 34, 36, 38, 40, 44, 48):
    ninit = min(min(max(20, 2), mfe), mfe - 1)
    fc = 2 + ninit
    reserve = min(10, mfe - fc)
    mfe2 = mfe - reserve
    print(
        f"MaxFunEvals {mfe}: after the design funccount {fc}, reserve "
        f"{reserve}, MaxFunEvals for the loop {mfe2}: the design alone "
        f"ends the run in its first iteration: {fc >= mfe2}",
        flush=True,
    )
