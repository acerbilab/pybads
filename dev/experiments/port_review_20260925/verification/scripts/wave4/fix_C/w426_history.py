"""W4-26: does iteration_history hold the final estimate at the chosen
iterate, as MATLAB's iterList (bads.m:1159-1160)? Noisy runs at D = 3,
both kinds of noise, seeds 0-4, and a run that ends in its first
iteration."""
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__, flush=True)


def make(target_noise, seed):
    rng = np.random.default_rng(seed)

    def fun(x):
        y = float(np.sum(np.atleast_2d(x) ** 2))
        if target_noise:
            sd = 1.0 + 0.1 * np.sqrt(y)
            return y + sd * rng.standard_normal(), sd * np.exp(
                0.3 * rng.standard_normal()
            )
        return y + rng.standard_normal()

    return fun


for target_noise in (False, True):
    for seed, extra in [(s, {}) for s in range(5)] + [(0, {"max_iter": 1})]:
        done = {}
        opts = {
            "display": "off",
            "max_fun_evals": 100,
            "random_seed": seed,
            "uncertainty_handling": True,
            "specify_target_noise": target_noise,
            "output_fcn": lambda x, s, st: done.update(s=s)
            if st == "done"
            else None,
            **extra,
        }
        b = BADS(
            make(target_noise, seed),
            np.ones(3) * 4,
            -100 * np.ones(3),
            100 * np.ones(3),
            -8 * np.ones(3),
            12 * np.ones(3),
            options=opts,
        )
        r = b.optimize()
        h = b.iteration_history
        xs = [np.ravel(x) for x in h.get("x")]
        idx = [
            i for i, x in enumerate(xs) if np.array_equal(x, np.ravel(r["x"]))
        ]
        fv = h.get("fval").astype(float)
        fs = h.get("fsd").astype(float)
        ok = any(fv[i] == r["fval"] and fs[i] == r["fsd"] for i in idx)
        s = done["s"]
        print(
            f"target_noise={target_noise} seed={seed} {extra}: iterations "
            f"{r['iterations']}, returned iterate(s) {idx} of {len(xs)}, "
            f"history holds the final estimate there: {ok}; done's "
            f"fval/fsd equal the result's: "
            f"{s['fval'] == r['fval'] and s['fsd'] == r['fsd']}",
            flush=True,
        )
