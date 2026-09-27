"""W4-5: no run reaches the level-2 merge of a repeated point (KD-B7-3),
also through W4-14's new path of the final samples."""
import gpyreg
import numpy as np

import pybads

print(pybads.__file__)
print(gpyreg.__file__)
from pybads import BADS
from pybads.function_logger import FunctionLogger

merges = []
orig = FunctionLogger._record


def rec(self, x_orig, x, fval_orig, fsd, t, record_duplicate_data=True):
    if (
        record_duplicate_data
        and fsd is not None
        and np.any(np.all(self.X == x, axis=1))
    ):
        merges.append(x.copy())
    return orig(
        self,
        x_orig,
        x,
        fval_orig,
        fsd,
        t,
        record_duplicate_data=record_duplicate_data,
    )


FunctionLogger._record = rec
D = 3
for opts in ({"max_fun_evals": 200}, {"max_iter": 1}, {"max_fun_evals": 40}):
    for seed in (0, 1):
        merges.clear()
        rng = np.random.default_rng(seed)

        def f(x):
            y = float(np.sum((np.ravel(x) - 0.2) ** 2))
            return y + 0.3 * rng.normal(), 0.3

        o = {
            "display": "off",
            "random_seed": seed,
            "specify_target_noise": True,
        }
        o.update(opts)
        b = BADS(
            f,
            np.zeros(D),
            -5 * np.ones(D),
            5 * np.ones(D),
            -2 * np.ones(D),
            2 * np.ones(D),
            options=o,
        )
        r = b.optimize()
        fl = b.function_logger
        print(
            opts,
            seed,
            "fc",
            r["func_count"],
            "rows",
            fl.Xn + 1,
            "merges",
            len(merges),
            "yval_vec",
            None if r["yval_vec"] is None else r["yval_vec"].shape,
        )
