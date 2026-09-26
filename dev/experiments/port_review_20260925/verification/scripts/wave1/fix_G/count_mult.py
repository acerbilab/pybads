"""On the six runs of the fingerprint: statistics stored, and how many were
predicted by a posterior whose noise gpyreg multiplied (sn2_mult != 1)."""
import gpyreg
import numpy as np

import pybads
import pybads.bads.bads as bads_module
from pybads import BADS

print(pybads.__file__, gpyreg.__file__)
predictions, rows = [], []
orig_acq = bads_module.acq_fcn_lcb
orig_save = BADS._save_gp_stats_


def spy_acq(xi, fc, gp, *a, **k):
    out = orig_acq(xi, fc, gp, *a, **k)
    predictions.append(gp)
    return out


def spy_save(self, fval, ymu, ys):
    rows.append(predictions[-1].posteriors[0].sn2_mult or 1)
    return orig_save(self, fval, ymu, ys)


bads_module.acq_fcn_lcb = spy_acq
BADS._save_gp_stats_ = spy_save
g = np.random.default_rng(0)
f = lambda x: float(np.sum(np.atleast_2d(x) ** 2))
fn = lambda x: float(np.sum(np.atleast_2d(x) ** 2) + g.standard_normal())
for fun, noisy in [(f, False), (fn, True)]:
    for seed in range(3):
        rows.clear()
        o = {"display": "off", "max_fun_evals": 80, "random_seed": seed}
        if noisy:
            o["uncertainty_handling"] = True
        BADS(
            fun,
            np.ones(3) * 4,
            -100 * np.ones(3),
            100 * np.ones(3),
            -8 * np.ones(3),
            12 * np.ones(3),
            options=o,
        ).optimize()
        r = np.array(rows)
        print(
            "noisy" if noisy else "det  ",
            seed,
            "stats",
            len(r),
            "with sn2_mult != 1:",
            int(np.sum(r != 1)),
            "values",
            np.unique(r),
        )
