import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__, "gpyreg", gpyreg.__file__, flush=True)
from pybads import BADS
from pybads.function_logger import FunctionLogger

counts = {"calls": 0, "merge": 0, "nodup_norecord": 0, "dup_norecord": 0}
orig = FunctionLogger._record


def rec(self, x_orig, x, fval_orig, fsd, t, record_duplicate_data=True):
    counts["calls"] += 1
    dup = np.any(np.all(self.X == x, axis=1))
    if record_duplicate_data and fsd is not None and dup:
        counts["merge"] += 1
    if not record_duplicate_data:
        counts["dup_norecord" if dup else "nodup_norecord"] += 1
    return orig(self, x_orig, x, fval_orig, fsd, t, record_duplicate_data)


FunctionLogger._record = rec


def make_target(level, seed):
    r = np.random.default_rng(seed)

    def f(x):
        x = np.asarray(x)
        v = float(np.sum((x - 0.3) ** 2) + 0.5 * np.sum(np.abs(x)))
        sd = 0.5 + 0.2 * abs(x[0])
        y = v + sd * r.standard_normal()
        return (y, sd) if level == 2 else y

    return f


D = 2
lb = -5 * np.ones(D)
ub = 5 * np.ones(D)
plb = -2 * np.ones(D)
pub = 2 * np.ones(D)
for level, extra in [
    (2, {"specify_target_noise": True}),
    (1, {"uncertainty_handling": True}),
    (0, {}),
]:
    for seed in range(3):
        for k in counts:
            counts[k] = 0
        opts = {"random_seed": seed, "display": "off", "max_fun_evals": 200}
        opts.update(extra)
        tgt = (
            make_target(level, 100 + seed)
            if level > 0
            else (lambda x: float(np.sum((np.asarray(x) - 0.3) ** 2)))
        )
        b = BADS(tgt, np.array([1.2, -0.7]), lb, ub, plb, pub, options=opts)
        res = b.optimize()
        fl = b.function_logger
        X = fl.X[: fl.Xn + 1]
        n_dup_rows = X.shape[0] - np.unique(X, axis=0).shape[0]
        print(
            f"level {level} seed {seed}: func_count {fl.func_count}, rows {fl.Xn+1}, sum(n_evals) {int(fl.n_evals.sum())}, "
            f"rows with n_evals>1 {int(np.sum(fl.n_evals>1))} ({np.flatnonzero(fl.n_evals[:,0]>1).tolist()}), duplicate rows {n_dup_rows}, "
            f"merge path hits {counts['merge']}, not-recorded w/ dup {counts['dup_norecord']}, not-recorded w/o dup {counts['nodup_norecord']}",
            flush=True,
        )
