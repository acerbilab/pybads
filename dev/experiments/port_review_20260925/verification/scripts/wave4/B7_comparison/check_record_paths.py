"""Count the paths FunctionLogger._record takes in short seeded runs at each
uncertainty level, and check the counts and the target's time."""
import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
from pybads import BADS
from pybads.function_logger import FunctionLogger

counts = {}
orig_record = FunctionLogger._record


def counting_record(
    self, x_orig, x, fval_orig, fsd, t, record_duplicate_data=True
):
    xn_before = self.Xn
    out = orig_record(
        self, x_orig, x, fval_orig, fsd, t, record_duplicate_data
    )
    if not record_duplicate_data:
        key = (
            "unrecorded (dup row)"
            if out[1] is not None
            else "unrecorded (no row)"
        )
    elif self.Xn == xn_before:
        key = "merged"
    else:
        key = "new row"
    counts[key] = counts.get(key, 0) + 1
    return out


FunctionLogger._record = counting_record

times = []


def ellipsoid(x):
    D = x.size
    return float(np.sum((10 ** (3 * np.arange(D) / max(D - 1, 1))) * x**2))


def run(level, seed, D=3, max_fun_evals=200):
    counts.clear()
    rng_noise = np.random.default_rng(seed)
    if level == 0:
        fun = ellipsoid
        opts = {}
    elif level == 1:
        fun = lambda x: ellipsoid(x) + rng_noise.normal()
        opts = {}
    else:

        def fun(x):
            f = ellipsoid(x)
            sd = 1 + np.sqrt(f)
            return f + sd * rng_noise.normal(), sd

        opts = {"specify_target_noise": True}
    opts.update(
        {"random_seed": seed, "max_fun_evals": max_fun_evals, "display": "off"}
    )
    x0 = np.full(D, 2.3)
    lb = np.full(D, -10.0)
    ub = np.full(D, 10.0)
    plb = np.full(D, -5.0)
    pub = np.full(D, 5.0)
    bads = BADS(fun, x0, lb, ub, plb, pub, options=opts)
    res = bads.optimize()
    fl = bads.function_logger
    rows = fl.Xn + 1
    t_rows = np.nansum(fl.fun_eval_time[:rows])
    print(
        f"level {level} seed {seed}: func_count {res['func_count']} "
        f"logger.func_count {fl.func_count} rows {rows} paths {dict(counts)} "
        f"n_evals[0] {fl.n_evals[0].item()} sum(n_evals) {fl.n_evals.sum()} "
        f"design rows {bads.optim_state['eff_starting_points']} "
        f"total_time {fl.total_fun_eval_time:.3e} sum_row_time {t_rows:.3e} "
        f"X rows allocated {fl.X.shape[0]} n_evals rows {fl.n_evals.shape[0]}",
        flush=True,
    )
    return bads


for level in (0, 1, 2):
    for seed in (1, 2, 3):
        run(level, seed)
