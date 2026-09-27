"""F1, consequence: the same runs as v_f1_selfmove.py (flat D=3, levels 1
and 2, seeds 3-5), as is and with a self-moving uncertain poll left unmarked;
compare every evaluated point and value bit for bit, and the wall time."""
import sys
import time

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__, flush=True)
print("gpyreg:", gpyreg.__file__, flush=True)
from pybads import BADS

orig_poll = BADS._poll_step_
orig_upd = BADS._update_incumbent_
FIX = {"on": False, "same": None}


def upd(self, u, y, f, s):
    if sys._getframe(1).f_code.co_name == "_poll_step_":
        FIX["same"] = (
            np.array_equal(np.ravel(u), np.ravel(self.u)) and f == self.fval
        )
    return orig_upd(self, u, y, f, s)


def poll(self, gp):
    FIX["same"] = None
    out = orig_poll(self, gp)
    if FIX["on"] and FIX["same"]:
        self.poll_moved = False
    return out


BADS._update_incumbent_ = upd
BADS._poll_step_ = poll


def mk(seed, level):
    rng = np.random.default_rng((10_000 if level == 1 else 30_000) + seed)
    if level == 1:
        return lambda x: float(0.01 * np.sum(np.asarray(x) ** 2)) + float(
            rng.standard_normal()
        )
    return lambda x: (
        float(0.02 * np.sum(np.asarray(x) ** 2))
        + float(rng.standard_normal()),
        1.0,
    )


for level in [1, 2]:
    for seed in [3, 4] if level == 1 else [5]:
        out = {}
        for fix in [False, True]:
            FIX["on"] = fix
            opts = {
                "display": "off",
                "random_seed": seed,
                "max_fun_evals": 200,
                "stobads": True,
                "opp_stobads": True,
                "uncertainty_handling": True,
            }
            if level == 2:
                opts["specify_target_noise"] = True
            b = BADS(
                mk(seed, level),
                3.0 * np.ones(3),
                -50 * np.ones(3),
                50 * np.ones(3),
                -10 * np.ones(3),
                10 * np.ones(3),
                options=opts,
            )
            t0 = time.perf_counter()
            r = b.optimize()
            dt = time.perf_counter() - t0
            fl = b.function_logger
            n = fl.func_count
            out[fix] = (
                fl.X[:n].copy(),
                fl.Y[:n].copy(),
                r["x"].copy(),
                r["fval"],
                dt,
            )
        X0, Y0, x0, f0, t0_ = out[False]
        X1, Y1, x1, f1, t1_ = out[True]
        same = (
            X0.shape == X1.shape
            and np.array_equal(X0, X1, equal_nan=True)
            and np.array_equal(Y0, Y1, equal_nan=True)
        )
        nan_rows = int(np.sum(np.any(np.isnan(X0), axis=1)))
        if X0.shape == X1.shape:
            dif = np.flatnonzero(
                np.any(~((X0 == X1) | (np.isnan(X0) & np.isnan(X1))), axis=1)
            )
            first = int(dif[0]) if dif.size else None
            maxd = float(np.nanmax(np.abs(X0 - X1))) if X0.size else 0.0
        else:
            first, maxd = "shape", float("nan")
        print(
            f"    rows {X0.shape[0]} vs {X1.shape[0]}, NaN rows {nan_rows}, first differing row {first}, max |dX| {maxd:.3g}, max |dY| {float(np.nanmax(np.abs(Y0 - Y1))) if Y0.shape == Y1.shape else float('nan'):.3g}",
            flush=True,
        )
        print(
            f"level {level} seed {seed}: evaluated points identical: {same}; "
            f"max |dx| of result {np.max(np.abs(x0 - x1)):.3g}, |dfval| {abs(f0 - f1):.3g}; "
            f"time as is {t0_:.1f}s, unmarked {t1_:.1f}s",
            flush=True,
        )
