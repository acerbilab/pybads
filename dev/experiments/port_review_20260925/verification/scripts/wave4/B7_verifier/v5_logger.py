"""B7 verifier: the function logger in runs at 0d866e8.
(a) K8: which evaluations count in total_fun_eval_time (a fake timer that
    gives 1 s per call), levels 0, 1 and 2.
(b) K7 / F6: row 0 after the noise test (n_evals, fun_eval_time with a timer
    that gives 1 s, 2 s, ...), and n_eff against eff_starting_points, init_N.
(c) K9 / F5 / F3: which calls repeat a logged point, by path, levels 0-2."""
import warnings

import gpyreg
import numpy as np

import pybads
import pybads.function_logger.function_logger as fl_module
from pybads import BADS
from pybads.function_logger import FunctionLogger

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)


class CountingTimer:
    """Each timed call lasts 1 s, 2 s, 3 s, ... in turn."""

    n = 0

    def start_timer(self, name):
        pass

    def stop_timer(self, name):
        CountingTimer.n += 1

    def get_duration(self, name):
        return float(CountingTimer.n) if MODE == "count" else 1.0


fl_module.Timer = CountingTimer
MODE = "one"

stats = {}
orig_record = FunctionLogger._record


def spy_record(
    self, x_orig, x, fval_orig, fsd, fun_eval_time, record_duplicate_data=True
):
    rows = self.X[: self.Xn + 1] if self.Xn >= 0 else np.zeros((0, self.D))
    repeat = bool(np.any(np.all(rows == x, axis=1)))
    if not record_duplicate_data:
        key = "unrecorded (repeat)" if repeat else "unrecorded (new point)"
    elif repeat and fsd is not None:
        key = "merge (level-2 repeat)"
    elif repeat:
        key = "recorded repeat, new row"
    else:
        key = "new row"
    stats[key] = stats.get(key, 0) + 1
    return orig_record(
        self,
        x_orig,
        x,
        fval_orig,
        fsd,
        fun_eval_time,
        record_duplicate_data=record_duplicate_data,
    )


FunctionLogger._record = spy_record


def quad(x):
    x = np.atleast_2d(x)
    return float(np.sum((x - 0.1) ** 2 * np.arange(1, x.size + 1)))


class Noisy:
    def __init__(self, seed, level2=False):
        self.rng = np.random.default_rng(seed)
        self.level2 = level2

    def __call__(self, x):
        x = np.atleast_2d(x)
        sd = 0.2 + 0.1 * float(np.sum(np.abs(x)))
        y = quad(x) + sd * self.rng.normal()
        return (y, sd) if self.level2 else y


def run(fun, D, seed, mfe, extra=None):
    opts = {"display": "off", "random_seed": seed, "max_fun_evals": mfe}
    opts.update(extra or {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = BADS(
            fun,
            0.4 * np.ones((1, D)),
            -5 * np.ones((1, D)),
            5 * np.ones((1, D)),
            -2 * np.ones((1, D)),
            2 * np.ones((1, D)),
            options=opts,
        )
        r = b.optimize()
    return b, r


print("\n(a) total_fun_eval_time with 1 s per call", flush=True)
for name, fun, extra in [
    ("level 0 (noise test)", quad, None),
    ("level 1 (noise test, final samples)", Noisy(1), None),
    (
        "level 2 (final samples)",
        Noisy(2, True),
        {"specify_target_noise": True},
    ),
    (
        "level 1 with uncertainty_handling=True (no noise test)",
        Noisy(3),
        {"uncertainty_handling": True},
    ),
]:
    stats.clear()
    b, r = run(fun, 2, 0, 100, extra)
    lg = b.function_logger
    print(
        f"{name}: level {b.optim_state['uncertainty_handling_level']}, "
        f"func_count {lg.func_count}, rows {lg.Xn + 1}, total target time "
        f"{lg.total_fun_eval_time:.0f} s, calls by path {stats}, "
        f"sum n_evals {np.sum(lg.n_evals[lg.X_flag]):.0f}, n_evals>1 at rows "
        f"{np.flatnonzero(lg.n_evals[:, 0] > 1).tolist()}",
        flush=True,
    )

print(
    "\n(b) row 0 after the noise test, and init_N at the first fit", flush=True
)
MODE = "count"
CountingTimer.n = 0
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    D = 2
    b = BADS(
        quad,
        0.4 * np.ones((1, D)),
        -5 * np.ones((1, D)),
        5 * np.ones((1, D)),
        -2 * np.ones((1, D)),
        2 * np.ones((1, D)),
        options={"display": "off", "random_seed": 0},
    )
    b.logging_action = []
    b._init_mesh_()
lg = b.function_logger
print(
    f"n_evals[0] {lg.n_evals[0, 0]:.0f}, fun_eval_time[0] "
    f"{lg.fun_eval_time[0, 0]} (calls timed 1 s and 2 s), "
    f"fun_eval_time[1] {lg.fun_eval_time[1, 0]}, total target time "
    f"{lg.total_fun_eval_time} (rows 0..{lg.Xn}: "
    f"{np.nansum(np.arange(1, lg.func_count + 1)) - 2} expected without "
    f"the noise test)",
    flush=True,
)
opts = b.options
n_eff = np.sum(lg.n_evals[lg.X_flag])
eff = b.optim_state["eff_starting_points"]
a = -(opts["gp_train_n_init"] - opts["gp_train_n_init_final"])
bb, c, d = -3 * a, 3 * a, opts["gp_train_n_init"]
n_budget = min(opts["max_fun_evals"], opts["n_train_max"]) - eff
f = lambda x_: a * x_**3 + bb * x_**2 + c * x_ + d  # noqa: E731
for extra in (0, 10):
    x_now = min(max((n_eff + extra - eff) / n_budget, 0), 1)
    x_rows = min(max((lg.Xn + 1 + extra - eff) / n_budget, 0), 1)
    print(
        f"after {extra} more evaluations: n_eff {n_eff + extra:.0f}, "
        f"eff_starting_points {eff}, n_budget {n_budget}: init_N "
        f"{max(round(f(x_now)), opts['gp_train_n_init_final'])} "
        f"(counting rows only: "
        f"{max(round(f(x_rows)), opts['gp_train_n_init_final'])})",
        flush=True,
    )
MODE = "one"

print(
    "\n(c) calls that repeat a logged point, by path, D = 3, 200 evaluations",
    flush=True,
)
for name, mk, extra in [
    ("level 0", lambda s: quad, None),
    ("level 1", lambda s: Noisy(10 + s), None),
    ("level 2", lambda s: Noisy(20 + s, True), {"specify_target_noise": True}),
]:
    for seed in range(3):
        stats.clear()
        b, r = run(mk(seed), 3, seed, 200, extra)
        lg = b.function_logger
        X = lg.X[: lg.Xn + 1]
        n_unique = np.unique(X, axis=0).shape[0]
        print(
            f"{name} seed {seed}: func_count {lg.func_count}, rows "
            f"{lg.Xn + 1}, distinct rows {n_unique}, calls by path {stats}",
            flush=True,
        )
