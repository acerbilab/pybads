"""Where a run with periodic variables spends its time.

Runs one configuration of dev/scripts/benchmark_targets.py at one seed,
with its periodic variables ("on") or as a bounded problem ("off"), with
timers around the stages of a run, gpyreg's kernel split by kind of call,
and gpyreg's periodic helpers. With --counterfactual, every call of the
kernel with periods is repeated on the same inputs with the periods
removed, timed apart from the run: what the kernel would cost there
without its periodic code. Run from the repository root:

    python -u dev/experiments/periodic_kernel_linux_20260929/profile_periodic.py \
        periodic_D3_homo 0 on --counterfactual --out OUT.json
"""
import argparse
import cProfile
import json
import os
import pstats
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.getcwd(), "dev", "scripts"))
import benchmark_targets as bt  # noqa: E402
import gpyreg  # noqa: E402
import gpyreg.covariance_functions as cf  # noqa: E402
import numpy as np  # noqa: E402
from gpyreg.gaussian_process import GP  # noqa: E402

import pybads  # noqa: E402
import pybads.bads.bads as bb  # noqa: E402
from pybads import BADS  # noqa: E402
from pybads.search.search_hedge import ESSearchHedge  # noqa: E402

TOT = defaultdict(float)
CNT = defaultdict(int)
SIZE = defaultdict(int)  # sum of N * M per kernel kind
STACK = []  # [name, child time]
CF = {"on": False}


def _enter(name):
    STACK.append([name, 0.0])
    return time.perf_counter()


def _leave(name, t0):
    el = time.perf_counter() - t0
    _, child = STACK.pop()
    TOT[name] += el - child
    CNT[name] += 1
    if STACK:
        STACK[-1][1] += el
    return el


def timed(name, fn):
    def w(*a, **k):
        t0 = _enter(name)
        try:
            return fn(*a, **k)
        finally:
            _leave(name, t0)

    w.__wrapped__ = fn
    return w


def _kind(X_star, compute_diag, compute_grad):
    if compute_grad:
        return "grad"
    if compute_diag:
        return "diag"
    return "cross" if X_star is not None else "self"


_orig_compute = cf.RationalQuadraticARD.compute


def kernel(self, hyp, X, X_star=None, compute_diag=False, compute_grad=False):
    kind = _kind(X_star, compute_diag, compute_grad)
    name = "kernel_" + kind
    N = X.shape[0]
    M = N if X_star is None else X_star.shape[0]
    SIZE[name] += N * (1 if compute_diag else M)
    t0 = _enter(name)
    try:
        out = _orig_compute(self, hyp, X, X_star, compute_diag, compute_grad)
    finally:
        _leave(name, t0)
    if CF["on"] and self.periods is not None:
        periods = self.periods
        self.periods = None
        t1 = _enter("cf_" + kind)
        try:
            _orig_compute(self, hyp, X, X_star, compute_diag, compute_grad)
        finally:
            _leave("cf_" + kind, t1)
            self.periods = periods
    return out


_orig_sq_dist = cf._scaled_sq_dist
_orig_sq_diff = cf._scaled_sq_diff


def sq_dist(*a):
    return timed("sq_dist", _orig_sq_dist)(*a)


def sq_diff(*a):
    parent = STACK[-1][0] if STACK else "?"
    return timed("sq_diff<-" + parent, _orig_sq_diff)(*a)


def install():
    cf.RationalQuadraticARD.compute = kernel
    cf._scaled_sq_dist = sq_dist
    cf._scaled_sq_diff = sq_diff
    GP.fit = timed("gp_fit", GP.fit)
    GP.predict = timed("gp_predict", GP.predict)
    GP.update = timed("gp_update", GP.update)
    ESSearchHedge.__call__ = timed("es_search", ESSearchHedge.__call__)
    bb.local_gp_fitting = timed("local_gp_fitting", bb.local_gp_fitting)
    bb.add_and_update_gp = timed("add_and_update_gp", bb.add_and_update_gp)
    BADS._re_evaluate_history_ = timed(
        "re_evaluate_history", BADS._re_evaluate_history_
    )
    BADS._search_step_ = timed("search_step", BADS._search_step_)
    BADS._poll_step_ = timed("poll_step", BADS._poll_step_)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("label")
    p.add_argument("seed", type=int)
    p.add_argument("arm", choices=["on", "off"])
    p.add_argument("--counterfactual", action="store_true")
    p.add_argument("--cprofile", help="write cProfile stats here")
    p.add_argument("--out", help="write the tallies here as JSON")
    a = p.parse_args()

    cfg = bt.find_config(a.label)
    prob = cfg.make(seed=a.seed, budget_scale=1.0)
    args, options = prob.bads_args()
    if a.arm == "off":
        options["periodic_vars"] = None
    if not a.cprofile:
        install()
    CF["on"] = a.counterfactual

    bads = BADS(*args, options=options)
    prof = cProfile.Profile() if a.cprofile else None
    t0 = time.perf_counter()
    if prof:
        prof.enable()
    res = bads.optimize()
    if prof:
        prof.disable()
    wall = time.perf_counter() - t0
    cf_time = sum(v for k, v in TOT.items() if k.startswith("cf_"))
    run = wall - cf_time
    fun_time = float(bads.function_logger.total_fun_eval_time)
    rec = {
        "label": a.label,
        "seed": a.seed,
        "arm": a.arm,
        "func_count": int(res["func_count"]),
        "wall_s": wall,
        "run_s": run,
        "fun_eval_s": fun_time,
        "fval": float(res["fval"]),
        "x": np.ravel(res["x"]).tolist(),
        "n_train_last": None,
        "tot": dict(TOT),
        "cnt": dict(CNT),
        "size": dict(SIZE),
        "gpyreg": gpyreg.__file__,
        "pybads": pybads.__file__,
        "threads": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }
    print(
        f"{a.label} seed {a.seed} {a.arm}: {rec['func_count']} evals, "
        f"run {run:.2f} s ({1000 * run / rec['func_count']:.1f} ms/eval), "
        f"cf {cf_time:.2f} s",
        flush=True,
    )
    for k in sorted(TOT, key=lambda k: -TOT[k]):
        print(
            f"  {k:28s} {TOT[k]:8.3f} s {100 * TOT[k] / run:6.1f} % "
            f"{CNT[k]:7d} calls",
            flush=True,
        )
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rec, f, indent=1)
    if prof:
        prof.dump_stats(a.cprofile)
        st = pstats.Stats(prof)
        st.sort_stats("tottime").print_stats(25)


if __name__ == "__main__":
    main()
