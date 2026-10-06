"""Run one benchmark configuration over seeds, one run at a time, with the
PyBADS on PYTHONPATH (an extracted revision), through
dev/scripts/benchmark_targets.py's definition of the configuration.

usage: runcfg.py LABEL MODE SEEDS OUT.jsonl
MODE: asis | force:<k> (the design's scrambling seed forced to k, without
the draw from the run's generator) | forcedraw:<k> (forced to k after the
draw, W4-1's later draws kept). SEEDS: a,b or a-b."""
import importlib.util
import json
import sys
import time

import gpyreg
import numpy as np

import pybads  # from PYTHONPATH, before benchmark_targets puts its root first
import pybads.init_functions  # noqa: F401

print(pybads.__file__, gpyreg.__file__, flush=True)
spec = importlib.util.spec_from_file_location(
    "benchmark_targets",
    "/home/user/pybads-review/dev/scripts/benchmark_targets.py",
)
bt = importlib.util.module_from_spec(spec)
sys.modules["benchmark_targets"] = bt
spec.loader.exec_module(bt)
from pybads import BADS  # noqa: E402

assert pybads.__file__ == sys.modules["pybads"].__file__

label, mode, seeds_arg, out = sys.argv[1:5]
if "-" in seeds_arg:
    a, b = map(int, seeds_arg.split("-"))
    seeds = list(range(a, b + 1))
else:
    seeds = [int(s) for s in seeds_arg.split(",")]

mod = sys.modules["pybads.init_functions.init_sobol"]
sobol_seeds = []
real_sobol = mod.Sobol


def sobol_spy(d, seed=None, **kw):
    sobol_seeds.append(int(seed))
    return real_sobol(d, seed=seed, **kw)


mod.Sobol = sobol_spy
if mode.startswith("force"):
    k = int(mode.split(":")[1])
    draw = mode.startswith("forcedraw")
    real_get_rng = mod.get_rng

    class _Forced:
        def __init__(self, rng):
            self.rng = rng

        def integers(self, *a, **kw):
            if draw:
                real_get_rng(self.rng).integers(*a, **kw)
            return k

    mod.get_rng = lambda rng=None: _Forced(rng)

cfg = bt.find_config(label)
with open(out, "a") as fh:
    for s in seeds:
        sobol_seeds.clear()
        prob = cfg.make(seed=s)
        args, options = prob.bads_args()
        t0 = time.perf_counter()
        res = BADS(*args, options=options).optimize()
        x = np.asarray(res["x"], dtype=float).ravel()
        rec = dict(
            label=label,
            mode=mode,
            seed=s,
            sobol_seeds=list(sobol_seeds),
            x=x.tolist(),
            fval=float(np.asarray(res["fval"]).item()),
            fsd=float(np.asarray(res["fsd"]).item()),
            true_error=float(prob.f_true(x) - prob.f_min),
            func_count=int(res["func_count"]),
            iterations=int(res["iterations"]),
            message=str(res["message"]),
            wall=time.perf_counter() - t0,
            tol=prob.tolerance,
        )
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print(
            s,
            rec["sobol_seeds"],
            rec["func_count"],
            f"{rec['true_error']:.3g}",
            f"{rec['wall']:.1f}s",
            flush=True,
        )
