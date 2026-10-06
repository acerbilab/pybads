"""The calls of ``udist`` in one run of a configuration of the benchmark, one
BLAS thread: their number, time and shapes. Run from the repository root.

    python udist_calls.py [LABEL [SEED]]
"""
import collections
import os
import sys
import time

# One BLAS thread, set before NumPy loads (benchmark_targets imports it)
for var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[var] = "1"

sys.path.insert(0, "dev/scripts")
import benchmark_targets as bt
import numpy as np

import pybads.bads.bads as bads_mod
import pybads.bads.gaussian_process_train as gpt
import pybads.search.grid_functions as gf
from pybads import BADS

label = sys.argv[1] if len(sys.argv) > 1 else "periodic_D3_homo"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
orig = gf.udist
stats = collections.Counter()
times = collections.defaultdict(float)


def timed(U, u2, *args, **kwargs):
    t0 = time.perf_counter()
    out = orig(U, u2, *args, **kwargs)
    dt = time.perf_counter() - t0
    key = (
        np.atleast_2d(U).shape[0],
        np.atleast_2d(u2).shape[0],
        np.atleast_2d(U).shape[1],
    )
    stats[key] += 1
    times[key] += dt
    return out


for mod in (gf, gpt, bads_mod):
    if hasattr(mod, "udist"):
        setattr(mod, "udist", timed)
cfg = bt.find_config(label)
prob = cfg.make(seed=seed, budget_scale=1.0)
args, options = prob.bads_args()
t0 = time.perf_counter()
BADS(*args, options=options, **prob.bads_kwargs()).optimize()
wall = time.perf_counter() - t0
n = sum(stats.values())
tt = sum(times.values())
print(gf.__file__)
print(
    f"{label} seed {seed}: wall {wall:.2f} s, udist {n} calls, {tt:.3f} s ({1e3 * tt / n:.3f} ms/call)"
)
by_m = collections.Counter()
for (N, M, D), c in stats.items():
    by_m["M=1" if M == 1 else ("N=M" if N == M else "other")] += c
print(dict(by_m))
top = sorted(times.items(), key=lambda kv: -kv[1])[:8]
for k, v in top:
    print(k, stats[k], f"{v:.3f} s")
