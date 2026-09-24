"""Budget calibration for the benchmark suite: each configuration of the
``default`` suite run at ``max_fun_evals`` = 500 D (BADS's default) for a few
seeds, recording where each run ends and the incumbent's true error at each
poll iteration (hence the evaluations it needs to reach the tolerance).

Its run of 2026-09-24 (4 seeds, the suite's first 15 configurations) found
every run ending on BADS's own termination, which set the suite's budgets
(``benchmark_targets.py``). Usage, from the repository root::

    PYTHONPATH=dev/scripts/runs/gpyreg/v1.3.1 python -u dev/scripts/calibrate_budgets.py OUT.json [n_seeds]
"""
import os

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, "dev/scripts")
import benchmark_targets as bt  # noqa: E402
import gpyreg  # noqa: E402

from pybads import BADS  # noqa: E402

out_path = sys.argv[1]
seeds = range(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
print(gpyreg.__file__, flush=True)
rows = []
t_all = time.time()
for cfg in bt.suite_configs("default"):
    for seed in seeds:
        prob = cfg.make(seed=seed)
        args, options = prob.bads_args()
        options["max_fun_evals"] = 500 * cfg.D
        t0 = time.perf_counter()
        bads = BADS(*args, options=options)
        res = bads.optimize()
        wall = time.perf_counter() - t0
        hist_x = bads.iteration_history["x"]
        hist_n = bads.iteration_history["func_count"]
        traj = []
        for x, n in zip(hist_x, hist_n):
            if x is None or n is None:
                continue
            traj.append(
                (
                    int(n),
                    prob.f_true(np.asarray(x, float).ravel()) - prob.f_min,
                )
            )
        first = next((n for n, e in traj if e < prob.tolerance), None)
        final_err = (
            prob.f_true(np.asarray(res["x"], float).ravel()) - prob.f_min
        )
        row = {
            "label": cfg.label,
            "D": cfg.D,
            "seed": seed,
            "tolerance": prob.tolerance,
            "func_count": int(res["func_count"]),
            "message": str(res["message"]),
            "final_error": final_err,
            "evals_to_tol": first,
            "wall_s": wall,
            "trajectory": traj,
        }
        rows.append(row)
        print(
            f"{cfg.label:24s} seed {seed}: {wall:6.1f} s, evals {row['func_count']:5d},"
            f" final err {final_err:.3g}, evals to tol {first},"
            f" [{(time.time() - t_all) / 60:.1f} min] {row['message'][:60]}",
            flush=True,
        )
json.dump(rows, open(out_path, "w"), indent=1)
print(f"done in {(time.time() - t_all) / 60:.1f} min", flush=True)
