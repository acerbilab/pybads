"""count_repeats.py WORKTREE OUT LABEL... : run seeds 0-29 of each label with the worktree's PyBADS and count repeated evaluations.

A repeat is an evaluation at a point evaluated before: a duplicate row of the function log (levels 0 and 1), or, with
specify_target_noise, an evaluation merged into an existing row (func_count minus rows, less the final samples)."""
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

WT, OUT, LABELS = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.path.insert(0, WT + "/dev/scripts")


def one(args):
    label, seed = args
    sys.path.insert(0, WT + "/dev/scripts")
    import population as P

    orig = P._final
    seen = {}

    def fin(prob, bads, res, exc, wall):
        out = orig(prob, bads, res, exc, wall)
        fl = bads.function_logger
        n = fl.Xn + 1
        X = fl.X[:n][fl.X_flag[:n]]
        seen.update(
            rows=int(X.shape[0]),
            dup_rows=int(X.shape[0] - np.unique(X, axis=0).shape[0]),
            func_count=int(fl.func_count),
            final_samples=int(bads.options["noise_final_samples"])
            if bads.optim_state["uncertainty_handling_level"] > 0
            else 0,
            level=int(bads.optim_state["uncertainty_handling_level"]),
        )
        return out

    P._final = fin
    P.run_task(label, seed, {}, 1.0, OUT)
    merged = (
        seen["func_count"] - seen["rows"] - seen["final_samples"]
        if seen.get("level") == 2
        else 0
    )
    return (
        label,
        seed,
        seen.get("dup_rows", -1) + max(merged, 0),
        seen.get("func_count", -1),
    )


if __name__ == "__main__":
    import os

    os.makedirs(OUT, exist_ok=True)
    tasks = [(l, s) for l in LABELS for s in range(30)]
    res = {}
    with ProcessPoolExecutor(2) as ex:
        for label, seed, rep, fc in ex.map(one, tasks):
            res.setdefault(label, []).append((rep, fc))
    for label, v in res.items():
        r = np.array(v)
        print(
            f"{label}: repeats {int(r[:, 0].sum())} of {int(r[:, 1].sum())} evaluations, in {int((r[:, 0] > 0).sum())} of 30 runs"
        )
