"""Control: the `periodic` target with its periodic minima in the middle of
the period, where wrapping is irrelevant; periodic_vars on vs off."""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "dev/scripts")


def run(args):
    import benchmark_targets as bt

    from pybads import BADS

    seed, noise, arm, where = args
    D = 3
    prob = bt.make_problem("periodic", D, noise=noise, seed=seed)
    h = (D + 1) // 2
    c = prob.x_min.copy()
    if where == "centre":
        c[:h] = np.pi + (c[:h] - np.round(c[:h] / (2 * np.pi)) * 2 * np.pi)

    def f_vec(X, c=c):
        Z = np.atleast_2d(X) - c
        return np.sum(2.0 * (1.0 - np.cos(Z[:, :h])), axis=1) + np.sum(
            Z[:, h:] ** 2, axis=1
        )

    prob.f_vec = f_vec
    prob.x_min = c
    args_, opts = prob.bads_args()
    opts.update(display="off", max_fun_evals=1500, random_seed=seed)
    if arm == "off":
        opts["periodic_vars"] = None
    res = BADS(*args_, options=opts).optimize()
    x = np.ravel(res["x"])
    z = x - c
    return dict(
        seed=seed,
        noise=noise,
        arm=arm,
        where=where,
        err=prob.f_true(x) - prob.f_min,
        per=float(np.sum(2 * (1 - np.cos(z[:h])))),
        non=float(np.sum(z[h:] ** 2)),
        evals=int(res["func_count"]),
    )


if __name__ == "__main__":
    jobs = [
        (s, n, a, w)
        for w in ("centre",)
        for n in ("hetero", "homo")
        for a in ("on", "off")
        for s in range(30)
    ]
    with Pool(3) as p:
        out = p.map(run, jobs)
    json.dump(out, open(sys.argv[1], "w"))
    import collections

    g = collections.defaultdict(list)
    for r in out:
        g[(r["where"], r["noise"], r["arm"])].append(r)
    for k, rs in sorted(g.items()):
        e = np.array([r["err"] for r in rs])
        pe = np.array([r["per"] for r in rs])
        ev = np.array([r["evals"] for r in rs])
        print(
            k,
            f"err median {np.median(e):.3g} solved {np.mean(e < 0.1):.2f} periodic part {np.median(pe):.3g} evals {np.median(ev):.0f}",
            flush=True,
        )
