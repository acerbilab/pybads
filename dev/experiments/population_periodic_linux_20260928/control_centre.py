"""The control of the `periodic` suite's noisy configurations: the target
`periodic` at D = 3 with the minima of its periodic variables moved to the
middle of the period, half a period from the bounds, run with
`periodic_vars` ("on") and as a bounded problem ("off"), both noise kinds,
seeds 0-29, a budget of 1500. Run from the repository root, with gpyreg's
`periods` on `PYTHONPATH`:

    python -u dev/experiments/population_periodic_linux_20260928/control_centre.py OUT.json

writes one record per run to OUT.json (running the runs only if OUT.json
does not exist) and prints, per noise kind and arm, the median error, the
fraction solved (error below 0.1) and the median evaluations, and per
noise kind the paired signed-rank test of the log10 errors."""
import collections
import json
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

sys.path.insert(0, "dev/scripts")
D = 3


def run(args):
    import benchmark_targets as bt

    from pybads import BADS

    seed, noise, arm = args
    prob = bt.make_problem("periodic", D, noise=noise, seed=seed)
    h = (D + 1) // 2
    c = prob.x_min.copy()
    # The same offsets from the bounds, moved to the middle of the period
    c[:h] = np.pi + (c[:h] - np.round(c[:h] / (2 * np.pi)) * 2 * np.pi)

    def f_vec(X):
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
        where="centre",
        err=prob.f_true(x) - prob.f_min,
        per=float(np.sum(2 * (1 - np.cos(z[:h])))),
        non=float(np.sum(z[h:] ** 2)),
        evals=int(res["func_count"]),
    )


if __name__ == "__main__":
    out_path = sys.argv[1]
    if os.path.exists(out_path):
        out = json.load(open(out_path))
    else:
        jobs = [
            (s, n, a)
            for n in ("hetero", "homo")
            for a in ("on", "off")
            for s in range(30)
        ]
        with Pool(3) as p:
            out = p.map(run, jobs)
        json.dump(out, open(out_path, "w"))
    groups = collections.defaultdict(dict)
    for r in out:
        groups[(r["noise"], r["arm"])][r["seed"]] = r
    for (noise, arm), rs in sorted(groups.items()):
        e = np.array([r["err"] for r in rs.values()])
        ev = np.array([r["evals"] for r in rs.values()])
        print(
            f"{noise:6s} {arm:3s} error median {np.median(e):.3g},"
            f" solved {np.mean(e < 0.1):.2f}, evaluations {np.median(ev):.0f}"
        )
    for noise in ("hetero", "homo"):
        on, off = groups[(noise, "on")], groups[(noise, "off")]
        seeds = sorted(on)
        a = np.log10(np.array([on[s]["err"] for s in seeds]) + 1e-12)
        b = np.log10(np.array([off[s]["err"] for s in seeds]) + 1e-12)
        print(
            f"{noise:6s} paired log10 error ratio (on/off): median"
            f" {np.median(a - b):+.3f}, signed-rank p"
            f" {wilcoxon(a, b).pvalue:.2g}"
        )
