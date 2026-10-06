"""F1 (both reports) and F2 (internal): the constant mean's bounds against
its re-centred prior, at each rebuild of default runs; and the same runs with
the mean unbounded, as in MATLAB (gpdefBads.m:173)."""

import sys
import time
import warnings

import numpy as np
from common import ackley, bads_mod, gpt, rosenbrock, sphere

from pybads import BADS

orig_fit = bads_mod.local_gp_fitting
orig_hyp = gpt._gp_hyp


def unbounded_gp_hyp(*a, **k):
    gp, hyp0, n = orig_hyp(*a, **k)
    b = gp.get_bounds()
    b["mean_const"] = (np.array([-np.inf]), np.array([np.inf]))
    gp.set_bounds(b)
    return gp, hyp0, n


def run(
    fun, D, seed, x0, lb, ub, plb, pub, label, unbounded=False, verbose=True
):
    rec = []

    def wrapped(gp, *a, **k):
        out = orig_fit(gp, *a, **k)
        g = out[0]
        i = g.D + 3
        mu = g.hyper_priors["mu"][i]
        sd = g.hyper_priors["sigma"][i]
        lo, hi = g.lower_bounds[i], g.upper_bounds[i]
        z = max((lo - mu) / sd, (mu - hi) / sd, 0.0)
        m = g.get_hyperparameters(as_array=True)[0][i]
        rec.append(
            (
                mu,
                sd,
                lo,
                hi,
                z,
                g.normalization_constants[i],
                m,
                float(np.min(g.y)),
                float(np.max(g.y)),
                bool(a[5]),
            )
        )
        return out

    bads_mod.local_gp_fitting = wrapped
    if unbounded:
        gpt._gp_hyp = unbounded_gp_hyp
    try:
        t = time.time()
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            res = BADS(
                fun,
                x0,
                lb,
                ub,
                plb,
                pub,
                options=dict(
                    display="off", random_seed=seed, max_fun_evals=200
                ),
            ).optimize()
        dt = time.time() - t
    finally:
        bads_mod.local_gp_fitting = orig_fit
        gpt._gp_hyp = orig_hyp
    r = np.array(rec, dtype=float)
    n_div = sum("divide by zero" in str(w.message) for w in wl)
    out_idx = r[:, 4] > 0
    zero = r[:, 5] == 0
    refit = r[:, 9] == 1
    print(
        f"--- {label}{' [mean unbounded]' if unbounded else ''}: "
        f"fval={res['fval']:.4g} n={res['func_count']} "
        f"it={res['iterations']} time={dt:.1f}s rebuilds={len(r)} "
        f"refits={int(refit.sum())}"
    )
    if verbose:
        print(
            f"   bounds of the mean: [{r[0,2]:.4g}, {r[0,3]:.4g}]"
            f" (constant over the run: "
            f"{np.all(r[:,2]==r[0,2]) and np.all(r[:,3]==r[0,3])})"
        )
        print(
            f"   prior centre outside the bounds at {int(out_idx.sum())}"
            f"/{len(r)} rebuilds; max z={r[:,4].max():.1f} SD; "
            f"normalization constant 0 at {int(zero.sum())} rebuilds "
            f"({int((zero & refit).sum())} refits); log(0) warnings: {n_div}"
        )
        j = np.where(refit)[0]
        for jj in list(j[:3]) + list(j[-3:]):
            mu, sd, lo, hi, z, c, m, ymin, ymax, _ = r[jj]
            print(
                f"   refit {jj:3d}: prior N({mu:.4g}, {sd:.3g}) "
                f"z={z:.1f} C={c:.3g} fitted m={m:.4g} "
                f"local y in [{ymin:.4g}, {ymax:.4g}]"
            )
    return res, r


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    cases = {
        "sphere2": (sphere, 2, 0, np.full(2, 0.7), -5, 5, -2, 2),
        "ackley6": (ackley, 6, 0, None, -32, 32, -8, 8),
        "rosen4_3": (
            lambda x: rosenbrock(np.atleast_1d(x) - 4),
            3,
            0,
            None,
            -10,
            10,
            -1,
            1,
        ),
    }
    for name, (f, D, seed, x0, lb, ub, plb, pub) in cases.items():
        if which != "all" and which != name:
            continue
        if x0 is None:
            x0 = np.random.default_rng(100 + seed).uniform(plb, pub, D)
        args = (
            f,
            D,
            seed,
            x0,
            np.full(D, lb * 1.0),
            np.full(D, ub * 1.0),
            np.full(D, plb * 1.0),
            np.full(D, pub * 1.0),
        )
        run(*args, label=f"{name} seed {seed}")
        run(*args, label=f"{name} seed {seed}", unbounded=True)
