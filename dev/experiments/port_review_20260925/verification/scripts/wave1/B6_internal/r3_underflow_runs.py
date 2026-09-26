"""Q2: BADS runs at default options in which the mean prior, re-centred at each
rebuild, leaves the mean bounds fixed on the initial design by so many SDs that
gpyreg's normalization constant is 0 (log posterior +inf, objective -inf)."""
import time
import warnings

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
import pybads.bads.bads as bmod
from pybads import BADS

records = []
orig_lgf = bmod.local_gp_fitting


def wrapped(gp, *a, **k):
    t0 = time.time()
    out = orig_lgf(gp, *a, **k)
    g = out[0]
    i = g.D + 3
    records.append(
        dict(
            refit=bool(a[5]),
            mu=g.hyper_priors["mu"][i],
            sd=g.hyper_priors["sigma"][i],
            lb=g.lower_bounds[i],
            ub=g.upper_bounds[i],
            nc=g.normalization_constants[i],
            m=g.get_hyperparameters(as_array=True)[0][i],
            t=time.time() - t0,
            q=(np.min(g.y), np.max(g.y)),
        )
    )
    return out


bmod.local_gp_fitting = wrapped


def run(name, fun, D, seed, x0=None, plb=-1.0, pub=1.0, lb=-10.0, ub=10.0):
    records.clear()
    x0 = np.zeros(D) if x0 is None else x0
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = BADS(
            fun,
            x0,
            np.full(D, lb),
            np.full(D, ub),
            np.full(D, plb),
            np.full(D, pub),
            options=dict(display="off", random_seed=seed, max_fun_evals=200),
        )
        res = b.optimize()
    zs = np.array(
        [
            max((r["lb"] - r["mu"]) / r["sd"], (r["mu"] - r["ub"]) / r["sd"])
            for r in records
        ]
    )
    nc0 = np.array([r["nc"] == 0 for r in records])
    rf = np.array([r["refit"] for r in records])
    tf = [r["t"] for r in records if r["refit"]]
    tf0 = [r["t"] for r in records if r["refit"] and r["nc"] == 0]
    print(
        f"{name} D={D} seed={seed}: fval {res['fval']:.6g} evals {res['func_count']} time {time.time()-t0:.1f}s | rebuilds {len(records)} refits {rf.sum()} | prior outside bounds {int((zs>0).sum())}, worst z {zs.max():.3g}, nc==0 in {int(nc0.sum())} rebuilds ({int((nc0 & rf).sum())} refits) | refit time mean {np.mean(tf):.2f}s, with nc==0 {np.mean(tf0) if tf0 else float('nan'):.2f}s"
    )
    if nc0.any():
        r = records[int(np.argmax(nc0))]
        print(
            f"     first nc==0: mean bounds [{r['lb']:.4g},{r['ub']:.4g}] prior N({r['mu']:.5g},{r['sd']:.3g}) local y [{r['q'][0]:.5g},{r['q'][1]:.5g}] fitted m {r['m']:.5g}"
        )
    return res


well = lambda x: float(
    -1000 * np.exp(-np.sum((np.atleast_1d(x) - 3) ** 2) / 8)
)
for D in (1, 2):
    for seed in (0, 1, 2):
        run("well(-1000 at 3)", well, D, seed)
