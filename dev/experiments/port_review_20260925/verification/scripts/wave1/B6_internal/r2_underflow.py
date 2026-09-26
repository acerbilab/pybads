"""Q2: runs where the local targets leave the initial design's range far behind, so
that the re-centred mean prior lies many SDs outside the frozen mean bounds."""
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
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = orig_lgf(gp, *a, **k)
    g = out[0]
    D = g.D
    i = D + 3
    records.append(
        dict(
            refit=a[5],
            mu=g.hyper_priors["mu"][i],
            sd=g.hyper_priors["sigma"][i],
            lb=g.lower_bounds[i],
            ub=g.upper_bounds[i],
            nc=g.normalization_constants.copy(),
            m=g.get_hyperparameters(as_array=True)[0][i],
            hyp=g.get_hyperparameters(as_array=True)[0].copy(),
            lpost=None,
            ymin=float(np.min(g.y)),
            ymax=float(np.max(g.y)),
            warn=sorted({str(x.message)[:50] for x in w}),
        )
    )
    return out


bmod.local_gp_fitting = wrapped


def run(name, fun, x0, lb, ub, plb, pub, seed=3):
    records.clear()
    with warnings.catch_warnings(record=True) as wall:
        warnings.simplefilter("always")
        b = BADS(
            fun,
            x0,
            lb,
            ub,
            plb,
            pub,
            options=dict(display="off", random_seed=seed, max_fun_evals=200),
        )
        res = b.optimize()
    print(
        f"\n=== {name}: fval {res['fval']:.6g} x {np.round(res['x'], 4)} evals {res['func_count']}"
    )
    zs = [
        max((r["lb"] - r["mu"]) / r["sd"], (r["mu"] - r["ub"]) / r["sd"])
        for r in records
    ]
    ncm = [r["nc"][-1] for r in records]
    print(
        f"rebuilds {len(records)}, refits {sum(bool(r['refit']) for r in records)}; mean prior outside bounds in {sum(z > 0 for z in zs)}; worst z {max(zs):.4g}; mean normalization constant == 0 in {sum(c == 0 for c in ncm)} rebuilds ({sum(c == 0 and bool(r['refit']) for c, r in zip(ncm, records))} of them refits)"
    )
    for r, z in zip(records, zs):
        if r["refit"]:
            print(
                f"   refit: mean bounds [{r['lb']:.4g},{r['ub']:.4g}] prior N({r['mu']:.4g},{r['sd']:.3g}) z={z:.3g} nc={r['nc'][-1]:.3g} fitted m={r['m']:.4g} local y [{r['ymin']:.4g},{r['ymax']:.4g}] hyp={np.round(r['hyp'],2)} warn={r['warn']}"
            )
    print("  run warnings:", sorted({str(x.message)[:70] for x in wall})[:6])


D = 2
bump = lambda x: float(
    -100 * np.exp(-0.5 * np.sum((np.atleast_1d(x) - 4) ** 2) / 4)
)
run(
    "Gaussian well at 4, plausible [-1,1]",
    bump,
    np.zeros(D),
    np.full(D, -10.0),
    np.full(D, 10.0),
    np.full(D, -1.0),
    np.full(D, 1.0),
)
shift = lambda x: float(np.sum((np.atleast_1d(x) - 20) ** 2) / 100)
run(
    "shallow quadratic at 20, plausible [-1,1]",
    shift,
    np.zeros(D),
    np.full(D, -50.0),
    np.full(D, 50.0),
    np.full(D, -1.0),
    np.full(D, 1.0),
)
