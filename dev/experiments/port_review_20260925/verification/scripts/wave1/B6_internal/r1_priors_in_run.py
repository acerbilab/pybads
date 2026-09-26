"""Q2: priors and bounds of the GP in real runs: after _gp_hyp and after each
local_gp_fitting; is a prior centre outside its bounds, and what is gpyreg's
normalization constant there?"""
import warnings

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
import pybads.bads.bads as bmod
import pybads.bads.gaussian_process_train as gpt
from pybads import BADS

records = []
orig_lgf = bmod.local_gp_fitting


def wrapped(gp, *a, **k):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = orig_lgf(gp, *a, **k)
    g = out[0]
    hp = g.hyper_priors
    lb, ub = g.lower_bounds, g.upper_bounds
    nc = g.normalization_constants
    hyp = g.get_hyperparameters(as_array=True)[0]
    refit = a[5] if len(a) > 5 else k.get("refit_flag")
    records.append(
        dict(
            n=len(g.y),
            refit=refit,
            mu=hp["mu"].copy(),
            sd=hp["sigma"].copy(),
            lb=lb.copy(),
            ub=ub.copy(),
            nc=nc.copy(),
            hyp=hyp.copy(),
            ymin=float(np.min(g.y)),
            ymax=float(np.max(g.y)),
            warn=[str(x.message)[:60] for x in w],
        )
    )
    return out


bmod.local_gp_fitting = wrapped


def run(name, fun, x0, lb, ub, plb, pub, opts):
    records.clear()
    b = BADS(
        fun,
        x0,
        lb,
        ub,
        plb,
        pub,
        options=dict(display="off", random_seed=3, max_fun_evals=200, **opts),
    )
    res = b.optimize()
    print(
        f"\n=== {name}: fval {res['fval']:.6g} x {np.round(res['x'], 4)} evals {res['func_count']}"
    )
    D = len(x0)
    names = [f"ell{i}" for i in range(D)] + ["sf", "alpha", "sn", "m"]
    n_out = {k: 0 for k in names}
    worst_z = {k: 0.0 for k in names}
    min_nc = {k: 1.0 for k in names}
    at_bound = {k: 0 for k in names}
    for r in records:
        for i, k in enumerate(names):
            z_lo = (r["lb"][i] - r["mu"][i]) / r["sd"][i]
            z_hi = (r["mu"][i] - r["ub"][i]) / r["sd"][i]
            z = max(z_lo, z_hi)
            if z > 0:
                n_out[k] += 1
            worst_z[k] = max(worst_z[k], z)
            min_nc[k] = min(min_nc[k], r["nc"][i])
            if np.isclose(r["hyp"][i], r["lb"][i], atol=1e-6) or np.isclose(
                r["hyp"][i], r["ub"][i], atol=1e-6
            ):
                at_bound[k] += 1
    print(
        f"rebuilds: {len(records)} (refits {sum(bool(r['refit']) for r in records)})"
    )
    print(
        "per hyperparameter: rebuilds with prior centre outside bounds / worst z beyond bound / min normalization constant / fits at a bound"
    )
    for k in names:
        print(
            f"  {k:6s} {n_out[k]:4d}  {worst_z[k]:10.3g}  {min_nc[k]:10.3g}  {at_bound[k]:4d}"
        )
    last = records[-1]
    i_m = D + 3
    print(
        f"  mean: bounds [{last['lb'][i_m]:.4g}, {last['ub'][i_m]:.4g}], prior N({last['mu'][i_m]:.4g}, {last['sd'][i_m]:.3g}), fitted m {last['hyp'][i_m]:.4g}, local y in [{last['ymin']:.4g}, {last['ymax']:.4g}]"
    )
    i_n = D + 2
    print(
        f"  noise: bounds [{last['lb'][i_n]:.4g}, {last['ub'][i_n]:.4g}], prior N({last['mu'][i_n]:.4g}, {last['sd'][i_n]:.3g}), fitted {last['hyp'][i_n]:.4g}"
    )
    ws = sorted({w for r in records for w in r["warn"]})
    print("  warnings during rebuilds:", ws[:5])
    return res


D = 2
sphere = lambda x: float(np.sum(np.atleast_1d(x) ** 2))
run(
    "sphere D=2 (default-like)",
    sphere,
    np.full(D, 0.5),
    np.full(D, -5.0),
    np.full(D, 5.0),
    np.full(D, -2.0),
    np.full(D, 2.0),
    {},
)
far = lambda x: float(np.sum((np.atleast_1d(x) - 50.0) ** 2))
run(
    "quadratic, optimum at 50 outside plausible [-1,1]",
    far,
    np.zeros(D),
    np.full(D, -100.0),
    np.full(D, 100.0),
    np.full(D, -1.0),
    np.full(D, 1.0),
    {},
)
lin = lambda x: float(
    100 * np.sum(np.atleast_1d(x)) + np.sum(np.atleast_1d(x) ** 2)
)
run(
    "linear+quadratic, optimum at -50",
    lin,
    np.zeros(D),
    np.full(D, -100.0),
    np.full(D, 100.0),
    np.full(D, -1.0),
    np.full(D, 1.0),
    {},
)
