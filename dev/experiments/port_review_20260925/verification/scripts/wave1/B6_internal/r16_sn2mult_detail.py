"""Q3 detail: at GP states with sn2_mult > 1, the fitted noise SD, the effective one,
the output scale, and the spread of the local targets; and whether the fitted model
itself (noise not inflated) has a positive definite training covariance."""
import warnings

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
import pybads.bads.bads as bmod
from pybads import BADS

rows = []
orig = bmod.local_gp_fitting


def w(*a, **k):
    out = orig(*a, **k)
    g = out[0]
    p = g.posteriors[0]
    if p.sn2_mult > 1:
        D = g.D
        h = p.hyp
        K = g.covariance.compute(h[: D + 2], g.X)
        sn2 = np.exp(2 * h[D + 2])
        ev = np.linalg.eigvalsh(K + sn2 * np.eye(len(K)))
        dmin = np.min(
            [
                np.min(np.linalg.norm(g.X[i] - np.delete(g.X, i, 0), axis=1))
                for i in range(len(g.X))
            ]
        )
        rows.append(
            (
                p.sn2_mult,
                h[D + 2],
                h[D],
                np.round(h[:D], 2),
                float(np.std(g.y)),
                float(np.ptp(g.y)),
                ev.min() / ev.max(),
                dmin,
                bool(a[6]),
            )
        )
    return out


bmod.local_gp_fitting = w


def rosen(z):
    z = np.atleast_1d(z)
    return float(np.sum(100 * (z[1:] - z[:-1] ** 2) ** 2 + (z[:-1] - 1) ** 2))


D = 4
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    b = BADS(
        rosen,
        np.full(D, 1.5),
        np.full(D, -5.0),
        np.full(D, 5.0),
        np.full(D, -2.0),
        np.full(D, 2.0),
        options=dict(display="off", random_seed=1, max_fun_evals=200),
    )
    r = b.optimize()
print(
    f"rosenbrock D=4 fval {r['fval']:.3g}; rebuilds with sn2_mult>1: {len(rows)}"
)
print(
    " mult  log_sn_fit  eff_sn   log_sf  log_ell              std(y_loc)  ptp(y_loc)  eig min/max (fitted, no inflation)  min NN dist  refit"
)
for m, lsn, lsf, lell, sy, py, cond, dmin, rf in rows[:12]:
    print(
        f" {m:5.0f}  {lsn:8.3f}  {np.exp(lsn)*np.sqrt(m):8.3g}  {lsf:6.2f}  {str(lell):22s} {sy:9.3g}  {py:9.3g}   {cond:10.3g}   {dmin:9.3g}  {rf}"
    )
