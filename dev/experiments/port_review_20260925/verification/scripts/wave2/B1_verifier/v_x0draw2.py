"""B1 verifier, K5 (rerun): bounds that the effective bounds leave
untouched; PyBADS's random x0 against setupvars.m:83-85 in x as well as
in u. Then the same comparison where the effective bounds do move plb
(lb = 1e-3, plb = 1e-2), to show F1's effect on the draw."""
import numpy as np
from common import m_setup, py_setup

for label, lb, ub, plb, pub in [
    (
        "untouched",
        [1.0, -10.0, 0.5],
        [1e3, 10.0, 50.0],
        [2.0, -3.0, 1.0],
        [500.0, 5.0, 20.0],
    ),
    (
        "plb moved by F1",
        [1e-3, -10.0, 0.5],
        [1e3, 10.0, 50.0],
        [1e-2, -3.0, 1.0],
        [1e2, 5.0, 20.0],
    ),
]:
    lb, ub, plb, pub = map(np.array, (lb, ub, plb, pub))
    N = 200
    same_u = same_x = 0
    maxrel = 0.0
    xs_py, xs_m = [], []
    for s in range(N):
        upy, xpy, b = py_setup(
            None, lb, ub, plb, pub, options={"random_seed": s}
        )
        ru = np.random.default_rng(s).random(size=(1, 3)).ravel()
        umat, xmat, t = m_setup(
            np.full(3, np.nan), lb, ub, plb, pub, rand_u=ru
        )
        xev_py = b.var_transf.inverse_transf(np.atleast_2d(upy)).ravel()
        same_u += np.array_equal(upy, umat)
        same_x += np.array_equal(xev_py, xmat)
        maxrel = max(maxrel, np.max(np.abs(xev_py - xmat) / np.abs(xmat)))
        xs_py.append(xev_py)
        xs_m.append(xmat)
    xs_py, xs_m = np.array(xs_py), np.array(xs_m)
    print(
        f"{label}: log flags {b.var_transf.apply_log_t.ravel()} / "
        f"{t['logct']}; evaluated start: u identical {same_u}/{N}, "
        f"x identical {same_x}/{N}, max rel diff {maxrel:.2g}"
    )
    print(
        f"   variable 1 of the evaluated start: PY range "
        f"[{xs_py[:, 0].min():.4g}, {xs_py[:, 0].max():.4g}], MAT range "
        f"[{xs_m[:, 0].min():.4g}, {xs_m[:, 0].max():.4g}]"
    )
