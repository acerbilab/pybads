"""W3-1, W3-2: contraints_check against uCheck.m on random sets, on grids
coarser and finer than its bins (tol_mesh / 2), with evaluated points among
and near the candidates."""
from types import SimpleNamespace

import hdr  # noqa: F401
import numpy as np
from ucheck_ref import ucheck

from pybads.function_logger.constraints_check import contraints_check

rng = np.random.default_rng(0)
tol_mesh = 2.0**-19  # BADS's tol_mesh at default (1e-6 on the 2-grid)
res = {}
for label, exps in [
    ("grid >= bin", range(8, 21)),
    ("grid < bin", range(21, 25)),
]:
    same_rows = same_set = total = 0
    set_diff_examples = []
    for trial in range(3000):
        D = int(rng.integers(1, 4))
        grid = 2.0 ** -int(rng.choice(list(exps)))
        base = rng.uniform(-0.5, 0.5, size=D)
        # candidates within a few bins of each other, on the grid
        n = int(rng.integers(1, 40))
        span = int(rng.integers(1, 12))
        U = base + rng.integers(-span, span + 1, size=(n, D)) * grid
        U = np.round(U / grid) * grid
        n_eval = int(rng.integers(0, 15))
        X_eval = (
            np.vstack(
                [
                    U[rng.integers(0, n, size=n_eval // 2)],
                    base
                    + rng.integers(
                        -span, span + 1, size=(n_eval - n_eval // 2, D)
                    )
                    * grid,
                ]
            )
            if n_eval
            else np.zeros((0, D))
        )
        fl = SimpleNamespace(
            X=np.vstack([X_eval, np.full((3, D), np.nan)]),
            X_max_idx=len(X_eval) - 1,
        )
        lb, ub = -np.ones((1, D)), np.ones((1, D))
        p = contraints_check(U, lb, ub, tol_mesh, fl, True)
        m = ucheck(U, tol_mesh, X_eval, lb, ub, lb, ub, True)
        total += 1
        same_rows += p.shape == m.shape and np.array_equal(p, m)
        # compare the bins kept (MATLAB's bins), whatever the representative
        t = tol_mesh / 2
        from ucheck_ref import mround

        bp = sorted(map(tuple, mround(p / t) + 0.0))
        bm = sorted(map(tuple, mround(m / t) + 0.0))
        ok = bp == bm
        same_set += ok
        if not ok and len(set_diff_examples) < 2:
            set_diff_examples.append(
                (grid, U.tolist(), X_eval.tolist(), p.tolist(), m.tolist())
            )
    print(
        f"{label}: identical output {same_rows}/{total}; same bins kept "
        f"{same_set}/{total}"
    )
    for ex in set_diff_examples[:1]:
        grid, U, X, p, m = ex
        print(
            "  example: grid",
            grid,
            "\n   U/(tol/2)",
            (np.array(U) / (tol_mesh / 2)).tolist(),
            "\n   X/(tol/2)",
            (np.array(X) / (tol_mesh / 2)).tolist(),
            "\n   port/(tol/2)",
            (np.array(p) / (tol_mesh / 2)).tolist(),
            "\n   uCheck/(tol/2)",
            (np.array(m) / (tol_mesh / 2)).tolist(),
        )

# The simplest case: an evaluated point and a candidate half a bin apart
t = tol_mesh / 2
fl = SimpleNamespace(X=np.array([[0.0], [np.nan]]), X_max_idx=0)
for c in [0.5 * t, 1.0 * t, -0.5 * t]:
    p = contraints_check(
        np.array([[c]]), -np.ones((1, 1)), np.ones((1, 1)), tol_mesh, fl, True
    )
    m = ucheck(
        np.array([[c]]),
        tol_mesh,
        np.array([[0.0]]),
        -np.ones((1, 1)),
        np.ones((1, 1)),
        None,
        None,
        True,
    )
    print(
        f"evaluated 0, candidate {c / t:+.1f} bins: port keeps {p.shape[0]}, uCheck keeps {m.shape[0]}"
    )
fl = SimpleNamespace(X=np.array([[1.0 * t], [np.nan]]), X_max_idx=0)
p = contraints_check(
    np.array([[0.5 * t]]),
    -np.ones((1, 1)),
    np.ones((1, 1)),
    tol_mesh,
    fl,
    True,
)
m = ucheck(
    np.array([[0.5 * t]]),
    tol_mesh,
    np.array([[1.0 * t]]),
    -np.ones((1, 1)),
    np.ones((1, 1)),
    None,
    None,
    True,
)
print(
    f"evaluated +1 bin, candidate +0.5 bins: port keeps {p.shape[0]}, uCheck keeps {m.shape[0]}"
)
