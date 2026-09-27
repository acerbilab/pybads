"""W4-21: contraints_check at 81385ac against the transcription of uCheck.m
(ucheck_ref.py, MATLAB's round), on random sets on grids coarser and finer
than a bin, and on the edge cases of the rounding."""
from types import SimpleNamespace

import hdr  # noqa
import numpy as np
from ucheck_ref import mround, ucheck

from pybads.function_logger.constraints_check import contraints_check
from pybads.rounding import round_half_away
from pybads.search.grid_functions import force_to_grid

tol_mesh = 2.0**-19
t = tol_mesh / 2


def fl_of(X):
    D = X.shape[1]
    return SimpleNamespace(
        X=np.vstack([X, np.full((3, D), np.nan)]), X_max_idx=len(X) - 1
    )


def bins(A):
    return sorted(map(tuple, mround(A / t) + 0.0))


# 1. round_half_away against an exact MATLAB round (fractions)
from fractions import Fraction


def exact_round(q):
    fq = Fraction(q)
    a = abs(fq)
    n = int(a)  # floor
    if a - n >= Fraction(1, 2):
        n += 1
    return float(n if fq >= 0 else -n)


edge = [
    0.5,
    -0.5,
    1.5,
    -1.5,
    2.5,
    -2.5,
    0.49999999999999994,
    -0.49999999999999994,
    np.nextafter(0.5, 1),
    -np.nextafter(0.5, 1),
    2.0**52 + 1,
    -(2.0**52 + 1),
    2.0**52 - 0.5,
    -(2.0**52 - 0.5),
    4503599627370495.5,
    1e-320,
    -1e-320,
    0.0,
    -0.0,
]
rng = np.random.default_rng(1)
vals = np.concatenate(
    [
        edge,
        rng.uniform(-1e6, 1e6, 20000),
        np.round(rng.uniform(-1e4, 1e4, 20000)) + 0.5,
        rng.integers(-(2**20), 2**20, 20000)
        * 2.0 ** -rng.integers(1, 5, 20000),
    ]
)
bad = [v for v in vals if round_half_away(v) != exact_round(v)]
print(
    f"round_half_away vs exact half-away rounding: {len(vals) - len(bad)}/{len(vals)} equal; bad {bad[:5]}"
)
print(
    "np.nan ->",
    round_half_away(np.nan),
    " inf ->",
    round_half_away(np.inf),
    round_half_away(-np.inf),
)


# force_to_grid still uses it, and equals the expression it had at 8c8d6f8
def ftg_old(x, s, tol=None):
    if tol is None:
        tol = s
    frac, r = np.modf(x / tol)
    return tol * (r + np.sign(frac) * (np.abs(frac) >= 0.5))


X = rng.uniform(-3, 3, (5000, 3))
for s in [2.0**-k for k in (3, 10, 20, 22)]:
    Xh = np.round(X / s * 2) / 2 * s  # many halves
    assert np.array_equal(force_to_grid(Xh, s), ftg_old(Xh, s), equal_nan=True)
    assert np.array_equal(force_to_grid(X, s, 0.1), ftg_old(X, s, 0.1))
print(
    "force_to_grid equals 8c8d6f8's expression on 40000 points (halves included)"
)

# 2. random sets, as the doublecheck's c2, on grids coarser and finer than a bin
for label, exps in [
    ("grid >= bin", range(8, 21)),
    ("grid < bin", range(21, 25)),
]:
    same_bins = same_rows_sorted = total = 0
    for trial in range(3000):
        D = int(rng.integers(1, 4))
        grid = 2.0 ** -int(rng.choice(list(exps)))
        base = rng.uniform(-0.5, 0.5, size=D)
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
        lb, ub = -np.ones((1, D)), np.ones((1, D))
        p = contraints_check(U, lb, ub, tol_mesh, fl_of(X_eval), True)
        m = ucheck(U, tol_mesh, X_eval, lb, ub, lb, ub, True)
        same_bins += bins(p) == bins(m)
        # with the candidates given sorted and unique, the representative is the same
        Us = np.unique(U, axis=0)
        ps = contraints_check(Us, lb, ub, tol_mesh, fl_of(X_eval), True)
        ms = ucheck(Us, tol_mesh, X_eval, lb, ub, lb, ub, True)
        same_rows_sorted += ps.shape == ms.shape and np.array_equal(ps, ms)
        total += 1
    print(
        f"{label}: same bins kept {same_bins}/{total}; identical rows with sorted input {same_rows_sorted}/{total}"
    )

# 3. edge cases, 1-D and 2-D
cases = {
    "half a bin either side of an evaluated 0": ([[0.5], [-0.5]], [[0.0]]),
    "halves k+0.5 for k in -4..4, evaluated at every integer bin": (
        [[k + 0.5] for k in range(-4, 5)],
        [[k] for k in range(-5, 6)],
    ),
    "halves, evaluated at nothing": (
        [[k + 0.5] for k in range(-4, 5)] + [[k] for k in range(-4, 5)],
        np.zeros((0, 1)),
    ),
    "largest double below 1/2, either side, evaluated 0": (
        [[0.49999999999999994], [-0.49999999999999994]],
        [[0.0]],
    ),
    "just above 1/2 either side, evaluated 1 and -1": (
        [[np.nextafter(0.5, 1)], [-np.nextafter(0.5, 1)]],
        [[1.0], [-1.0]],
    ),
    "candidate -0.3 bins, evaluated +0.3 bins (bins -0 and +0)": (
        [[-0.3]],
        [[0.3]],
    ),
    "candidate +0.3, evaluated -0.3": ([[0.3]], [[-0.3]]),
    "2-D -0 bins": ([[-0.3, 0.2], [0.2, -0.3]], [[0.3, -0.2]]),
    "candidates half a bin apart": (
        [
            [2.5, 2],
            [3, 2],
            [0, 3],
            [0.5, 3],
            [-1, -2],
            [-0.5, -2],
            [-1.5, 0],
            [-2, 0],
        ],
        np.zeros((0, 2)),
    ),
    "evaluated on halves": (
        [[4, 1], [5, 1], [-4, 1], [-5, 1]],
        [[4.5, 1], [-4.5, 1]],
    ),
}
for name, (Ub, Xb) in cases.items():
    Ub = np.array(Ub, float) * t
    Xb = np.array(Xb, float).reshape(-1, Ub.shape[1]) * t
    D = Ub.shape[1]
    lb, ub = -np.ones((1, D)), np.ones((1, D))
    p = contraints_check(Ub, lb, ub, tol_mesh, fl_of(Xb), True)
    m = ucheck(Ub, tol_mesh, Xb, lb, ub, lb, ub, True)
    print(
        f"{name}: bins equal {bins(p) == bins(m)}; port {(p / t).ravel().tolist()} uCheck {(m / t).ravel().tolist()}"
    )
