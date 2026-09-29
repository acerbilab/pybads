"""``udist`` with periodic variables against another version of it, bit for
bit, on 3000 random cases: D from 1 to 20, points inside and outside the
bounds, differences of exactly one period, NaN and inf, the length scale as
one number, a (D,) array or a (1, D) array, and one point as ``u2``. Prints
the number of identical cases and, for the others, the largest relative
difference at each D; exits 1 unless every case is identical.

    python random_cases.py BASE_GRID_FUNCTIONS_PY
"""

import importlib.util
import sys

import numpy as np

from pybads.search.grid_functions import udist as new


def main(base_path):
    spec = importlib.util.spec_from_file_location("base_grid", base_path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    rng = np.random.default_rng(3)
    n_same = 0
    worst = {}
    for _ in range(3000):
        D = int(rng.integers(1, 21))
        N = int(rng.integers(1, 40))
        M = int(rng.integers(1, 40))
        lb = rng.uniform(-5, 0, (1, D))
        ub = lb + rng.uniform(0.1, 6, (1, D))
        spread = rng.choice([0.0, 0.5, 3.0]) * (ub - lb)
        U = rng.uniform(lb - spread, ub + spread, (N, D))
        u2 = rng.uniform(lb - spread, ub + spread, (M, D))
        if rng.random() < 0.3:
            u2 = u2[0]
        if rng.random() < 0.2:
            U[0] = lb[0]
            np.atleast_2d(u2)[0] = ub[0]
        if N > 3 and rng.random() < 0.2:
            U[1, 0] = np.nan
            U[2, -1] = np.inf
        periodic = rng.random((1, D)) < rng.uniform(0.1, 1)
        periodic[0, 0] |= not periodic.any()
        kind = rng.integers(3)
        len_scale = (
            rng.uniform(0.01, 5, D),
            rng.uniform(0.01, 5),
            rng.uniform(0.01, 5, (1, D)),
        )[kind]
        scale = rng.choice([1.0, 0.5, 2.0])
        args = (U, u2, len_scale, lb, ub, scale, periodic)
        with np.errstate(invalid="ignore"):
            a, b = base.udist(*args), new(*args)
        if a.shape == b.shape and np.array_equal(a, b, equal_nan=True):
            n_same += 1
        else:
            # inf when the two differ in which distances are finite
            finite = np.isfinite(a)
            if a.shape != b.shape or not np.array_equal(
                finite, np.isfinite(b)
            ):
                rel = np.inf
            else:
                gap = np.abs(a[finite] - b[finite])
                rel = np.max(gap / np.maximum(a[finite], 1e-300), initial=0.0)
            worst[D] = max(worst.get(D, 0.0), float(rel))
    print(f"{n_same} of 3000 cases identical")
    if worst:
        print("largest relative difference by D:", dict(sorted(worst.items())))
    return 0 if n_same == 3000 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
