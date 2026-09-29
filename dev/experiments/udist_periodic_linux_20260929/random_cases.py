"""``udist`` with periodic variables against another version of it, bit for
bit, on random cases: 3000 at D from 1 to 20, and 100 at D from 129 to 300,
with points inside and outside the bounds, differences of exactly one
period, NaN and inf, the length scale as one number, a (D,) array or a
(1, D) array, and one point as ``u2``. Prints the number of identical cases
in each range of D (below 8, from 8 to 128 and above 128 terms, the ranges
of ``_pairwise_sum``) and, for the others, the largest relative difference
at each D; exits 1 unless every case is identical.

``--control`` replaces the installed ``udist``'s ``_pairwise_sum`` by a sum
of the terms in turn, the positive control: it differs from ``np.sum``'s
order from 8 terms on.

    python random_cases.py [--control] BASE_GRID_FUNCTIONS_PY
"""

import importlib.util
import sys
from collections import Counter

import numpy as np

import pybads.search.grid_functions as grid_functions


def in_turn(terms):
    total = np.zeros_like(terms[0])
    for term in terms:
        total += term
    return total


def case(rng, D, n_max):
    N = int(rng.integers(1, n_max))
    M = int(rng.integers(1, n_max))
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
    return U, u2, len_scale, lb, ub, scale, periodic


def compare(a, b):
    """0 for identical results, else their largest relative difference (inf
    when they differ in shape or in which distances are finite)."""
    if a.shape == b.shape and np.array_equal(a, b, equal_nan=True):
        return 0.0
    finite = np.isfinite(a)
    if a.shape != b.shape or not np.array_equal(finite, np.isfinite(b)):
        return np.inf
    gap = np.abs(a[finite] - b[finite])
    return float(np.max(gap / np.maximum(a[finite], 1e-300), initial=0.0))


def main(args):
    control = "--control" in args
    base_path = [a for a in args if a != "--control"][0]
    spec = importlib.util.spec_from_file_location("base_grid", base_path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    if control:
        grid_functions._pairwise_sum = in_turn
    ranges = ("below 8", "8 to 128", "above 128")
    same, total = Counter(), Counter()
    worst = {}
    sweeps = [(np.random.default_rng(3), 3000, (1, 21), 40)]
    sweeps.append((np.random.default_rng(4), 100, (129, 301), 11))
    for rng, n_cases, (d_low, d_high), n_max in sweeps:
        for _ in range(n_cases):
            D = int(rng.integers(d_low, d_high))
            args = case(rng, D, n_max)
            with np.errstate(invalid="ignore"):
                gap = compare(base.udist(*args), grid_functions.udist(*args))
            where = ranges[(D >= 8) + (D > 128)]
            total[where] += 1
            if gap == 0.0:
                same[where] += 1
            else:
                worst[D] = max(worst.get(D, 0.0), gap)
    for where in ranges:
        print(f"D {where}: {same[where]} of {total[where]} cases identical")
    if worst:
        print("largest relative difference by D:", dict(sorted(worst.items())))
    return 0 if not worst else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
