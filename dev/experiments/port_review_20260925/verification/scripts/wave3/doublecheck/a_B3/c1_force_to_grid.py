"""W3-14: force_to_grid against MATLAB's round (halves away from zero),
computed exactly in rational arithmetic, on many doubles."""
import math
import warnings
from fractions import Fraction

import hdr  # noqa: F401
import numpy as np

from pybads.search.grid_functions import force_to_grid


def matlab_round_exact(q):
    """MATLAB's round of the double q, exactly: sign(q) * floor(|q| + 1/2)
    in rationals (NaN and infinities returned as they are)."""
    if math.isnan(q) or math.isinf(q):
        return q
    f = Fraction(q)
    a = abs(f)
    r = math.floor(a + Fraction(1, 2))
    r = float(r)
    return math.copysign(r, q) if r != 0 else math.copysign(0.0, q)


def ref_force_to_grid(x, tol):
    # u = tol .* round(u ./ tol), with the division and the product in
    # doubles, as MATLAB does them
    q = x / tol
    return np.array([tol * matlab_round_exact(v) for v in np.ravel(q)])


rng = np.random.default_rng(0)
half = 0.5
cases = [
    0.5,
    -0.5,
    1.5,
    -1.5,
    2.5,
    -2.5,
    0.49,
    -0.49,
    0.51,
    3.0,
    -3.0,
    0.0,
    -0.0,
    np.nextafter(0.5, 0),
    -np.nextafter(0.5, 0),
    np.nextafter(0.5, 1),
    np.nextafter(1.5, 0),
    np.nextafter(1.5, 2),
    np.nextafter(2.5, 0),
    2.0**51 + 0.5,
    -(2.0**51 + 0.5),
    2.0**52 + 1,
    2.0**52 - 0.5,
    2.0**53 + 2,
    -(2.0**52 + 1),
    1e300,
    -1e300,
    5e-324,
    -5e-324,
    np.inf,
    -np.inf,
    np.nan,
    4503599627370495.5,
    -4503599627370495.5,
]
x = np.array(cases)
with warnings.catch_warnings():
    warnings.simplefilter("error")
    got = force_to_grid(x, 1.0)
ref = ref_force_to_grid(x, 1.0)
bad = [
    (c, g, r)
    for c, g, r in zip(cases, got, ref)
    if not (
        np.array_equal([g], [r], equal_nan=True)
        and math.copysign(1, g) == math.copysign(1, r)
    )
]
print("special cases:", len(cases), "mismatches (value or sign of zero):", bad)

# random doubles: halves of many magnitudes, random fractions, random bit
# patterns
n = 0
mism = 0
for tol in [1.0, 2.0**-10, 2.0**-42, 2.0**-20, 0.1, 3.0]:
    for kind in range(4):
        if kind == 0:
            k = rng.integers(-(2**40), 2**40, size=20000)
            q = k + 0.5
        elif kind == 1:
            q = rng.uniform(-1e6, 1e6, size=20000)
        elif kind == 2:
            bits = rng.integers(0, 2**63, size=20000, dtype=np.int64)
            q = bits.view(np.float64)
            q = q[np.isfinite(q)]
        else:
            q = rng.uniform(-1, 1, size=20000) * 2.0 ** rng.integers(
                -60, 60, size=20000
            )
        xx = q * tol
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )  # overflow of x/tol is MATLAB's too
            g = force_to_grid(xx, tol)
            r = ref_force_to_grid(xx, tol)
        eq = (g == r) | (np.isnan(g) & np.isnan(r))
        n += len(g)
        mism += int(np.sum(~eq))
print(f"random doubles: {n}, mismatches: {mism}")

# The old np.round and the agent's first formula at the largest double
# below one half
b = np.nextafter(0.5, 0)
print(
    "np.sign(q)*floor(|q|+0.5) at 0.49999999999999994:",
    np.sign(b) * np.floor(abs(b) + 0.5),
)
