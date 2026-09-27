"""B7 verifier, K1: MATLAB's seed mod(prod(uint64(num2str(u0))), 997) + 1,
with prod in double (MATLAB's documented default for an integer input),
under three readings of mod for a double p beyond flintmax (2^53):
 exact     - the exact remainder of the double p (as C's fmod),
 compens.  - the quotient p/997 counted as an integer when it is within
             eps of one, which gives 0 (MathWorks' description of mod's
             round-off compensation),
 naive     - p - floor(p/997)*997 in double.
How large p is for typical starts decides which reading matters."""
import math

import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)


def num2str_row(x):
    x = np.asarray(x, float)
    neg = int(np.any(x < 0))
    xmax = np.max(np.abs(x))
    if np.all(x == np.fix(x)):
        d = 1 if xmax == 0 else int(math.floor(math.log10(xmax))) + 1
        s = "".join(("%" + str(d + 2 + neg) + "d") % v for v in x)
    else:
        p = int(math.floor(math.log10(xmax))) if xmax > 0 else 0
        p = max(p + 5, 5)
        s = "".join(
            ("%" + str(p + 7 + neg) + "." + str(p) + "g") % v for v in x
        )
    return s.lstrip()


def seeds(u0):
    s = num2str_row(u0[: min(10, len(u0))])
    p = 1.0
    for c in s:
        p *= float(ord(c))
    q = p / 997.0
    exact = math.fmod(p, 997.0)
    comp = (
        0.0
        if abs(q - round(q)) <= np.spacing(q)
        else p - math.floor(q) * 997.0
    )
    naive = p - math.floor(q) * 997.0
    return s, p, exact + 1, comp + 1, naive + 1


g = 2.0**-10
rng = np.random.default_rng(5)
print(
    "p for typical starts, and the seeds under the three readings", flush=True
)
cases = [
    np.zeros(1),
    np.zeros(2),
    np.zeros(4),
    np.zeros(5),
    np.array([0.25]),
    np.array([0.2998046875]),
    np.array([0.25, -0.5]),
]
cases += [np.round(rng.uniform(-1, 1, D) / g) * g for D in (1, 1, 2, 2, 3, 10)]
for u0 in cases:
    s, p, e, c, n = seeds(u0)
    print(
        f"D={u0.size:2d} {s[:30]!r:34s} p={p:.3e} (> 2^53: {p > 2**53}, "
        f"p/997 > 2^53: {p / 997 > 2**53}) seed exact {e:.0f}, compens. "
        f"{c:.0f}, naive {n:.3g}",
        flush=True,
    )

print(
    "\nover 500 random grid starts: distinct seeds, and share with seed 1",
    flush=True,
)
for D in (1, 2, 3, 5, 10):
    out = [seeds(np.round(rng.uniform(-1, 1, D) / g) * g) for _ in range(500)]
    for k, name in ((2, "exact"), (3, "compens.")):
        vals = [o[k] for o in out]
        print(
            f"D={D:2d} {name:8s}: {len(set(vals)):3d} distinct, seed 1 in "
            f"{sum(v == 1 for v in vals)} of 500",
            flush=True,
        )
