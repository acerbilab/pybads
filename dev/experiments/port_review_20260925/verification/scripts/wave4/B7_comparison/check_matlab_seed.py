"""Transcription of MATLAB initSobol.m:9-15 (seed from num2str of u0(1:10)),
under the two readings of prod on a uint64 array (accumulated in double, or
natively in uint64 with saturation), against PyBADS's init_sobol seed."""
import math

import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)


def num2str_row(x):
    # MATLAB num2str for a real row vector, no format given (R2016+ logic):
    # integers: '%{d+2+neg}d'; otherwise '%{ndgt+7+neg}.{ndgt}g' with
    # ndgt = max(floor(log10(max|x|)) + 5, 5); leading blanks removed.
    x = np.asarray(x, float)
    neg = int(np.any(x < 0))
    xmax = np.max(np.abs(x))
    if np.all(x == np.fix(x)):
        d = 1 if xmax == 0 else int(math.floor(math.log10(xmax))) + 1
        s = "".join(("%" + str(d + 2 + neg) + "d") % v for v in x)
    else:
        ndgt = int(math.floor(math.log10(xmax))) if xmax > 0 else 0
        ndgt = max(ndgt + 5, 5)
        s = "".join(
            ("%" + str(ndgt + 7 + neg) + "." + str(ndgt) + "g") % v for v in x
        )
    return s.lstrip()


def matlab_seed(u0):
    s = num2str_row(u0[: min(10, len(u0))])
    codes = [ord(c) for c in s]
    # reading 1: prod in double (left to right), mod, +1
    p = 1.0
    for c in codes:
        p *= c
    seed_double = (math.fmod(p, 997) + 1) if math.isfinite(p) else float("nan")
    # reading 2: native uint64 with saturation
    q = 1
    for c in codes:
        q = min(q * c, 2**64 - 1)
    seed_sat = q % 997 + 1
    return s, seed_double, seed_sat


def py_seed(u0):
    s = u0[0 : min(11, len(u0))].astype(np.uint64)
    s = np.array2string(s)[1:-1]
    return int(np.mod(np.prod(np.array([ord(c) for c in s])), 997) + 1)


g = (
    2.0**-10
)  # search mesh at start (poll_mesh_multiplier 2, search_grid_number 10)
rng = np.random.default_rng(0)
cases = [
    np.array([0.0]),
    np.array([0.25]),
    np.array([-0.4990234375]),
    np.zeros(2),
    np.array([0.25, -0.5]),
    np.array([0.3, 0.7]) // g * g,
    np.zeros(3),
    (rng.uniform(-1, 1, 3) // g) * g,
    (rng.uniform(-1, 1, 6) // g) * g,
    (rng.uniform(-1, 1, 10) // g) * g,
    (rng.uniform(-1, 1, 10) // g) * g,
]
for u0 in cases:
    s, sd, ss = matlab_seed(u0)
    print(
        f"D={u0.size:2d} u0[:3]={np.round(u0[:3], 4).tolist()} num2str={s[:40]!r}... "
        f"MATLAB seed (double prod) {sd:.0f}, (saturating uint64) {ss}; PyBADS seed {py_seed(u0)}",
        flush=True,
    )
print("mod(2^64-1, 997)+1 =", (2**64 - 1) % 997 + 1, flush=True)
