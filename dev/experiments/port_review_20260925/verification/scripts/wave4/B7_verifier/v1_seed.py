"""B7 verifier: the seed of init_sobol at 0d866e8, its designs, the uint64
cast, and a transcription of MATLAB initSobol.m's seed (num2str, uint64,
prod, mod) under MATLAB's documented arithmetic."""
import math
import warnings

import gpyreg
import numpy as np

import pybads
from pybads.init_functions.init_sobol import init_sobol

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
print("numpy", np.__version__, flush=True)


def py_seed(u0, int_dtype=None):
    """The seed lines of init_sobol.py:55-62, with the product optionally in
    another integer type (int32 = NumPy 1.x's default integer on Windows)."""
    s = u0[0 : np.minimum(11, len(u0))].astype(np.uint64)
    s = np.array2string(s)[1:-1]
    codes = np.array([ord(ch) for ch in s], dtype=int_dtype)
    p = np.prod(codes, dtype=int_dtype)
    return int(np.mod(p, 997) + 1), s, p


print(
    "\n== 1. PyBADS seed for a start inside the plausible box ==", flush=True
)
rng = np.random.default_rng(1)
for D in range(1, 21):
    seeds = set()
    for _ in range(20):
        u0 = rng.uniform(-1, 1, D)
        u0 = np.round(u0 / 2**-10) * 2**-10  # on the search grid
        u0 = np.clip(u0, -1 + 2**-10, 1 - 2**-10)
        seeds.add(py_seed(u0)[0])
    s64, string, p64 = py_seed(np.zeros(D))
    s32, _, p32 = py_seed(np.zeros(D), np.int32)
    exact = 1
    for c in string:
        exact *= ord(c)
    print(
        f"D={D:2d} seeds over 20 random interior starts {sorted(seeds)}; "
        f"int64 product {int(p64)} (exact {exact}, wraps: {exact != int(p64)}); "
        f"int32 seed {s32} (product {int(p32)})",
        flush=True,
    )

print(
    "\n== 2. Designs for different starts and generators (D=3) ==", flush=True
)
lb, ub = -np.inf * np.ones((1, 3)), np.inf * np.ones((1, 3))
plb, pub = -np.ones((1, 3)), np.ones((1, 3))
d1, m1 = init_sobol(
    np.array([0.1, 0.2, -0.3]),
    lb,
    ub,
    plb,
    pub,
    3,
    rng=np.random.default_rng(0),
)
d2, m2 = init_sobol(
    np.array([-0.7, 0.9, 0.0]),
    lb,
    ub,
    plb,
    pub,
    3,
    rng=np.random.default_rng(12345),
)
d3, m3 = init_sobol(
    np.array([-1.0, 0.9, 0.0]),
    lb,
    ub,
    plb,
    pub,
    3,
    rng=np.random.default_rng(0),
)
d4, m4 = init_sobol(
    np.array([1.0, 0.9, 0.0]),
    lb,
    ub,
    plb,
    pub,
    3,
    rng=np.random.default_rng(0),
)
print(
    "interior starts give the same design:", np.array_equal(d1, d2), flush=True
)
print(
    "start with u=-1 gives a different design:",
    not np.array_equal(d1, d3),
    flush=True,
)
print(
    "start with u=+1 gives a different design:",
    not np.array_equal(d1, d4),
    flush=True,
)
print("second return value:", m1, "for", d1.shape[0], "rows", flush=True)

print("\n== 3. The float -> uint64 cast on this machine ==", flush=True)
for v in [-0.5, -0.99, -1.0, -1.5, -2.0, -2.5, 0.99, 1.0, 1.5]:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = np.array([v]).astype(np.uint64)[0]
    print(
        f"{v:6.2f} -> {int(r)}  warnings: {[str(x.message) for x in w]}",
        flush=True,
    )
for u0 in [
    np.array([-1.0, 0.5]),
    np.array([-1.5, 0.5]),
    np.array([-1.0]),
    np.array([-1.0, 0.0, 0.0]),
    np.array([0.0, 0.5]),
]:
    s, string, p = py_seed(u0)
    print(
        f"u0={u0.tolist()}: string {string!r}, int64 product {int(p)}, "
        f"seed {s}",
        flush=True,
    )
# What a saturating cast (negative -> 0, as AArch64's fcvtzu gives) would
# give: the string of an interior start
for D in (1, 2, 3):
    print(
        f"saturating cast, D={D}: seed of '0 ... 0' = {py_seed(np.zeros(D))[0]}",
        flush=True,
    )


print("\n== 4. MATLAB initSobol.m:10-12, transcribed ==", flush=True)


def num2str_row(x):
    """MATLAB num2str of a real row vector without a format: integers as
    '%{w}d' with w = digits + 2 (+1 with a negative); otherwise
    '%{p+7(+1)}.{p}g' with p = max(floor(log10(max|x|)) + 5, 5); then the
    leading blanks removed. For |x| <= 1, p = 5 in every variant of
    num2str.m that I know of."""
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


def matlab_seed(u0):
    s = num2str_row(u0[: min(10, len(u0))])
    codes = [ord(c) for c in s]  # uint64(char): the character codes
    # prod with the default outtype: accumulated in double for an integer
    # input, left to right
    pd = 1.0
    for c in codes:
        pd *= float(c)
    exact = 1
    for c in codes:
        exact *= c
    exact_in_double = float(exact) == pd and exact < 2**53 * 2 ** (
        exact.bit_length()
    )
    # mod: exact remainder of the double value (fmod), and the naive formula
    # x - floor(x/y)*y evaluated in double
    m_exact = math.fmod(pd, 997.0)
    m_naive = pd - math.floor(pd / 997.0) * 997.0
    # the reading in which prod keeps uint64 and saturates
    q = 1
    for c in codes:
        q = min(q * c, 2**64 - 1)
    return dict(
        s=s,
        double_exact=(pd == float(exact) and int(pd) == exact),
        seed_fmod=m_exact + 1,
        seed_naive=m_naive + 1,
        seed_exact_int=exact % 997 + 1,
        seed_sat=q % 997 + 1,
    )


g = 2.0**-10
cases = [
    np.zeros(1),
    np.zeros(2),
    np.zeros(3),
    np.zeros(10),
    np.array([0.25, -0.5]),
    np.round(np.array([0.3, 0.7]) / g) * g,
    np.round(np.array([0.3001, 0.7]) / g) * g,
    np.round(np.array([-0.2, 0.45]) / g) * g,
    np.array([-1.0, 0.5]),
]
rng = np.random.default_rng(3)
for D in (2, 3, 5, 10, 20):
    for _ in range(2):
        cases.append(np.round(rng.uniform(-1, 1, D) / g) * g)
for u0 in cases:
    r = matlab_seed(u0)
    print(
        f"D={u0.size:2d} num2str={r['s'][:44]!r:48s} double product exact: "
        f"{r['double_exact']!s:5s} seed (fmod) {r['seed_fmod']:.0f}, "
        f"(naive mod) {r['seed_naive']:.0f}, (exact integers) "
        f"{r['seed_exact_int']}, (saturating uint64) {r['seed_sat']}; "
        f"PyBADS {py_seed(u0)[0]}",
        flush=True,
    )
print(
    "mod(2^64-1, 997)+1 =",
    (2**64 - 1) % 997 + 1,
    "; mod(2^64, 997)+1 =",
    2**64 % 997 + 1,
    flush=True,
)

print("\n== 5. MATLAB seeds over random interior starts ==", flush=True)
rng = np.random.default_rng(4)
for D in (2, 3, 5, 10):
    seeds = [
        matlab_seed(np.round(rng.uniform(-1, 1, D) / g) * g)["seed_fmod"]
        for _ in range(200)
    ]
    print(
        f"D={D:2d}: {len(set(seeds))} distinct seeds (fmod reading) in 200 "
        f"random starts on the grid",
        flush=True,
    )
