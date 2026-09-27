import warnings

import gpyreg
import numpy as np
import scipy

import pybads

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
print("numpy", np.__version__, "scipy", scipy.__version__, flush=True)
import platform

print(platform.machine(), platform.system(), flush=True)
from pybads.init_functions import init_sobol


def py_seed(u0):
    # transcription of init_sobol.py:52-62
    max_seed = 997
    s = u0[0 : np.minimum(11, len(u0))].astype(np.uint64)
    s = np.array2string(s)[1:-1] if s.ndim == 1 else np.array2string(s)[2:-2]
    codes = np.array([ord(ch) for ch in s])
    return int(np.mod(np.prod(codes), max_seed) + 1), s, codes.dtype


warnings.simplefilter("always")
for u0 in [
    np.zeros(2),
    np.array([0.25, -0.5]),
    np.array([0.999, -0.999]),
    np.array([1.0, -1.0]),
    np.array([-1.0, 0.0]),
    np.array([-2.5, 3.7]),
    np.zeros(5),
    np.full(5, 0.3),
    np.zeros(12),
    np.full(12, 0.7),
]:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        seed, s, dt = py_seed(u0)
    print(
        u0.tolist(),
        "-> str",
        repr(s[:60]),
        "seed",
        seed,
        "dtype",
        dt,
        "warnings:",
        [str(x.message)[:60] for x in w],
        flush=True,
    )

# int32 product (what np.array of Python ints gives on Windows with NumPy < 2)
for D in [2, 3, 5, 8, 11]:
    s = " ".join(["0"] * min(11, D))
    c64 = np.array([ord(ch) for ch in s], dtype=np.int64)
    c32 = np.array([ord(ch) for ch in s], dtype=np.int32)
    print(
        "D",
        D,
        "seed int64",
        int(np.mod(np.prod(c64), 997) + 1),
        "seed int32",
        int(np.mod(np.prod(c32), 997) + 1),
        flush=True,
    )

# Does Sobol(seed=...) warn? Is the design the same for two starts and two rng seeds?
plb = -np.ones((1, 3))
pub = np.ones((1, 3))
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    a, m = init_sobol(
        np.array([0.1, 0.2, -0.3]),
        None,
        None,
        plb,
        pub,
        3,
        rng=np.random.default_rng(1),
    )
    b, _ = init_sobol(
        np.array([-0.7, 0.9, 0.0]),
        None,
        None,
        plb,
        pub,
        3,
        rng=np.random.default_rng(2),
    )
    c, _ = init_sobol(
        np.array([-1.0, 0.9, 0.0]),
        None,
        None,
        plb,
        pub,
        3,
        rng=np.random.default_rng(2),
    )
print("warnings from Sobol:", [str(x.message)[:80] for x in w], flush=True)
print(
    "second return value (docstring: number of samples):",
    m,
    "rows:",
    a.shape[0],
    flush=True,
)
print(
    "same design for two starts inside the box:",
    np.array_equal(a, b),
    flush=True,
)
print(
    "same design when a start coordinate is -1:",
    np.array_equal(a, c),
    flush=True,
)
# sizes at default fun_eval_start = D
for D in range(1, 21):
    u, _ = init_sobol(
        np.zeros(D), None, None, -np.ones((1, D)), np.ones((1, D)), D
    )
    print(D, u.shape[0], end="; ")
print(flush=True)
