import warnings

import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__, "gpyreg", gpyreg.__file__, flush=True)
from pybads.init_functions import init_sobol


def seed_of(u0):
    max_seed = 997
    str_seed = u0[0 : np.minimum(11, len(u0))].astype(np.uint64)
    s = np.array2string(str_seed)[1:-1]
    codes = np.array([ord(ch) for ch in s])
    return s, codes.dtype, int(np.mod(np.prod(codes), max_seed) + 1)


rng = np.random.default_rng(0)
for D in [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 20]:
    seeds = set()
    for _ in range(200):
        u0 = rng.uniform(-1, 1, D)
        seeds.add(seed_of(u0)[2])
    print(
        f"D={D}: distinct seeds over 200 random u0 in (-1,1)^D: {sorted(seeds)}",
        flush=True,
    )

# boundaries
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    for u in [
        np.array([-1.0, 0.3]),
        np.array([1.0, 0.3]),
        np.array([-1.5, 0.3]),
        np.array([-0.99, 0.3]),
    ]:
        print(u, "->", seed_of(u), flush=True)
    print("warnings:", [str(x.message) for x in w], flush=True)

# same design for different u0 and different rng
plb = -np.ones((1, 3))
pub = np.ones((1, 3))
a, na = init_sobol(
    np.array([0.1, -0.5, 0.7]),
    plb * 5,
    pub * 5,
    plb,
    pub,
    3,
    rng=np.random.default_rng(1),
)
b, nb = init_sobol(
    np.array([-0.9, 0.2, 0.0]),
    plb * 5,
    pub * 5,
    plb,
    pub,
    3,
    rng=np.random.default_rng(2),
)
print("second return value:", na, nb, "rows:", a.shape[0])
print("identical designs:", np.array_equal(a, b))
print(a)
