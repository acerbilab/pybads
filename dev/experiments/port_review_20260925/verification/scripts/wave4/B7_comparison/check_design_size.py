import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
from scipy.stats.qmc import Sobol

from pybads.init_functions import init_sobol

for D in (2, 4, 8):
    row = []
    for fes in range(1, 2 * D + 2):
        u, _ = init_sobol(
            np.zeros(D), None, None, -np.ones((1, D)), np.ones((1, D)), fes
        )
        row.append(f"{fes}->{u.shape[0]}")
    print("D", D, ", ".join(row), flush=True)
# affine rank of the design without x0, n = D (no doubling) and n = 2D (doubling)
for D in (2, 4, 8, 16):
    m = int(np.log2(D))
    for seed in (1, 948, 7):
        a = Sobol(D, seed=seed).random_base2(m)
        b = Sobol(D, seed=seed).random_base2(m + 1)
        ra = np.linalg.matrix_rank(a - a.mean(0))
        rb = np.linalg.matrix_rank(b - b.mean(0))
        print(
            f"D={D} seed={seed}: affine rank of the first {2**m} points {ra}, of {2**(m+1)} points {rb}",
            flush=True,
        )
