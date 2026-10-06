from common import *

from pybads.variable_transformer import VariableTransformer

rng = np.random.default_rng(1)
fails = 0
tot = 0
for mag in [1e8, 1e9, 1e10, 1e11, 1e12]:
    f = 0
    for k in range(200):
        lbv = -mag * rng.uniform(1, 9.99)
        plb = rng.uniform(-3, 0)
        pub = plb + rng.uniform(0.1, 3)
        try:
            VariableTransformer(
                1,
                np.array([[lbv]]),
                np.array([[10.0]]),
                np.array([[plb]]),
                np.array([[pub]]),
                np.full((1, 1), np.nan),
            )
        except ValueError:
            f += 1
    print(f"linear |lb|~{mag:g}: {f}/200 refused")
for ubv in [1e7, 3e7, 1e8, 3e8, 1e9, 3e9]:
    f = 0
    for k in range(200):
        u = ubv * rng.uniform(1, 2.99)
        try:
            VariableTransformer(
                1,
                np.array([[1e-3]]),
                np.array([[u]]),
                np.array([[1.0]]),
                np.array([[1e5]]),
                np.full((1, 1), np.nan),
            )
        except ValueError:
            f += 1
    print(f"log ub~{ubv:g}: {f}/200 refused")
