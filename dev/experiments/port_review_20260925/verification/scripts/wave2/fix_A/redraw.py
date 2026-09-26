import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
f = lambda x: float(np.sum(np.asarray(x) ** 2))
nbc = lambda x: np.sum(np.atleast_2d(x) ** 2, 1) > 1
out = []
for s in range(20):
    b = BADS(
        f,
        None,
        -np.ones((1, 2)) * 2,
        np.ones((1, 2)) * 2,
        -np.ones((1, 2)),
        np.ones((1, 2)),
        non_box_cons=nbc,
        options={"display": "off", "random_seed": s},
    )
    first = np.random.default_rng(s).uniform(-1, 1, size=(1, 2))
    out.append((s, bool(nbc(b.x0)[0]), bool(np.allclose(b.x0, first))))
print("infeasible x0:", [s for s, v, _ in out if v])
print("x0 != first draw:", [s for s, _, same in out if not same])
