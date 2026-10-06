import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__, "gpyreg", gpyreg.__file__, flush=True)

from pybads import BADS


def f(x):
    # an ellipsoid whose minimum is off-centre
    x = np.asarray(x)
    c = np.array([0.6, -0.4, 0.3])[: x.size]
    return float(np.sum((np.arange(1, x.size + 1)) * (x - c) ** 2))


D = 3
lb = -5 * np.ones(D)
ub = 5 * np.ones(D)
plb = -2 * np.ones(D)
pub = 2 * np.ones(D)
designs = []
starts = []
for seed in range(20):
    b = BADS(
        f,
        None,
        lb,
        ub,
        plb,
        pub,
        options={"random_seed": seed, "display": "off"},
    )
    b._init_mesh_()
    fl = b.function_logger
    X = fl.X[: fl.Xn + 1]
    designs.append(X[1:].copy())
    starts.append(
        (
            tuple(np.round(X[0], 4)),
            tuple(np.round(b.u, 4)),
            bool(np.all(b.u == X[0])),
        )
    )
same = all(np.array_equal(designs[0], d) for d in designs)
print(
    "design rows per run:",
    designs[0].shape[0],
    "; identical across the 20 seeds:",
    same,
    flush=True,
)
print("design (u space):\n", designs[0])
n_from_design = sum(1 for s in starts if not s[2])
distinct = len(set(s[1] for s in starts))
print(
    f"start after the design is a design point in {n_from_design}/20 runs; distinct starts after the design: {distinct}",
    flush=True,
)
for s in starts[:6]:
    print(s)
