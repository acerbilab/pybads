"""B7 verifier: (1) x0 = plb given explicitly, inside the hard bounds:
u0 = -1 exactly (the undefined cast without a hard bound); (2) periodic_vars
[] and an out-of-range index with a random x0; (3) MATLAB funlogger's ring
(funlogger.m:120-121), transcribed."""
import warnings

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
f = lambda x: float(np.sum(np.atleast_2d(x) ** 2))  # noqa: E731
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    b = BADS(
        f,
        np.array([[-2.0, 0.5]]),
        -5 * np.ones((1, 2)),
        5 * np.ones((1, 2)),
        -2 * np.ones((1, 2)),
        2 * np.ones((1, 2)),
        options={"display": "off", "random_seed": 0},
    )
print(
    f"(1) x0 = plb = -2 inside [-5, 5]: u0 {b.u.tolist()}, u0[0] == -1: "
    f"{b.u[0] == -1.0}, cast {b.u.astype(np.uint64).tolist()}",
    flush=True,
)
for pv, x0 in (
    ([], np.array([[0.1, 0.2]])),
    ([5], None),
    ([5], np.array([[0.1, 0.2]])),
):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            BADS(
                f,
                x0,
                -5 * np.ones((1, 2)),
                5 * np.ones((1, 2)),
                -2 * np.ones((1, 2)),
                2 * np.ones((1, 2)),
                options={"display": "off", "periodic_vars": pv},
            )
        print(f"(2) periodic_vars {pv}, x0 {x0}: accepted", flush=True)
    except Exception as e:  # noqa: BLE001
        print(
            f"(2) periodic_vars {pv}, x0 {'random' if x0 is None else 'given'}"
            f": {type(e).__name__}: {str(e)[:70]}",
            flush=True,
        )
nmax, Xn, Xmax, rows = 5, 0, 0, []
for _ in range(9):
    Xn = max(1, (Xn + 1) % nmax)
    Xmax = min(Xmax + 1, nmax)
    rows.append(Xn)
print(
    f"(3) MATLAB ring, nmax {nmax}: rows written {rows}, Xmax {Xmax} (row "
    f"{nmax} never written, read by U(1:Xmax))",
    flush=True,
)
