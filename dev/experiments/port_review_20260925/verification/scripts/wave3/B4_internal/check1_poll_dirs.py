"""Record the poll directions of default runs: n_max, and whether the
displacement from the incumbent is +-mesh_size along the coordinate axes."""
import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
import pybads.bads.bads as bm
from pybads import BADS

orig = bm.poll_mads_2n
records = []


def wrapped(dim_x, poll_scale, search_mesh_size, mesh_size, rng=None):
    B = orig(dim_x, poll_scale, search_mesh_size, mesh_size, rng=rng)
    vv = B * mesh_size * poll_scale
    n_max = max(1, round(search_mesh_size / mesh_size))
    unit = vv / mesh_size
    is_coord = np.allclose(
        np.sort(np.abs(unit), axis=1)[:, :-1], 0
    ) and np.allclose(np.max(np.abs(unit), axis=1), 1)
    records.append(
        (
            search_mesh_size,
            mesh_size,
            n_max,
            is_coord,
            np.ptp(np.log(np.ravel(poll_scale))),
        )
    )
    return B


bm.poll_mads_2n = wrapped


def rosen(x):
    x = np.atleast_2d(x)
    return float(
        np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2)
    )


def ellip(x):
    x = np.ravel(x)
    D = x.size
    return float(np.sum((10 ** (3 * np.arange(D) / (D - 1)) * x) ** 2))


for name, f, D in [("rosen", rosen, 3), ("ellip", ellip, 4)]:
    records.clear()
    lb = -5 * np.ones((1, D))
    ub = 5 * np.ones((1, D))
    plb = -2 * np.ones((1, D))
    pub = 2 * np.ones((1, D))
    x0 = np.full((1, D), 1.5)
    b = BADS(
        f,
        x0,
        lb,
        ub,
        plb,
        pub,
        options={"random_seed": 1, "display": "off", "max_fun_evals": 200},
    )
    r = b.optimize()
    nmax = set(rr[2] for rr in records)
    coord = all(rr[3] for rr in records)
    ratios = [rr[0] / rr[1] for rr in records]
    print(
        name,
        "polls:",
        len(records),
        "n_max values:",
        nmax,
        "all coordinate directions:",
        coord,
        "max search/mesh ratio:",
        max(ratios),
        "max log-range of poll_scale:",
        max(rr[4] for rr in records),
        "fval",
        r["fval"],
    )
