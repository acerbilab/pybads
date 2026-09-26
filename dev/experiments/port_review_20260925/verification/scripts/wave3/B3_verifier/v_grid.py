"""B3-K8: force_to_grid rounds halves to even; MATLAB's round (force2grid.m) away from zero."""
import numpy as np
import vhdr  # noqa

from pybads import BADS
from pybads.init_functions.init_sobol import init_sobol
from pybads.search.grid_functions import force_to_grid


def force2grid_matlab(u, tol):
    q = u / tol
    return tol * np.sign(q) * np.floor(np.abs(q) + 0.5)


tol = 2.0**-10
x = np.array([0.5, 1.5, 2.5, -0.5, -1.5, 0.49, 0.51]) * tol
print("port  :", (force_to_grid(x, tol) / tol).tolist())
print("MATLAB:", (force2grid_matlab(x, tol) / tol).tolist())

b = BADS(
    lambda x: float(np.sum(np.atleast_2d(x) ** 2)),
    np.array([1.0, 3.0]),
    np.full(2, -4096.0),
    np.full(2, 4096.0),
    np.full(2, -2048.0),
    np.full(2, 2048.0),
    options={"display": "off", "random_seed": 0},
)
print(
    "x0 = [1, 3] in plausible box [-2048, 2048]: search_mesh_size",
    b.optim_state["search_mesh_size"],
    "| port starts at u0/tol =",
    (b.u / b.optim_state["search_mesh_size"]).tolist(),
    "| x start =",
    np.ravel(b.var_transf.inverse_transf(np.atleast_2d(b.u))).tolist(),
)
u_raw = np.array([1.0, 3.0]) / 2048
print(
    "  MATLAB would start at u0/tol =",
    (
        force2grid_matlab(u_raw, b.optim_state["search_mesh_size"])
        / b.optim_state["search_mesh_size"]
    ).tolist(),
)

# Exact halves in the Sobol design (seeded by u0) over many starting points
nh = 0
tot = 0
for D in (2, 3, 5):
    for k in range(200):
        u0 = np.random.default_rng(k).uniform(-1, 1, size=D)
        U, _ = init_sobol(
            u0,
            -np.ones(D),
            np.ones(D),
            -np.ones((1, D)),
            np.ones((1, D)),
            2 ** int(np.ceil(np.log2(10 * D))),
            rng=np.random.default_rng(k),
        )
        q = np.asarray(U) / tol
        nh += np.sum(np.abs(q - np.floor(q) - 0.5) == 0)
        tot += q.size
print(f"exact halves among {tot} Sobol design coordinates (grid 2^-10): {nh}")
