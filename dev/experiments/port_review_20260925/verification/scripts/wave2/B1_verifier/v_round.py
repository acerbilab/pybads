"""New item (unverified beyond this): force_to_grid uses np.round (halves
to even); MATLAB's force2grid.m uses round (halves away from zero). A start
point exactly half a search-grid step off the grid."""
import numpy as np
from common import m_setup, mround, py_setup

from pybads.search.grid_functions import force_to_grid

h = 2.0**-10
for v in (0.5 * h, 1.5 * h, 2.5 * h, -0.5 * h):
    print(
        f"u = {v / h:+.1f} h: numpy -> {force_to_grid(v, h) / h:+.0f} h, "
        f"MATLAB -> {mround(v / h) * h / h:+.0f} h"
    )
x0, lb, ub, plb, pub = [1.0], [-4096.0], [4096.0], [-2048.0], [2048.0]
u, x, b = py_setup(x0, lb, ub, plb, pub)
um, xm, t = m_setup(x0, lb, ub, plb, pub)
print(
    "x0 = 1 in plausible box [-2048, 2048]: u = x0/2048 = 2^-11 = h/2;",
    "PyBADS u0 =",
    u,
    " MATLAB u0 =",
    um,
    "-> evaluated x0:",
    b.var_transf.inverse_transf(np.atleast_2d(u)).ravel(),
    "vs",
    t["ginv"](um),
)
