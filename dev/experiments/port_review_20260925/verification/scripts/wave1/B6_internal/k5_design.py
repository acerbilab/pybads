"""F2: the space-filling design of gp.fit when the mean prior's mass inside its bounds underflows."""
import warnings

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from gpyreg.f_min_fill import f_min_fill

warnings.simplefilter("ignore")
hp = {k: np.full(2, np.nan) for k in ("mu", "sigma", "df", "a", "b")}
hp["mu"][:] = [0.0, 100.0]
hp["sigma"][:] = [1.0, 1.0]
hp["df"][:] = [0, 0]
seen = []
X, y = f_min_fill(
    lambda h: (seen.append(h.copy()), float(np.sum(h**2)))[1],
    np.array([[0.5, 0.5]]),
    np.array([-5.0, 0.0]),
    np.array([5.0, 1.0]),
    np.array([-1.0, 0.0]),
    np.array([1.0, 1.0]),
    hp,
    6,
    "rand",
    rng=np.random.default_rng(0),
)
print("design points (second coordinate: prior N(100,1), bounds [0,1]):")
print(np.round(np.array(seen), 3))
