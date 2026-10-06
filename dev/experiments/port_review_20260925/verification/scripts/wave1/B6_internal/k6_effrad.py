"""effective_radius = sqrt(alpha*(exp(1/alpha)-1)) against the kernel gpyreg computes."""
import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from gpyreg.covariance_functions import RationalQuadraticARD

cov = RationalQuadraticARD()
print(
    " log_alpha  r_code   k_gpyreg(r_code)  (1+r^2/a)^-a at r_code   r where k_gpyreg = exp(-1/2)"
)
for la in (-5, -2, -1, 0, 1, 2, 3, 5):
    a = np.exp(la)
    r = np.sqrt(a * (np.exp(1 / a) - 1)) if la > -5 else np.inf
    if not np.isfinite(r):
        print(f"{la:9d}  overflow")
        continue
    k = cov.compute(
        np.array([0.0, 0.0, la]), np.array([[0.0]]), np.array([[r]])
    )[0, 0]
    r_half = np.sqrt(2 * a * (np.exp(1 / (2 * a)) - 1))
    print(
        f"{la:9d}  {r:8.4g}  {k:10.4f}        {(1 + r**2 / a) ** (-a):10.4f}             {r_half:8.4g}"
    )
print(
    "exp(-1/2) =", round(np.exp(-0.5), 4), " exp(-1) =", round(np.exp(-1), 4)
)
with np.errstate(over="ignore"):
    print(
        "alpha at the lower bound exp(-5): alpha*(exp(1/alpha)-1) =",
        np.exp(-5) * (np.exp(np.exp(5)) - 1),
    )
