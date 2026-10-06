"""The poll's p_less as written at bads.py:2273-2284 (8aecb6a), on the (n, 1)
columns that acq_fcn_lcb returns (gp.predict's shape), against MATLAB's
bads.m:862-869 (fpi sorted descending, the top min(nvars, n))."""
import gpyreg
import numpy as np
from scipy.special import erfc

import pybads

print(pybads.__file__)
print(gpyreg.__file__)
D = 3
f_target, suff = 0.0, 1e-3
# six poll points: the first has a large probability of improvement
f_mu = np.array([[-0.5], [6.0], [6.0], [6.0], [6.0], [6.0]])
fs = np.ones((6, 1))
gamma_z = (f_target - suff - f_mu) / fs  # line 2273-2277
f_pi = 0.5 * erfc(-gamma_z / np.sqrt(2))  # line 2279
f_pi_py = np.sort(f_pi)[::-1]  # line 2281
p_less_py = np.prod(
    1 - f_pi_py[0 : np.minimum(D + 1, len(f_pi_py))]
)  # 2282-2284
fpi_m = np.sort(np.ravel(f_pi))[::-1]  # MATLAB: sort(fpi,'descend')
p_less_m = np.prod(
    1 - fpi_m[: min(D, len(fpi_m))]
)  # prod(1-fpi(1:min(nvars,end)))
tol_poi = 1e-6 / D
print("PI per point:", np.ravel(f_pi).round(8))
print("PyBADS takes:", np.ravel(f_pi_py[: D + 1]).round(8))
print(f"p_less PyBADS={p_less_py:.10f} stop={p_less_py > 1 - tol_poi}")
print(f"p_less MATLAB={p_less_m:.10f} stop={p_less_m > 1 - tol_poi}")
