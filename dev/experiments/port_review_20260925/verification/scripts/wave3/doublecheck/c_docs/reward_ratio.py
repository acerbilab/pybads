"""Ratio of 1.1.0's hedge reward to MATLAB's, by gamma = (fval_old - f)/fs."""
import gpyreg
import numpy as np
from scipy.special import erfc

import pybads

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
for g in (2, 1, 0.5, 0, -0.5, -1, -2, -3, -4, -5):
    fpi = 0.5 * erfc(-g / np.sqrt(2))
    port = g * fpi + np.exp(-0.5 * g**2 / np.sqrt(2 * np.pi))
    mat = g * fpi + np.exp(-0.5 * g**2) / np.sqrt(2 * np.pi)
    print(f"gamma {g:5}: 1.1.0 / MATLAB = {port / mat:.4g}", flush=True)
