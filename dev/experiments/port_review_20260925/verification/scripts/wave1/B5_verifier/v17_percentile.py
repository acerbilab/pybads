"""F12 (comparison): points above the 95th percentile removed, np.percentile (linear) vs MATLAB prctile1."""
import numpy as np
from matlab_transcriptions import prctile1

r = np.random.default_rng(0)
for n in (5, 10, 19, 20, 21, 49, 50, 100):
    y = r.standard_normal(n)
    print(
        f"n={n:3d}: removed by np.percentile(linear) {int(np.sum(y > np.percentile(y, 95)))}, "
        f"by prctile1 {int(np.sum(y > prctile1(y, 95)))}, by np.percentile(hazen) {int(np.sum(y > np.percentile(y, 95, method='hazen')))}"
    )
