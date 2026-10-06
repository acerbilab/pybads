import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
# Bookkeeping of the ES loop only (es_search.py:177-205 vs searchES.m:170-193), with the
# same z values of each iteration's new candidates.
rng = np.random.default_rng(0)
lam = 2048
n_iter = 5
zs = [rng.normal(loc=-0.3 * i, size=lam) for i in range(n_iter)]

# Python
fr_py = []
us_rows = lam
nold_py = None
zc = None
us_len = us_rows
for i, zn in enumerate(zs):
    nold = us_len
    zc = zn.copy() if i == 0 else np.append(zc, zn)
    N = min(len(zc), lam)
    z_idx = np.argsort(zc, kind="stable")
    ntest = min(len(zn), nold)
    n_new = np.sum(z_idx[0 : ntest + 1] > nold)
    us_len = N
    fr_py.append(n_new / ntest)
# MATLAB
fr_ml = []
zold = np.empty(0)
for i, zn in enumerate(zs):
    nold = len(zold)
    z = np.concatenate([zold, zn])
    N = min(len(z), lam)
    index = np.argsort(z, kind="stable") + 1  # 1-based
    ntest = min(len(zn), nold)
    nnew = np.sum(index[:ntest] > nold)
    zold = np.sort(z, kind="stable")[:N]
    fr_ml.append(nnew / ntest if ntest else np.nan)
print("iteration (1-based):", list(range(1, n_iter + 1)))
print("frac Python:", np.round(fr_py, 3))
print("frac MATLAB:", np.round(fr_ml, 3))
