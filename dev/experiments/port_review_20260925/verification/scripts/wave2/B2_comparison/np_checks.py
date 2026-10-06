import gpyreg
import numpy as np

import pybads

print(pybads.__file__)
print(gpyreg.__file__)
print("numpy", np.__version__)
a = np.empty(3)
try:
    a[0] = None
    print("assign None ->", a[0])
except Exception as e:
    print("assign None raises", type(e).__name__, e)
# object arrays with NaN in nanargmin / nanargmax
obj = np.full([4], None)
obj[0] = 1.0
obj[1] = np.nan
obj[2] = -5.0
obj[3] = 0.5
sd = np.full([4], None)
sd[:] = [0.1, np.nan, 0.2, 0.3]
q = obj + 3.09 * sd
print("q", q, q.dtype)
print("nanargmin q[1:]", np.nanargmin(q[1:]))
allnan = np.full([2], None)
allnan[:] = [np.nan, np.nan]
try:
    print(np.nanargmin(allnan))
except Exception as e:
    print("all-NaN nanargmin raises", type(e).__name__, e)
yv = np.empty(1)
yv[0] = 3.0
v = np.vstack((yv, 2.0))
print("vstack shape", v.shape, "mean", np.mean(v), "std", np.std(v, ddof=1))
