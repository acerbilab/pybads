import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
from pybads.search.es_search import ucov

rng = np.random.default_rng(0)
D, k = 3, 5
U = rng.normal(size=(k, D))
u = rng.normal(size=D)
mu = k + 0.5
w = np.log(mu + 0.5) - np.log(np.arange(1, np.floor(mu) + 1))
w /= w.sum()
S = U - u
C_py = ucov(U, u, w, np.ones(D), -np.ones(D), 1.0, None)
C_unweighted = (
    S.T @ S
)  # MATLAB: sum(bsxfun(@times,weights,ushift'*ushift),3) = sum(w)*ushift'*ushift
C_weighted = S.T @ (w[:, None] * S)  # what the name "weighted covariance" says
print("weights:", np.round(w, 3))
print(
    "Python ucov == unweighted S'S (MATLAB):", np.allclose(C_py, C_unweighted)
)
print(
    "Python ucov == weighted sum_i w_i s_i s_i':",
    np.allclose(C_py, C_weighted),
)
