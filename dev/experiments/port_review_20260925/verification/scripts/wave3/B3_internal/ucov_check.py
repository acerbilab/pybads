import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
from pybads.search.es_search import ucov

rng = np.random.default_rng(0)
nb, D = 6, 2
U = rng.normal(size=(nb, D))
u = np.zeros(D)
w = np.log(nb - 1 + 0.5) - np.log(np.arange(1, nb))
w /= w.sum()  # nb-1 weights, as _initialize_ builds them
C_port = ucov(U, u, w, np.ones(D), -np.ones(D), np.ones(D), None)
C_unw = (U - u).T @ (U - u)
wp = np.r_[w, 0.0]  # weights on the first nb-1 rows
C_w = (U - u).T @ (wp[:, None] * (U - u))
print(
    "port ucov:\n",
    C_port,
    "\nunweighted sum over all rows:\n",
    C_unw,
    "\nweighted (first nb-1 rows):\n",
    C_w,
)
print("port == unweighted:", np.allclose(C_port, C_unw))
