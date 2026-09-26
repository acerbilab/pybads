import sys

import numpy as np

sys.path.insert(0, "pybads/testing/bads")
from test_gaussian_process_train import _initialized_bads

from pybads.bads.gaussian_process_train import get_grid_search_neighbors

bads, gp = _initialized_bads()
print(
    "len_scale",
    gp.temporary_data["len_scale"],
    "eff",
    gp.temporary_data["effective_radius"],
)
print(
    "n_train_min",
    bads.options["n_train_min"],
    "n_train_max",
    bads.options["n_train_max"],
    "buffer",
    bads.options["buffer_ntrain"],
)
logger = bads.function_logger
g = np.arange(-3, 4)
pts = np.array([(i, j) for i in g for j in g], dtype=float)
pts = pts[np.random.default_rng(0).permutation(len(pts))]
n = len(pts)
print("n", n, "X_max_idx before", logger.X_max_idx)
logger.X[:n] = pts
logger.Y[:n, 0] = np.arange(n)
logger.X_flag[:n] = True
logger.X_flag[n:] = False
logger.X_max_idx = n - 1
X, Y, S = get_grid_search_neighbors(
    logger, np.zeros((1, 2)), gp, bads.options, bads.optim_state
)
d = np.sum(pts**2, axis=1)
stable = np.argsort(d, kind="stable")
print("ntrain", X.shape[0])
print("returned idx", Y.ravel().astype(int))
print("stable idx  ", stable[: X.shape[0]])
print("equal", np.array_equal(Y.ravel().astype(int), stable[: X.shape[0]]))
print("np default == stable on d:", np.array_equal(np.argsort(d), stable))
