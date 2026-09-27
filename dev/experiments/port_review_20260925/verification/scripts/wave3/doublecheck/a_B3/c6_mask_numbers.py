"""W3-5's numbers: the old mask's sum at (mu, lambda) = (1, 2048), MATLAB's
1-based sum, and the correct 0-based sum; the offspring counts at
mu = lambda = 2048 and at (1, 2048)."""
import hdr  # noqa: F401
import numpy as np


def esupdate(mu, lam):
    tot = mu + lam
    s = 1.0 / np.sqrt(np.arange(1, tot + 1))
    w = np.ceil(s / np.sum(s) * lam).astype(int)
    nonzero = np.sum(w > 0)
    while np.sum(w) - lam > nonzero:
        w = np.maximum(0, w - 1)
        nonzero = np.sum(w > 0)
    delta = np.sum(w) - lam
    last = np.flatnonzero(w > 0)[-1] + 1
    w[max(1, last - delta + 1) - 1 : last] -= 1
    cw = np.cumsum(w) - w + 1
    idx = np.zeros(np.max(cw), dtype=int)
    idx[cw - 1] = 1
    return np.cumsum(idx[:-1]), w, cw


for mu, lam in [(1, 2048), (2048, 2048)]:
    sm, w, cw = esupdate(mu, lam)
    print(
        f"(mu, lambda) = ({mu}, {lam}): MATLAB 1-based sum {int(sm.sum())}, correct 0-based sum {int((sm - 1).sum())}, "
        f"offspring of parents 1-6 {w[:6].tolist()}"
    )
