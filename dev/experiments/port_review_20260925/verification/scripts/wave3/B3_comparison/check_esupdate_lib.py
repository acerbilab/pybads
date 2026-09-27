import numpy as np


def esupdate_matlab(mu, lam):
    """Transcription of utils/ESupdate.m; returns selectmask as 1-based values."""
    tot = mu + lam
    s = 1.0 / np.sqrt(np.arange(1, tot + 1))
    w = np.ceil(s / s.sum() * lam).astype(int)
    nonzero = np.sum(w > 0)
    while (w.sum() - lam) > nonzero:
        w = np.maximum(0, w - 1)
        nonzero = np.sum(w > 0)
    delta = w.sum() - lam
    last = np.flatnonzero(w > 0)[-1] + 1  # 1-based
    lo = max(1, last - delta + 1)
    w[lo - 1 : last] -= 1
    cw = np.cumsum(w) - w + 1  # 1-based positions
    idx = np.zeros(cw.max(), dtype=int)  # idx(cw) = 1 grows idx to max(cw)
    idx[cw - 1] = 1
    selectmask = np.cumsum(idx[:-1])  # cumsum(idx(1:end-1))
    return selectmask, w
