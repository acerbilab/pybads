import numpy as np

from pybads.search.es_search import ESSearch


def esupdate(mu, lamb):
    tot = mu + lamb
    s = 1.0 / np.sqrt(np.arange(1, tot + 1))
    w = np.ceil(s / np.sum(s) * lamb).astype(int)
    nonzero = np.sum(w > 0)
    while np.sum(w) - lamb > nonzero:
        w = np.maximum(0, w - 1)
        nonzero = np.sum(w > 0)
    delta = np.sum(w) - lamb
    last = np.flatnonzero(w > 0)[-1] + 1
    w[max(1, last - delta + 1) - 1 : last] -= 1
    cw = np.cumsum(w) - w + 1
    idx = np.zeros(np.max(cw), dtype=int)
    idx[cw - 1] = 1
    return np.cumsum(idx[:-1]), w


bad = 0
for mu in list(range(1, 60)) + [100, 1000, 1365, 2048, 4096]:
    for lamb in list(range(1, 60)) + [100, 1024, 1365, 2048, 4096]:
        M, w = esupdate(mu, lamb)
        w2 = w.copy()
        new = np.repeat(np.arange(len(w)), w)
        port = ESSearch._get_selection_idx_mask_(None, mu, lamb)
        ok = np.array_equal(new, M - 1) and len(M) == lamb
        bad += not ok
print("mismatches:", bad)
M, w = esupdate(1, 2048)
print((M - 1).sum(), M.sum())
