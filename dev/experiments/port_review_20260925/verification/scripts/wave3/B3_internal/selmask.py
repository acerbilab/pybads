import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
from pybads.search.es_search import ESSearchELL

opts = {
    "poll_mesh_multiplier": 2.0,
    "es_start": 0.25,
    "n_search_iter": 2,
    "search_acq_fcn": ("acq_LCB", None),
    "es_beta": 1,
}
s = ESSearchELL(2048, 2048, opts, np.random.default_rng(0))


def w_of(mu, lamb):
    tot = mu + lamb
    sq = np.sqrt(np.arange(1, tot + 1))
    w = np.ceil((1.0 / sq) / np.sum(1.0 / sq) * lamb).astype(int)
    nonzero = np.sum(w > 0)
    while (np.sum(w) - lamb) > nonzero:
        w = np.maximum(0, w - 1)
        nonzero = np.sum(w > 0)
    delta = np.sum(w) - lamb
    last = np.argwhere(w > 0)[-1].item()
    st = max(0, last - int(delta) + 1)
    w[st : last + 1] -= 1
    return w


for mu, lamb in [(3, 6), (2048, 2048), (1500, 2048), (10, 2048)]:
    m = s._get_selection_idx_mask_(mu, lamb)
    w = w_of(mu, lamb)
    ll = min(lamb, mu)
    counts = np.bincount(m[:ll], minlength=5)
    print(
        f"mu={mu} lamb={lamb}: len(mask)={len(m)} sum(w)={w.sum()} nonzero(w)={np.sum(w>0)} w[:6]={w[:6].tolist()} "
        f"mask[:12]={m[:12].tolist()} max(mask)={m.max()} used ll={ll}: offspring of parents 0..5 = {counts[:6].tolist()} max parent used={m[:ll].max()}"
    )
