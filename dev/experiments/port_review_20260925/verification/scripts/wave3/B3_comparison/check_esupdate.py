import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
from pybads.search.es_search import ESSearchWM


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


opts = {
    "poll_mesh_multiplier": 2.0,
    "es_start": 0.25,
    "n_search_iter": 2,
    "search_acq_fcn": ("acq_LCB", None),
    "es_beta": 1,
}
es = ESSearchWM(2048, 2048, opts)
for mu, lam in [(4, 4), (10, 10), (2048, 2048), (1500, 2048), (37, 2048)]:
    ml, w = esupdate_matlab(mu, lam)
    py = es._get_selection_idx_mask_(mu, lam)
    ll = min(lam, mu)
    ml0 = ml[:ll] - 1  # MATLAB parents, 0-based
    py0 = py[:ll]
    print(
        f"mu={mu} lam={lam}: len MATLAB mask={len(ml)}, len Python mask={len(py)}"
    )
    print("  MATLAB parents (0-based) first 12:", ml0[:12].tolist())
    print("  Python parents           first 12:", py0[:12].tolist())
    cm = np.bincount(ml0, minlength=5)[:6]
    cp = np.bincount(py0, minlength=5)[:6]
    print(
        "  offspring of ranks 0..5  MATLAB:",
        cm.tolist(),
        " Python:",
        cp.tolist(),
        " w[:6]:",
        w[:6].tolist(),
    )
    print(
        "  identical:",
        np.array_equal(ml0, py0),
        " Python == [0] + MATLAB[:-1]:",
        np.array_equal(py0, np.concatenate([[0], ml[: ll - 1]])),
    )
