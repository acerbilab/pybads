"""F4: ES selection mask vs utils/ESupdate.m; paired effect on the ES's best LCB."""
import numpy as np
import vhdr  # noqa
from capture import capture_states

from pybads.search.es_search import ESSearch, ESSearchELL, ESSearchWM


def esupdate_matlab(mu, lam):
    """ESupdate.m, returning the 1-based selectmask."""
    tot = mu + lam
    s = 1.0 / np.sqrt(np.arange(1, tot + 1))
    w = np.ceil(s / s.sum() * lam).astype(int)
    nonzero = np.sum(w > 0)
    while w.sum() - lam > nonzero:
        w = np.maximum(0, w - 1)
        nonzero = np.sum(w > 0)
    delta = w.sum() - lam
    last = np.flatnonzero(w > 0)[-1] + 1  # 1-based
    a = max(1, last - delta + 1)
    w[a - 1 : last] -= 1
    cw = np.cumsum(w) - w + 1  # 1-based start positions
    idx = np.zeros(cw.max(), dtype=int)  # idx(cw) = 1 grows to max(cw)
    idx[cw - 1] = 1
    return np.cumsum(idx[:-1])  # selectmask, 1-based values


port = ESSearch._get_selection_idx_mask_
for mu, lam in [(2048, 2048), (100, 2048), (7, 20), (1, 5), (3, 3)]:
    M = esupdate_matlab(mu, lam)
    P = port(None, mu, lam)
    ll = min(lam, mu)
    counts_m = np.bincount(M[:ll] - 1)[:6]
    counts_p = np.bincount(P[:ll])[:6]
    print(
        f"mu={mu:5d} lam={lam:5d}: len MATLAB {len(M)}, port {len(P)}; "
        f"port == [0] + MATLAB(1-based)[:-0]? {np.array_equal(P[1:], M[:len(P) - 1])}; "
        f"offspring of parents 0..5 (first ll): MATLAB {counts_m.tolist()} port {counts_p.tolist()}; "
        f"sum MATLAB 1-based {M.sum()}, port {P.sum()}, correct 0-based {(M - 1).sum()}"
    )


def mask_fixed(self, mu, lamb):
    return esupdate_matlab(mu, lamb) - 1


states = capture_states(D=3, seed=0, max_fun_evals=120)
res = {"ES-wcm": [], "ES-ell": []}
for st in states:
    for name, cls in (("ES-wcm", ESSearchWM), ("ES-ell", ESSearchELL)):
        out = []
        for maskfun in (port, mask_fixed):
            ESSearch._get_selection_idx_mask_ = maskfun
            zs = []
            for s in range(5):
                es = cls(
                    2048,
                    2048,
                    st["options"],
                    rng=np.random.default_rng(100 + s),
                )
                us, z = es(
                    st["u"],
                    None,
                    None,
                    st["func_logger"],
                    st["gp"],
                    st["optim_state"],
                    True,
                    None,
                )
                zs.append(float(np.ravel(z)[0]))
            out.append(np.array(zs))
        ESSearch._get_selection_idx_mask_ = port
        res[name].append(out)
for name, lst in res.items():
    p = np.concatenate([a for a, b in lst])
    m = np.concatenate([b for a, b in lst])
    print(
        f"{name}: {len(p)} paired searches; fixed mask lower best LCB in {np.mean(m < p):.2f}, "
        f"port lower in {np.mean(p < m):.2f}, ties {np.mean(p == m):.2f}; median (fixed - port) {np.median(m - p):.3g}; "
        f"median |port| {np.median(np.abs(p)):.3g}"
    )
