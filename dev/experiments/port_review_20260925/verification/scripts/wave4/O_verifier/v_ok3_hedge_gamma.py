"""O-K3: hedge_gamma outside [0, 1/n]. (1) ESSearchHedge.__call__ (with the
two ES searches replaced by stubs) against a transcription of searchHedge.m:
42-60, the probabilities and the choice frequencies over 4000 draws, for g
= [10, 0] and [10, 9] and hedge_beta = 1 (the default 1e-3/tol_fun).
(2) BADS accepts such values when it is created, and a short run goes."""
import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__, flush=True)
print("gpyreg:", gpyreg.__file__, flush=True)
import pybads.search.search_hedge as sh
from pybads import BADS


class Stub:
    def __init__(self, *a, **k):
        pass

    def __call__(self, u, *a, **k):
        return u, np.zeros(1)


sh.ESSearchWM = Stub
sh.ESSearchELL = Stub


def matlab_p(g, beta, gamma):
    n = len(g)
    p = np.exp(beta * (g - g.max())) / np.sum(np.exp(beta * (g - g.max())))
    return p * (1 - n * gamma) + gamma


def matlab_choice(p, r):
    idx = np.flatnonzero(r < np.cumsum(p))
    return idx[0] if idx.size else None


opts = {
    "hedge_gamma": None,
    "hedge_beta": 1.0,
    "hedge_decay": 0.1 ** (1 / 6),
    "n_search_iter": 2,
    "n_search": 4096,
}
for g0 in [np.array([10.0, 0.0]), np.array([10.0, 9.0])]:
    for gamma in [0.125, 0.5, 0.75, 1.0, 1.25, -0.1]:
        opts["hedge_gamma"] = gamma
        h = sh.ESSearchHedge(options_dict=opts, rng=np.random.default_rng(1))
        counts = np.zeros(2, int)
        phats = set()
        rr = np.random.default_rng(1)
        mcounts = np.zeros(2, int)
        for _ in range(4000):
            h.g = g0.copy()
            h(np.zeros((1, 2)), None, None, None, None, None)
            counts[h.chosen_hedge.item()] += 1
            phats.add(tuple(np.round(h.phat, 4)))
            c = matlab_choice(matlab_p(g0, 1.0, gamma), rr.random())
            mcounts[c] += 1
        print(
            f"g={g0} gamma={gamma}: p PyBADS {np.round(h.prob, 4)}, MATLAB "
            f"{np.round(matlab_p(g0, 1.0, gamma), 4)}; chosen PyBADS {counts} "
            f"(phat {sorted(phats)}), MATLAB rule {mcounts}",
            flush=True,
        )

rng_n = np.random.default_rng(3)
for gamma in [1.25, -0.5]:
    b = BADS(
        lambda x: float(np.sum(np.asarray(x) ** 2)),
        np.array([2.0, -1.0]),
        -5 * np.ones(2),
        5 * np.ones(2),
        -3 * np.ones(2),
        3 * np.ones(2),
        options={
            "display": "off",
            "random_seed": 0,
            "max_fun_evals": 60,
            "hedge_gamma": gamma,
        },
    )
    r = b.optimize()
    print(
        f"BADS with hedge_gamma={gamma}: created and ran; fval {r['fval']:.3g}, "
        f"final gains {np.round(b.search_es_hedge.g, 3)}, last p {np.round(b.search_es_hedge.prob, 3)}",
        flush=True,
    )
