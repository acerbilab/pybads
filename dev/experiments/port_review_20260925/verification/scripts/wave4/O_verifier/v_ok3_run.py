"""O-K3, part 2: BADS created with hedge_gamma outside [0, 1/2]; record the
hedge's gamma, its probabilities and its choices at every search."""
import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__, flush=True)
print("gpyreg:", gpyreg.__file__, flush=True)
import pybads.search.search_hedge as sh
from pybads import BADS

orig = sh.ESSearchHedge.__call__
LOG = []


def call(self, *a, **k):
    out = orig(self, *a, **k)
    LOG.append(
        (
            self.chosen_hedge.item(),
            tuple(np.round(self.prob, 3)),
            tuple(np.round(self.g, 2)),
        )
    )
    return out


sh.ESSearchHedge.__call__ = call
for gamma in [0.125, 1.25, -0.5]:
    LOG.clear()
    b = BADS(
        lambda x: float(np.sum(np.asarray(x) ** 2 * np.array([1.0, 30.0]))),
        np.array([2.0, -1.0]),
        -5 * np.ones(2),
        5 * np.ones(2),
        -3 * np.ones(2),
        3 * np.ones(2),
        options={
            "display": "off",
            "random_seed": 0,
            "max_fun_evals": 100,
            "hedge_gamma": gamma,
        },
    )
    r = b.optimize()
    ch = np.array([c for c, _, _ in LOG])
    negp = sum(1 for _, p, _ in LOG if min(p) < 0)
    print(
        f"hedge_gamma={b.search_es_hedge.gamma}: searches {len(LOG)}, ES-wcm chosen {np.sum(ch == 0)}, "
        f"ES-ell {np.sum(ch == 1)}; searches with a negative probability {negp}; first 4 (choice, p, g) {LOG[:4]}; "
        f"fval {r['fval']:.3g}",
        flush=True,
    )
