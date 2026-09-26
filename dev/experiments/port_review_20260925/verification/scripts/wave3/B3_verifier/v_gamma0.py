"""F6 (internal) / F8 (comparison): hedge_gamma = 0."""
import traceback

import numpy as np
import vhdr  # noqa

from pybads import BADS
from pybads.search.search_hedge import ESSearchHedge

calls = {"hedge": 0, "update": 0}
oc, ou = ESSearchHedge.__call__, ESSearchHedge.update_hedge


def c(self, *a, **k):
    calls["hedge"] += 1
    return oc(self, *a, **k)


def u(self, *a, **k):
    calls["update"] += 1
    print(
        f"  update_hedge #{calls['update']}: chosen={self.chosen_hedge.item()} prob={np.round(self.prob, 6).tolist()} len(u_search)={len(a[0])}"
    )
    return ou(self, *a, **k)


ESSearchHedge.__call__, ESSearchHedge.update_hedge = c, u
D = 3
try:
    BADS(
        lambda x: float(np.sum(np.atleast_2d(x) ** 2)),
        np.full(D, 2.0),
        np.full(D, -10.0),
        np.full(D, 10.0),
        np.full(D, -3.0),
        np.full(D, 3.0),
        options={
            "display": "off",
            "random_seed": 0,
            "max_fun_evals": 60,
            "hedge_gamma": 0,
        },
    ).optimize()
    print("no error")
except Exception as e:
    print(
        "searches started:",
        calls["hedge"],
        "| exception:",
        type(e).__name__,
        e,
    )
    print("".join(traceback.format_exc().splitlines(True)[-6:]))
