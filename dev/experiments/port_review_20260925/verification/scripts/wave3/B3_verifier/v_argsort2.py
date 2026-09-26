"""B3-K9 diagnosis: what the ties at es_search.py:190 and :246 are."""
import sys

import numpy as np
import vhdr  # noqa

import pybads.search.es_search as es_mod
from pybads import BADS

rec = {190: [], 246: []}


class NP:
    def __getattr__(self, k):
        return getattr(np, k)

    def argsort(self, a, *args, **kw):
        fr = sys._getframe(1)
        line = fr.f_lineno
        d = np.argsort(a, *args, **kw)
        s = np.argsort(a, kind="stable")
        if line == 190 and not np.array_equal(d, s):
            L = fr.f_locals
            C = L["us_candidates"]
            N = min(C.shape[0], L["self"].lamb)
            first_same = np.array_equal(C[d[0]], C[s[0]])
            seq_same = np.array_equal(C[d[:N]], C[s[:N]])
            set_same = {tuple(r) for r in C[d[:N]]} == {
                tuple(r) for r in C[s[:N]]
            }
            # ties between distinct points among the selected
            zs = a[d[:N]]
            Cs = C[d[:N]]
            distinct_tie = any(
                zs[i] == zs[i + 1] and not np.array_equal(Cs[i], Cs[i + 1])
                for i in range(N - 1)
            )
            rec[190].append(
                (
                    L["i"],
                    first_same,
                    set_same,
                    seq_same,
                    distinct_tie,
                    np.isinf(zs).sum(),
                )
            )
        if line == 246 and not np.array_equal(d, s):
            L = fr.f_locals
            U = L["U"]
            k = int(np.floor(L["mu"] + 1))
            rec[246].append(
                ({tuple(r) for r in U[d[:k]]} == {tuple(r) for r in U[s[:k]]},)
            )
        return d


es_mod.np = NP()
D = 3
for name, f in (
    ("sphere", lambda x: float(np.sum(np.atleast_2d(x) ** 2))),
    (
        "rosenbrock",
        lambda x: float(
            np.sum(
                100
                * (np.atleast_2d(x)[:, 1:] - np.atleast_2d(x)[:, :-1] ** 2)
                ** 2
                + (1 - np.atleast_2d(x)[:, :-1]) ** 2
            )
        ),
    ),
):
    rec[190].clear()
    rec[246].clear()
    BADS(
        f,
        np.full(D, 1.5),
        np.full(D, -5.0),
        np.full(D, 5.0),
        np.full(D, -2.0),
        np.full(D, 2.0),
        options={"display": "off", "random_seed": 1, "max_fun_evals": 100},
    ).optimize()
    r = rec[190]
    print(
        f"{name}: line 190 differing calls {len(r)}; by generation {sorted(set(x[0] for x in r))}; "
        f"best point differs in {sum(not x[1] for x in r)}; selected set differs in {sum(not x[2] for x in r)}; "
        f"selected order (coords) differs in {sum(not x[3] for x in r)}; ties between distinct points in {sum(x[4] for x in r)}; "
        f"calls with inf z among selected: {sum(x[5] > 0 for x in r)}"
    )
    print(
        f"   line 246 differing calls {len(rec[246])}; selected set of best training points differs in {sum(not x[0] for x in rec[246])}"
    )
