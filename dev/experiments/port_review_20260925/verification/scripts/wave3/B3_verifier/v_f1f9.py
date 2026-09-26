"""Coupling of F1's fix and F9: with evaluated points removed (MATLAB's uCheck), how often is an ES generation emptied without non_box_cons?"""
import numpy as np
import vhdr  # noqa

import pybads.bads.bads as bads_mod
import pybads.search.es_search as es_mod
from pybads import BADS

exec(
    open("v_repeats.py")
    .read()
    .split("stage = {")[0]
    .split("import vhdr  # noqa")[1]
)
cnt = {"gens": 0, "empty": 0, "calls": 0, "empty_ret": 0}
orig_call = es_mod.ESSearch.__call__


def cc_count(U, *a, **k):
    out = cc_matlab(U, *a, **k)
    cnt["gens"] += 1
    cnt["empty"] += len(out) == 0
    return out


def call(self, *a, **k):
    cnt["calls"] += 1
    us, z = orig_call(self, *a, **k)
    cnt["empty_ret"] += np.ndim(us) == 2 and len(us) == 0
    return us, z


es_mod.contraints_check = cc_count
bads_mod.contraints_check = cc_matlab
es_mod.ESSearch.__call__ = call
for D in (1, 2):
    for seed in range(4):
        for k in cnt:
            cnt[k] = 0
        BADS(
            lambda x: float(np.sum((np.atleast_2d(x) + 1) ** 2)),
            np.full(D, 2.5),
            np.zeros(D),
            np.full(D, 5.0),
            np.full(D, 0.1),
            np.full(D, 4.9),
            options={
                "display": "off",
                "random_seed": seed,
                "max_fun_evals": 60,
            },
        ).optimize()
        print(
            f"bound optimum D={D} seed={seed}: ES calls {cnt['calls']}, generations emptied {cnt['empty']}/{cnt['gens']}, ES calls returning empty {cnt['empty_ret']}"
        )
